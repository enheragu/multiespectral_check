"""Calibration report cache — derives and persists lightweight plot data.

Stores pre-computed quads and poses so the Calibration Report dialog does not
need to reload every corner file on each open.  The cache is invalidated when
the GUI version or the calibration ``updated_at`` timestamp changes.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common.yaml_utils import load_yaml, save_yaml

CACHE_FILENAME = ".calibration_report_cache.yaml"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_report_cache(
    dataset_path: Path,
    matrices: Dict[str, Any],
    file_metadata: Dict[str, Any],
) -> None:
    """Compute quads + poses from corner files and write the cache.

    Safe to call from the calibration solver — never raises (logs on failure).
    ``matrices`` maps channel names to their calibration dict
    (keys: ``camera_matrix``, ``distortion``, ``image_size``).
    ``file_metadata`` must contain at minimum ``pattern_size`` and ``updated_at``.
    """
    from config import APP_VERSION
    try:
        from backend.services.calibration_corners_io import load_corners_for_dataset
        all_corners = load_corners_for_dataset(dataset_path)
    except Exception:  # noqa: BLE001
        return

    if not all_corners:
        return

    pattern_size = file_metadata.get("pattern_size")
    square_size_m = _compute_square_size_m(file_metadata)
    dist_unit = "m" if square_size_m else "grid sq."

    lwir_quads = _extract_chessboard_quads(all_corners, "lwir", pattern_size)
    vis_quads = _extract_chessboard_quads(all_corners, "visible", pattern_size)

    lwir_poses: List[Tuple[float, float, float]] = []
    vis_poses: List[Tuple[float, float, float]] = []
    if pattern_size:
        lwir_poses = _compute_poses(all_corners, "lwir", matrices.get("lwir") or {}, pattern_size)
        vis_poses = _compute_poses(all_corners, "visible", matrices.get("visible") or {}, pattern_size)
        if square_size_m:
            lwir_poses = [(tx, ty, d * square_size_m) for tx, ty, d in lwir_poses]
            vis_poses = [(tx, ty, d * square_size_m) for tx, ty, d in vis_poses]

    cache = {
        "gui_version": APP_VERSION,
        "calibration_updated_at": file_metadata.get("updated_at"),
        "lwir_quads": lwir_quads,
        "vis_quads": vis_quads,
        "lwir_poses": [list(p) for p in lwir_poses],
        "vis_poses": [list(p) for p in vis_poses],
        "square_size_m": square_size_m,
        "dist_unit": dist_unit,
    }
    try:
        save_yaml(dataset_path / CACHE_FILENAME, cache, sort_keys=False)
    except Exception:  # noqa: BLE001
        pass


def load_report_cache(
    dataset_path: Path,
    gui_version: str,
    calibration_updated_at: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Return cached report data if it matches the current version and timestamp.

    Returns ``None`` if the cache file is missing, unreadable, or stale.
    """
    cache_path = dataset_path / CACHE_FILENAME
    if not cache_path.exists():
        return None
    try:
        data = load_yaml(cache_path)
    except Exception:  # noqa: BLE001
        return None
    if not data:
        return None
    if data.get("gui_version") != gui_version:
        return None
    if data.get("calibration_updated_at") != calibration_updated_at:
        return None
    return data


# ---------------------------------------------------------------------------
# Compute helpers (also imported by the dialog for fallback computation)
# ---------------------------------------------------------------------------

def _extract_chessboard_quads(
    all_corners: Dict[str, Dict[str, Any]],
    channel: str,
    pattern_size: Any,
) -> List[List[List[float]]]:
    """Extract the 4 outer corners of each chessboard detection as a quad."""
    cols: Optional[int] = None
    rows: Optional[int] = None
    if isinstance(pattern_size, (list, tuple)) and len(pattern_size) == 2:
        cols, rows = int(pattern_size[0]), int(pattern_size[1])

    quads: List[List[List[float]]] = []
    for _base, data in all_corners.items():
        corners = data.get(channel)
        if not corners or len(corners) < 4:
            continue
        if cols and rows and len(corners) == cols * rows:
            quad = [corners[0], corners[cols - 1], corners[-1], corners[-cols]]
        else:
            n = len(corners)
            quad = [corners[0], corners[n // 2 - 1], corners[-1], corners[n // 2]]
        quads.append(quad)
    return quads


def _compute_poses(
    all_corners: Dict[str, Dict[str, Any]],
    channel: str,
    channel_payload: Dict[str, Any],
    pattern_size: Any,
) -> List[Tuple[float, float, float]]:
    """Compute (tilt_x_deg, tilt_y_deg, distance) for each calibration image via solvePnP."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []

    if not channel_payload:
        return []

    cam_matrix = channel_payload.get("camera_matrix")
    dist_coeffs = channel_payload.get("distortion")
    image_size = channel_payload.get("image_size")
    if not cam_matrix or not dist_coeffs or not image_size or len(image_size) < 2:
        return []
    if not isinstance(pattern_size, (list, tuple)) or len(pattern_size) < 2:
        return []

    cols, rows = int(pattern_size[0]), int(pattern_size[1])
    expected = cols * rows
    obj_pts = np.zeros((expected, 3), dtype=np.float32)
    obj_pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    cam_mat = np.array(cam_matrix, dtype=np.float64)
    dist = np.array(dist_coeffs, dtype=np.float64)
    img_w, img_h = float(image_size[0]), float(image_size[1])

    poses: List[Tuple[float, float, float]] = []
    for _base, data in all_corners.items():
        corners = data.get(channel)
        if not corners or len(corners) != expected:
            continue
        img_pts = np.array(
            [[c[0] * img_w, c[1] * img_h] for c in corners], dtype=np.float32
        ).reshape(-1, 1, 2)
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, cam_mat, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        nx, ny, nz = float(R[0, 2]), float(R[1, 2]), float(R[2, 2])
        nz_safe = nz if abs(nz) > 1e-6 else 1e-6
        tilt_x = math.degrees(math.atan2(nx, nz_safe))
        tilt_y = math.degrees(math.atan2(ny, nz_safe))
        distance = float(np.linalg.norm(tvec))
        poses.append((tilt_x, tilt_y, distance))
    return poses


def _compute_square_size_m(file_metadata: Dict[str, Any]) -> Optional[float]:
    """Extract physical square size in metres from calibration file metadata."""
    sq_meta = file_metadata.get("square_size") or file_metadata.get("square_length")
    if isinstance(sq_meta, dict):
        val = sq_meta.get("value")
        unit = (sq_meta.get("unit") or "").lower()
        if isinstance(val, (int, float)):
            if unit == "mm":
                return float(val) / 1000.0
            if unit == "cm":
                return float(val) / 100.0
            if unit in ("m", "meters"):
                return float(val)
    elif isinstance(sq_meta, (int, float)):
        return float(sq_meta) / 1000.0
    return None
