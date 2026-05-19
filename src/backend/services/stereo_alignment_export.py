"""Write-only export of the active stereo alignment for external consumers.

This module dumps the current LWIR→Visible homography (including parallax
correction) and its parameters to ``.stereo_alignment.yaml`` at dataset
level. The GUI never reads this file back — runtime state lives in the
global cache and the calibration YAMLs. The file exists purely so external
tools can apply the same alignment without re-running the GUI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from common.log_utils import log_debug, log_warning
from common.yaml_utils import get_metadata_fields, save_yaml
from config import get_config


def export_stereo_alignment(
    dataset_path: Path,
    *,
    parallax_h: float,
    parallax_v: float,
    apply_parallax: bool,
    reference_depth_m: float,
    square_size_mm: Optional[float],
    lwir_matrices: Optional[Dict[str, Any]],
    vis_matrices: Optional[Dict[str, Any]],
    extrinsic: Optional[Dict[str, Any]],
    source_dataset: Optional[str] = None,
) -> bool:
    """Dump the active stereo alignment to ``.stereo_alignment.yaml``.

    Always overwrites. Skips silently if calibration is not available
    (homography cannot be computed without K_lwir, K_vis, R, T).

    The "apply_parallax" flag mirrors the GUI's "Apply Parallax Correction"
    toggle: when False, the dumped homography uses parallax=0 so it matches
    what the GUI is actually showing.

    Returns True on successful write, False otherwise.
    """
    if not dataset_path or not lwir_matrices or not vis_matrices or not extrinsic:
        return False

    try:
        from backend.utils.stereo_alignment import compute_alignment_homography
    except ImportError:
        return False

    rotation = extrinsic.get("rotation") or extrinsic.get("R")
    translation = extrinsic.get("translation") or extrinsic.get("T")
    if rotation is None or translation is None:
        return False

    lwir_size = _read_image_size(lwir_matrices)
    vis_size = _read_image_size(vis_matrices)
    if lwir_size is None or vis_size is None:
        return False

    effective_h = parallax_h if apply_parallax else 0.0
    effective_v = parallax_v if apply_parallax else 0.0

    homography = compute_alignment_homography(
        source_matrix=lwir_matrices,
        target_matrix=vis_matrices,
        rotation=rotation,
        translation=translation,
        image_size=lwir_size,
        source_is_lwir=True,
        parallax_h=effective_h,
        parallax_v=effective_v,
    )

    if homography is None:
        log_debug("Skipping stereo alignment export: homography unavailable", "ALIGN_EXPORT")
        return False

    payload: Dict[str, Any] = {
        "version": 1,
        **get_metadata_fields(),
        "source_dataset": source_dataset or dataset_path.name,
        "parallax_correction": {
            "h_px": float(parallax_h),
            "v_px": float(parallax_v),
            "apply": bool(apply_parallax),
            "reference_depth_m": float(reference_depth_m),
            "square_size_mm": float(square_size_mm) if square_size_mm else None,
        },
        # 3x3 homography mapping LWIR pixels to Visible pixels (cv2.warpPerspective).
        # Includes parallax shift when apply=True; row-major.
        "homography_lwir_to_visible": [
            [float(homography[i, j]) for j in range(3)] for i in range(3)
        ],
        "calibration_size": {
            "lwir": [int(lwir_size[0]), int(lwir_size[1])],
            "visible": [int(vis_size[0]), int(vis_size[1])],
        },
    }

    target = dataset_path / get_config().stereo_alignment_filename
    if not save_yaml(target, payload, sort_keys=False):
        log_warning(f"Failed to write {target.name}", "ALIGN_EXPORT")
        return False

    log_debug(f"Wrote stereo alignment export to {target.name}", "ALIGN_EXPORT")
    return True


def _read_image_size(matrix: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    raw = matrix.get("image_size")
    if isinstance(raw, list) and len(raw) >= 2:
        return (int(raw[0]), int(raw[1]))
    return None


__all__ = ["export_stereo_alignment"]
