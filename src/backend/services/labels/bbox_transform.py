"""Bounding box transformation utilities for projecting labels between cameras.

This module provides functions to project bounding boxes from one camera view
to another using stereo calibration. This is used to show "projected" labels
on the other camera view (displayed with dashed lines to indicate they are
computed, not directly labeled).

IMPORTANT: This module REUSES the transformation logic from stereo_alignment.py
to avoid code duplication. The homography computation and point transformation
are delegated to existing, tested functions.

Key principle from DESIGN_PHILOSOPHY.md:
- Projected labels are COMPUTED, not stored
- Only source labels are saved; projections are recalculated on display
- This ensures projections update automatically when calibration improves
- NO code duplication - reuse existing transformations

Usage:
    # Project a visible bbox to LWIR view
    lwir_bbox = project_bbox_to_other_channel(
        bbox=(0.5, 0.5, 0.1, 0.2),
        source_channel="visible",
        calibration_data=session.get_calibration_data(),
        source_image_size=(1920, 1080),
        target_image_size=(640, 480),
    )
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from common.log_utils import log_debug, log_warning

# Import from stereo_alignment to REUSE existing code
from backend.utils.stereo_alignment import compute_alignment_homography

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore


def project_bbox_with_homography(
    bbox: Tuple[float, float, float, float],
    homography: Any,
    source_image_size: Tuple[int, int],
    target_image_size: Tuple[int, int],
) -> Optional[Tuple[float, float, float, float]]:
    """Project a bounding box using a precomputed homography.

    This is the core projection function that uses cv2.perspectiveTransform,
    the same function used in stereo_alignment.py for corner transformation.

    Args:
        bbox: Source bbox as (x_center, y_center, width, height) normalized [0,1]
        homography: 3x3 homography matrix (from compute_alignment_homography)
        source_image_size: (width, height) of source image
        target_image_size: (width, height) of target image

    Returns:
        Projected bbox as (x_center, y_center, width, height) normalized [0,1]
        or None if projection fails
    """
    if cv2 is None or np is None or homography is None:
        return None

    src_w, src_h = source_image_size
    tgt_w, tgt_h = target_image_size

    # Convert normalized bbox to pixel corners
    xc, yc, w, h = bbox
    corners_norm = _bbox_to_corners(xc, yc, w, h)

    # Denormalize to source pixels
    corners_px = np.array([
        [c[0] * src_w, c[1] * src_h] for c in corners_norm
    ], dtype=np.float32).reshape(-1, 1, 2)

    # Project corners through homography
    # This is the SAME function used in stereo_alignment.py compute_overlap_region()
    try:
        projected = cv2.perspectiveTransform(corners_px, homography.astype(np.float32))
        projected = projected.reshape(-1, 2)
    except Exception as e:
        log_warning(f"perspectiveTransform failed: {e}", "LABELS")
        return None

    # Find bounding box of projected corners
    min_x = float(projected[:, 0].min())
    max_x = float(projected[:, 0].max())
    min_y = float(projected[:, 1].min())
    max_y = float(projected[:, 1].max())

    # Clamp to target image bounds
    min_x = max(0.0, min_x)
    max_x = min(float(tgt_w), max_x)
    min_y = max(0.0, min_y)
    max_y = min(float(tgt_h), max_y)

    # Check if projected box is valid
    proj_w = max_x - min_x
    proj_h = max_y - min_y
    if proj_w <= 0 or proj_h <= 0:
        return None

    # Convert back to normalized center format
    proj_xc = (min_x + max_x) / 2 / tgt_w
    proj_yc = (min_y + max_y) / 2 / tgt_h
    proj_w_norm = proj_w / tgt_w
    proj_h_norm = proj_h / tgt_h

    log_debug(
        f"Projected bbox: ({xc:.3f},{yc:.3f},{w:.3f},{h:.3f}) -> "
        f"({proj_xc:.3f},{proj_yc:.3f},{proj_w_norm:.3f},{proj_h_norm:.3f})",
        "LABELS"
    )

    return (proj_xc, proj_yc, proj_w_norm, proj_h_norm)


def project_bbox_to_other_channel(
    bbox: Tuple[float, float, float, float],
    source_channel: str,
    calibration_data: Dict[str, Any],
    source_image_size: Tuple[int, int],
    target_image_size: Tuple[int, int],
) -> Optional[Tuple[float, float, float, float]]:
    """Project a bounding box from one camera to the other.

    Uses compute_alignment_homography from stereo_alignment.py to compute
    the transformation, then applies it to the bbox corners.

    Note: This is an APPROXIMATION. True 3D projection requires depth information.
    We use the homography from stereo rectification as a planar approximation,
    which works reasonably well for objects at typical distances.

    Args:
        bbox: Source bbox as (x_center, y_center, width, height) normalized [0,1]
        source_channel: "visible" or "lwir" - where the bbox was labeled
        calibration_data: Dict containing intrinsic and extrinsic calibration
        source_image_size: (width, height) of source image
        target_image_size: (width, height) of target image

    Returns:
        Projected bbox as (x_center, y_center, width, height) normalized [0,1]
        or None if projection fails
    """
    if source_channel not in ("visible", "lwir"):
        log_warning(f"Invalid source_channel: {source_channel}", "LABELS")
        return None

    target_channel = "lwir" if source_channel == "visible" else "visible"

    # Get or compute homography using existing function from stereo_alignment
    homography = _get_cached_homography(
        source_channel,
        target_channel,
        calibration_data,
        source_image_size,
    )

    if homography is None:
        log_debug(
            f"No homography available for {source_channel}->{target_channel}",
            "LABELS"
        )
        return None

    return project_bbox_with_homography(
        bbox, homography, source_image_size, target_image_size
    )


def project_annotations_to_other_channel(
    annotations: List[Tuple[int, Tuple[float, float, float, float], Dict[str, Any]]],
    source_channel: str,
    calibration_data: Dict[str, Any],
    source_image_size: Tuple[int, int],
    target_image_size: Tuple[int, int],
) -> List[Tuple[int, Tuple[float, float, float, float], Dict[str, Any], bool]]:
    """Project multiple annotations to the other channel.

    Args:
        annotations: List of (class_id, bbox, attributes) tuples
        source_channel: Source camera channel
        calibration_data: Calibration data dict
        source_image_size: Source image dimensions
        target_image_size: Target image dimensions

    Returns:
        List of (class_id, bbox, attributes, is_projected) tuples
        where is_projected=True for all returned annotations
    """
    result = []
    for class_id, bbox, attrs in annotations:
        projected_bbox = project_bbox_to_other_channel(
            bbox, source_channel, calibration_data,
            source_image_size, target_image_size
        )
        if projected_bbox is not None:
            result.append((class_id, projected_bbox, attrs, True))
    return result


def _bbox_to_corners(
    xc: float, yc: float, w: float, h: float
) -> Tuple[Tuple[float, float], ...]:
    """Convert center-format bbox to corner points.

    Returns corners in order: top-left, top-right, bottom-right, bottom-left
    """
    half_w, half_h = w / 2, h / 2
    return (
        (xc - half_w, yc - half_h),
        (xc + half_w, yc - half_h),
        (xc + half_w, yc + half_h),
        (xc - half_w, yc + half_h),
    )


def _get_cached_homography(
    source_channel: str,
    target_channel: str,
    calibration_data: Dict[str, Any],
    image_size: Tuple[int, int],
) -> Optional[Any]:
    """Get or compute homography for channel projection.

    REUSES compute_alignment_homography from stereo_alignment.py
    instead of duplicating the computation logic.

    Args:
        source_channel: "visible" or "lwir"
        target_channel: "visible" or "lwir"
        calibration_data: Dict with intrinsic and extrinsic calibration
        image_size: Image size for homography computation

    Returns:
        3x3 homography matrix or None
    """
    # Check for cached homography in calibration_data
    cache_key = f"_homography_{source_channel}_to_{target_channel}"
    if cache_key in calibration_data:
        return calibration_data[cache_key]

    # Need both intrinsics and extrinsics
    intrinsic = calibration_data.get("intrinsic", {})
    extrinsic = calibration_data.get("extrinsic", {})

    if not intrinsic or not extrinsic:
        return None

    # Get camera data for source and target
    source_data = intrinsic.get(source_channel, {})
    target_data = intrinsic.get(target_channel, {})

    if not source_data or not target_data:
        return None

    # Use compute_alignment_homography from stereo_alignment.py
    # This is the SAME function used for image alignment
    source_is_lwir = source_channel == "lwir"

    homography = compute_alignment_homography(
        source_matrix=source_data,
        target_matrix=target_data,
        rotation=extrinsic.get("R", {}),
        translation=extrinsic.get("T", {}),
        image_size=image_size,
        source_is_lwir=source_is_lwir,
    )

    # Cache for future use
    if homography is not None:
        calibration_data[cache_key] = homography

    return homography
