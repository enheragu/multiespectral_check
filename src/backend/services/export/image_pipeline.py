"""Headless image transformation pipeline for dataset export.

Mirrors the GUI's ``align_max_overlap`` behaviour for stereo alignment
without invoking Qt: project the LWIR FOV into visible coords through
the homography (with parallax baked in), intersect with visible bounds
to get the max-overlap rectangle, and apply the same axis-aligned
crop fractions to the LWIR's native image. The two cropped images are
then resized to a common output size depending on ``resolution_mode``.

Three independent toggles, all default ON:
- undistort:  cv2.undistort with K, D
- align_fov:  do the FOV crop / resize described above
- parallax:   adds the depth-based pixel shift to the homography

Resolution modes (only meaningful when both channels are exported):
- ``upsample_to_largest``: output size = visible-cropped size; LWIR
  cropped patch is upsampled to match.
- ``downsample_to_smallest``: output size = LWIR-cropped size; visible
  cropped patch is downsampled to match.

The pipeline is **affine** in LWIR coords (no projective warp on the
image), matching the GUI's behaviour. Labels follow the same affine
projection so they stay coherent with what the user sees in the GUI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore

from backend.utils.stereo_alignment import compute_alignment_homography
from common.log_utils import log_debug, log_warning


RESOLUTION_UPSAMPLE = "upsample_to_largest"
RESOLUTION_DOWNSAMPLE = "downsample_to_smallest"


@dataclass
class TransformParams:
    """Knobs for the export image pipeline."""
    undistort: bool = True
    align_fov: bool = True
    parallax: bool = True
    resolution_mode: str = RESOLUTION_UPSAMPLE
    parallax_h: float = 0.0
    parallax_v: float = 0.0


@dataclass
class CalibrationBundle:
    """Loaded calibration for a single dataset, or None if missing."""
    lwir_matrix: Dict[str, Any]      # {camera_matrix, distortion, image_size}
    vis_matrix: Dict[str, Any]
    extrinsic: Dict[str, Any]        # {rotation, translation}
    square_size_mm: Optional[float] = None


@dataclass
class AlignedPair:
    """Result of running ``transform_aligned_pair``.

    Carries enough metadata to re-normalize bboxes from each channel's
    native frame into the cropped, possibly resized output.
    """
    lwir_image: Optional["np.ndarray"]
    vis_image: Optional["np.ndarray"]
    output_size: Tuple[int, int]                      # (w, h) of saved images
    lwir_crop_in_native: Tuple[int, int, int, int]    # (x, y, w, h) in LWIR-native pixels
    vis_crop_in_native: Tuple[int, int, int, int]     # (x, y, w, h) in visible-native pixels


@dataclass
class TransformResult:
    """Result of ``transform_image_single`` (single-channel export, no alignment)."""
    image: "np.ndarray"
    output_size: Tuple[int, int]


def compute_export_homography(
    calib: CalibrationBundle,
    params: TransformParams,
    *,
    source_is_lwir: bool,
) -> Optional[Any]:
    """Build the homography to use for the export, with parallax baked in.

    Returns the matrix that maps SOURCE pixel coords to TARGET pixel coords
    (LWIR→Visible if source_is_lwir, otherwise Visible→LWIR).
    """
    h = params.parallax_h if params.parallax else 0.0
    v = params.parallax_v if params.parallax else 0.0

    if source_is_lwir:
        return compute_alignment_homography(
            source_matrix=calib.lwir_matrix,
            target_matrix=calib.vis_matrix,
            rotation=calib.extrinsic.get("rotation") or calib.extrinsic.get("R"),
            translation=calib.extrinsic.get("translation") or calib.extrinsic.get("T"),
            image_size=_image_size(calib.lwir_matrix),
            source_is_lwir=True,
            parallax_h=h,
            parallax_v=v,
        )
    return compute_alignment_homography(
        source_matrix=calib.vis_matrix,
        target_matrix=calib.lwir_matrix,
        rotation=calib.extrinsic.get("rotation") or calib.extrinsic.get("R"),
        translation=calib.extrinsic.get("translation") or calib.extrinsic.get("T"),
        image_size=_image_size(calib.vis_matrix),
        source_is_lwir=False,
        parallax_h=h,
        parallax_v=v,
    )


def transform_image_single(
    image: "np.ndarray",
    *,
    channel: str,
    calib: Optional[CalibrationBundle],
    params: TransformParams,
) -> TransformResult:
    """Transform a single channel without inter-channel alignment.

    Used when only one channel is being exported (no FOV alignment to
    do). Applies undistort if requested and crops the magenta sentinel
    border so no residual undistort halo remains.
    """
    if cv2 is None or np is None:
        h, w = image.shape[:2]
        return TransformResult(image=image, output_size=(w, h))

    if calib is not None:
        out, _rect = _maybe_undistort_and_crop(image, channel, calib, params)
        if out is None:
            out = image
    else:
        out = image
    return TransformResult(image=out, output_size=(out.shape[1], out.shape[0]))


def transform_aligned_pair(
    lwir_image: Optional["np.ndarray"],
    vis_image: Optional["np.ndarray"],
    *,
    calib: CalibrationBundle,
    params: TransformParams,
    keep_lwir: bool = True,
    keep_vis: bool = True,
) -> Optional[AlignedPair]:
    """Crop + resize a pair so both end up at the same size, FOV-aligned.

    Mirrors the GUI's ``align_max_overlap`` math:

    1. Undistort each channel (if ``params.undistort``).
    2. Project the LWIR's 4 corners through ``H`` (LWIR→Visible, with
       parallax) to get the FOV polygon in visible coords.
    3. ``fov_bbox`` = AABB of that polygon. ``overlap_rect`` =
       ``fov_bbox`` ∩ visible bounds — the visible crop region.
    4. Compute the clip fractions (how much each side of ``fov_bbox``
       was cut by visible bounds) and apply the same fractions to the
       LWIR-native rectangle to get the LWIR crop region.
    5. Output size depends on ``params.resolution_mode``:
       - upsample → visible-cropped size; LWIR is upsampled to match.
       - downsample → LWIR-cropped size; visible is downsampled to match.

    ``keep_lwir`` / ``keep_vis`` control which channel arrays are
    actually returned (the math runs either way; we only skip writing
    the unwanted channel). Either of ``lwir_image`` / ``vis_image``
    may be ``None`` if that channel was unreadable; the corresponding
    output channel will be ``None`` too.

    Returns ``None`` if there's no valid overlap or calibration is bad.
    """
    if cv2 is None or np is None:
        return None

    lwir_size_calib = _image_size(calib.lwir_matrix)
    vis_size_calib = _image_size(calib.vis_matrix)
    if lwir_size_calib is None or vis_size_calib is None:
        return None

    # ── 1) Undistort each channel and crop to the valid area ───────────
    # Both ops in one shot: ``_maybe_undistort_and_crop`` returns the
    # cropped image plus its rect in post-undistort native coords (same
    # frame H expects and labels live in).
    lwir_proc, lwir_valid = _maybe_undistort_and_crop(
        lwir_image, "lwir", calib, params
    )
    vis_proc, vis_valid = _maybe_undistort_and_crop(
        vis_image, "visible", calib, params
    )
    lvx, lvy, lvw, lvh = lwir_valid
    vvx, vvy, vvw, vvh = vis_valid

    if lvw <= 0 or lvh <= 0 or vvw <= 0 or vvh <= 0:
        log_warning("transform_aligned_pair: empty valid area after undistort crop", "EXPORT")
        return None

    # ── 3) Project the LWIR's VALID-area corners to visible coords ─────
    # We feed the homography corners that correspond to the LWIR's actual
    # data (excluding the magenta border), expressed in native LWIR
    # pixel coords — which is what H is calibrated for.
    H = compute_export_homography(calib, params, source_is_lwir=True)
    if H is None:
        return None

    lwir_corners_native = np.array(
        [
            [lvx, lvy],
            [lvx + lvw, lvy],
            [lvx + lvw, lvy + lvh],
            [lvx, lvy + lvh],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    fov_corners = cv2.perspectiveTransform(
        lwir_corners_native, H.astype(np.float32)
    ).reshape(-1, 2)

    fov_x1 = float(fov_corners[:, 0].min())
    fov_y1 = float(fov_corners[:, 1].min())
    fov_x2 = float(fov_corners[:, 0].max())
    fov_y2 = float(fov_corners[:, 1].max())
    fov_w = fov_x2 - fov_x1
    fov_h = fov_y2 - fov_y1
    if fov_w <= 0 or fov_h <= 0:
        log_warning("transform_aligned_pair: degenerate FOV polygon", "EXPORT")
        return None

    # ── 4) Visible crop = FOV bbox ∩ visible's valid area ──────────────
    # Clamp to the visible's valid (post-magenta-crop) rect, expressed
    # in visible native coords.
    crop_x1 = max(float(vvx), fov_x1)
    crop_y1 = max(float(vvy), fov_y1)
    crop_x2 = min(float(vvx + vvw), fov_x2)
    crop_y2 = min(float(vvy + vvh), fov_y2)
    overlap_w = crop_x2 - crop_x1
    overlap_h = crop_y2 - crop_y1
    if overlap_w <= 0 or overlap_h <= 0:
        log_warning("transform_aligned_pair: empty overlap", "EXPORT")
        return None

    # ── 5) Map clip fractions back to LWIR's valid area ────────────────
    clip_left = (crop_x1 - fov_x1) / fov_w
    clip_top = (crop_y1 - fov_y1) / fov_h
    clip_right = (fov_x2 - crop_x2) / fov_w
    clip_bottom = (fov_y2 - crop_y2) / fov_h

    # LWIR crop, expressed in NATIVE (post-undistort) coords.
    lwir_x1_native = lvx + max(0, int(round(clip_left * lvw)))
    lwir_y1_native = lvy + max(0, int(round(clip_top * lvh)))
    lwir_x2_native = lvx + min(lvw, int(round(lvw - clip_right * lvw)))
    lwir_y2_native = lvy + min(lvh, int(round(lvh - clip_bottom * lvh)))
    lwir_crop_w = lwir_x2_native - lwir_x1_native
    lwir_crop_h = lwir_y2_native - lwir_y1_native
    if lwir_crop_w <= 0 or lwir_crop_h <= 0:
        log_warning("transform_aligned_pair: empty LWIR crop", "EXPORT")
        return None

    # Integer-pixel visible crop bounds (in visible native coords).
    vx1 = int(round(crop_x1))
    vy1 = int(round(crop_y1))
    vx2 = int(round(crop_x2))
    vy2 = int(round(crop_y2))
    vis_crop_w = max(0, vx2 - vx1)
    vis_crop_h = max(0, vy2 - vy1)
    if vis_crop_w <= 0 or vis_crop_h <= 0:
        return None

    # ── 6) Determine output size and resize as needed ───────────────────
    if params.resolution_mode == RESOLUTION_UPSAMPLE:
        out_w, out_h = vis_crop_w, vis_crop_h
    else:  # RESOLUTION_DOWNSAMPLE
        out_w, out_h = lwir_crop_w, lwir_crop_h

    log_debug(
        f"[export] aligned_pair: lwir_valid=({lvx},{lvy},{lvw},{lvh}); "
        f"vis_valid=({vvx},{vvy},{vvw},{vvh}); "
        f"fov_bbox=({fov_x1:.1f},{fov_y1:.1f},{fov_x2:.1f},{fov_y2:.1f}); "
        f"vis_crop=({vx1},{vy1},{vis_crop_w},{vis_crop_h}); "
        f"lwir_crop=({lwir_x1_native},{lwir_y1_native},{lwir_crop_w},{lwir_crop_h}); "
        f"out=({out_w}x{out_h}) mode={params.resolution_mode}",
        "EXPORT",
    )

    # ``lwir_proc`` and ``vis_proc`` are already cropped to the valid area;
    # convert the native rects to LOCAL rects within them for slicing.
    lwir_local_x1 = lwir_x1_native - lvx
    lwir_local_y1 = lwir_y1_native - lvy
    vis_local_x1 = vx1 - vvx
    vis_local_y1 = vy1 - vvy

    lwir_out: Optional[np.ndarray] = None
    if keep_lwir and lwir_proc is not None:
        patch = lwir_proc[
            lwir_local_y1:lwir_local_y1 + lwir_crop_h,
            lwir_local_x1:lwir_local_x1 + lwir_crop_w,
        ]
        lwir_out = _resize_if_needed(patch, (out_w, out_h))

    vis_out: Optional[np.ndarray] = None
    if keep_vis and vis_proc is not None:
        patch = vis_proc[
            vis_local_y1:vis_local_y1 + vis_crop_h,
            vis_local_x1:vis_local_x1 + vis_crop_w,
        ]
        vis_out = _resize_if_needed(patch, (out_w, out_h))

    return AlignedPair(
        lwir_image=lwir_out,
        vis_image=vis_out,
        output_size=(out_w, out_h),
        lwir_crop_in_native=(lwir_x1_native, lwir_y1_native, lwir_crop_w, lwir_crop_h),
        vis_crop_in_native=(vx1, vy1, vis_crop_w, vis_crop_h),
    )


# ── helpers ──────────────────────────────────────────────────────────────


def _maybe_undistort_and_crop(
    image: Optional["np.ndarray"],
    channel: str,
    calib: CalibrationBundle,
    params: TransformParams,
) -> Tuple[Optional["np.ndarray"], Tuple[int, int, int, int]]:
    """Undistort + crop to the valid (in-frame) area.

    Builds the inverse rectification map and remaps both the image and
    a uniform "valid" mask. The mask tells us exactly which output
    pixels came from inside the source frame; the largest axis-aligned
    rect mostly inside that mask is used to crop.

    This works for any channel count (grayscale and BGR) and avoids the
    sentinel-colour pitfall (LWIR thermal data may legitimately span
    full intensity range, so colour-based detection is unreliable).

    Returns ``(processed_image, valid_rect_in_native)`` where
    ``valid_rect_in_native`` is ``(x, y, w, h)`` in the **post-undistort
    native pixel grid** (= same coords H is calibrated for; same coords
    where labels live). When ``params.undistort`` is False this is just
    the full image rect.
    """
    if image is None:
        return None, (0, 0, 0, 0)
    h, w = image.shape[:2]
    if not params.undistort:
        return image, (0, 0, w, h)
    K, D = _read_intrinsic(calib.lwir_matrix if channel == "lwir" else calib.vis_matrix)
    if K is None or D is None:
        return image, (0, 0, w, h)
    try:
        map1, map2 = cv2.initUndistortRectifyMap(
            K, D, None, K, (w, h), cv2.CV_32FC1
        )
        undistorted = cv2.remap(
            image, map1, map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        # Remap a uniform 1-mask the same way to learn which output
        # pixels came from inside the source frame.
        valid_src = np.ones((h, w), dtype=np.uint8)
        valid_mask = cv2.remap(
            valid_src, map1, map2,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        cropped, rect = _crop_to_valid_rect(undistorted, valid_mask)
        log_debug(
            f"[export] undistorted {channel} {w}x{h} -> valid rect={rect}",
            "EXPORT",
        )
        return cropped, rect
    except Exception as e:
        log_warning(f"undistort failed on {channel}: {e}", "EXPORT")
        return image, (0, 0, w, h)


def _crop_to_valid_rect(
    img: "np.ndarray",
    mask: "np.ndarray",
    threshold: float = 0.95,
) -> Tuple["np.ndarray", Tuple[int, int, int, int]]:
    """Find the largest AABB whose rows/cols are mostly inside ``mask``.

    Shrinks the four sides until each border row/column has at least
    ``threshold`` fraction of valid pixels. Mirrors the heuristic used
    by ``_crop_black_borders`` in ``stereo_alignment.py``.
    """
    h, w = img.shape[:2]
    top, bottom, left, right = 0, h, 0, w

    # Shrink top
    for row in range(h):
        if np.sum(mask[row, :]) >= threshold * w:
            top = row
            break

    # Shrink bottom
    for row in range(h - 1, top, -1):
        if np.sum(mask[row, :]) >= threshold * w:
            bottom = row + 1
            break

    height = bottom - top
    if height <= 0:
        return img, (0, 0, w, h)

    # Shrink left
    for col in range(w):
        if np.sum(mask[top:bottom, col]) >= threshold * height:
            left = col
            break

    # Shrink right
    for col in range(w - 1, left, -1):
        if np.sum(mask[top:bottom, col]) >= threshold * height:
            right = col + 1
            break

    return img[top:bottom, left:right], (left, top, right - left, bottom - top)


def _resize_if_needed(
    patch: "np.ndarray", target_size: Tuple[int, int]
) -> "np.ndarray":
    """Resize ``patch`` to ``target_size`` if needed; INTER_AREA when shrinking."""
    h, w = patch.shape[:2]
    tw, th = target_size
    if (w, h) == (tw, th):
        return patch
    interp = cv2.INTER_AREA if (tw < w or th < h) else cv2.INTER_LINEAR
    return cv2.resize(patch, (tw, th), interpolation=interp)


def _read_intrinsic(matrix: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Any]]:
    """Return (K, D) numpy arrays from a calibration matrix dict, or (None, None)."""
    if np is None:
        return None, None
    K_data = matrix.get("camera_matrix") or matrix.get("data")
    if isinstance(K_data, dict):
        K_data = K_data.get("data")
    if not K_data:
        return None, None
    try:
        K = np.array(K_data, dtype=np.float64).reshape(3, 3)
    except Exception:
        return None, None

    D_data = matrix.get("distortion")
    if isinstance(D_data, dict):
        D_data = D_data.get("data")
    if D_data is None:
        D = np.zeros(5, dtype=np.float64)
    else:
        try:
            D = np.array(D_data, dtype=np.float64).flatten()
        except Exception:
            D = np.zeros(5, dtype=np.float64)
    return K, D


def _image_size(matrix: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    raw = matrix.get("image_size")
    if isinstance(raw, list) and len(raw) >= 2:
        return (int(raw[0]), int(raw[1]))
    return None


__all__ = [
    "RESOLUTION_UPSAMPLE",
    "RESOLUTION_DOWNSAMPLE",
    "TransformParams",
    "CalibrationBundle",
    "AlignedPair",
    "TransformResult",
    "compute_export_homography",
    "transform_image_single",
    "transform_aligned_pair",
]
