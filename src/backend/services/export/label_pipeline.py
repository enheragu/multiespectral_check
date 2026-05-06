"""Label transformation pipeline for dataset export.

Mirrors the GUI's ``align_max_overlap`` semantics for labels: each
channel's bbox is mapped affinely from its native frame into the
shared cropped output (no projective warp), so labels stay coherent
with the cropped/resized images produced by ``transform_aligned_pair``.

Behavior:
- Only ``manual`` and ``reviewed`` annotations are exported (auto-pending
  are dropped).
- Output is the **union** of both channels' annotations expressed in
  the exported channel's frame:
    * Annotations native to this channel pass through (subject to the
      affine crop / resize).
    * Annotations from the other channel are projected via the
      homography ``H`` (LWIR→Visible or its inverse) and then through
      the same affine crop, so they end up in the same coordinate
      system as the native ones.
- When both channels are exported aligned, both YAMLs end up carrying
  identical normalized coordinates because the cropped+resized frames
  share the same scene area at the same output size.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore

from backend.services.export.image_pipeline import (
    CalibrationBundle,
    TransformParams,
    compute_export_homography,
)
from backend.services.labels.bbox_transform import project_bbox_with_homography
from backend.services.labels.label_types import (
    Annotation,
    AnnotationSource,
    ImageLabels,
)
from common.log_utils import log_debug, log_warning


PERSISTENT_SOURCES = (AnnotationSource.MANUAL, AnnotationSource.REVIEWED)


def build_exported_labels(
    *,
    target_channel: str,
    target_image_file: str,
    target_image_size: Tuple[int, int],
    own_native_size: Optional[Tuple[int, int]],
    other_native_size: Optional[Tuple[int, int]],
    own_labels: Optional[ImageLabels],
    other_labels: Optional[ImageLabels],
    calib: Optional[CalibrationBundle],
    params: TransformParams,
    both_channels_selected: bool,
    own_crop_in_native: Optional[Tuple[int, int, int, int]] = None,
    other_crop_in_native: Optional[Tuple[int, int, int, int]] = None,
) -> ImageLabels:
    """Return the ImageLabels object to write for ``target_channel``.

    Args:
        target_channel: ``"lwir"`` or ``"visible"`` — channel being exported.
        target_image_file: Filename to record in the YAML.
        target_image_size: Size of the exported image (post crop+resize).
        own_native_size: This channel's native pixel size (matches
            normalized bboxes in ``own_labels``).
        other_native_size: Other channel's native pixel size (matches
            normalized bboxes in ``other_labels``).
        own_labels / other_labels: Source label sets.
        calib: Calibration bundle (or None — no projection then).
        params: Transform parameters.
        both_channels_selected: Whether the export includes both channels.
        own_crop_in_native: For aligned exports, the (x, y, w, h) crop
            applied to this channel in its own native frame (from
            ``AlignedPair``). The bbox is shifted by this crop offset
            and re-normalized to the cropped patch size — which equals
            normalized coords in the resized output (resize is uniform).
        other_crop_in_native: Same idea for the other channel; used
            after projecting other-channel bboxes onto this channel's
            native frame via H.
    """
    out = ImageLabels(image_file=target_image_file, channel=target_channel)
    aligned_export = (
        both_channels_selected
        and params.align_fov
        and own_crop_in_native is not None
    )

    # ── own labels ────────────────────────────────────────────────────────
    if own_labels is not None and own_native_size is not None:
        own_w, own_h = own_native_size
        crop = own_crop_in_native if aligned_export else (0, 0, own_w, own_h)
        for ann in own_labels.annotations:
            if ann.source not in PERSISTENT_SOURCES:
                continue
            new_bbox = _apply_affine_crop(ann.bbox, own_native_size, crop)
            if new_bbox is None:
                continue
            out.add_annotation(_clone_annotation(ann, new_bbox))

    # ── projected labels from the other channel ──────────────────────────
    if (
        other_labels is not None
        and calib is not None
        and other_native_size is not None
    ):
        other_channel = "visible" if target_channel == "lwir" else "lwir"
        if aligned_export and other_crop_in_native is not None:
            # Aligned export: project from other-native to this-native via H,
            # then apply this channel's affine crop.
            H = _homography(calib, params, source_channel=other_channel)
            if H is not None and own_native_size is not None:
                own_w, own_h = own_native_size
                this_crop = own_crop_in_native if own_crop_in_native is not None else (0, 0, own_w, own_h)
                for ann in other_labels.annotations:
                    if ann.source not in PERSISTENT_SOURCES:
                        continue
                    projected_norm = project_bbox_with_homography(
                        ann.bbox, H, other_native_size, own_native_size
                    )
                    if projected_norm is None:
                        continue
                    new_bbox = _apply_affine_crop(projected_norm, own_native_size, this_crop)
                    if new_bbox is None:
                        continue
                    out.add_annotation(
                        _clone_annotation(ann, new_bbox, projected_from=other_channel)
                    )
            else:
                log_debug(
                    f"[export] no homography {other_channel}->{target_channel}; "
                    f"skipping projected labels",
                    "EXPORT",
                )
        else:
            # Single-channel export: project other → this via H. No crop.
            H = _homography(calib, params, source_channel=other_channel)
            if H is not None and own_native_size is not None:
                for ann in other_labels.annotations:
                    if ann.source not in PERSISTENT_SOURCES:
                        continue
                    projected_norm = project_bbox_with_homography(
                        ann.bbox, H, other_native_size, own_native_size
                    )
                    if projected_norm is None:
                        continue
                    out.add_annotation(
                        _clone_annotation(ann, projected_norm, projected_from=other_channel)
                    )

    return out


def _homography(
    calib: CalibrationBundle,
    params: TransformParams,
    *,
    source_channel: str,
) -> Optional[Any]:
    """``H`` mapping ``source_channel`` pixel coords to the other channel's."""
    return compute_export_homography(
        calib, params, source_is_lwir=(source_channel == "lwir")
    )


def _apply_affine_crop(
    bbox_norm: Tuple[float, float, float, float],
    native_size: Tuple[int, int],
    crop: Tuple[int, int, int, int],
) -> Optional[Tuple[float, float, float, float]]:
    """Re-normalize a normalized bbox into a cropped patch of ``native_size``.

    ``crop = (x, y, w, h)`` is the patch in native pixel coords. Returns
    ``None`` if the bbox falls entirely outside the patch.
    """
    nw, nh = native_size
    cx, cy, cw, ch = crop
    if cw <= 0 or ch <= 0:
        return None

    # Native pixel bounds of the bbox.
    px_xc = bbox_norm[0] * nw
    px_yc = bbox_norm[1] * nh
    px_w = bbox_norm[2] * nw
    px_h = bbox_norm[3] * nh
    left = px_xc - px_w / 2.0
    right = px_xc + px_w / 2.0
    top = px_yc - px_h / 2.0
    bottom = px_yc + px_h / 2.0

    # Clamp to crop bounds.
    left = max(left, float(cx))
    top = max(top, float(cy))
    right = min(right, float(cx + cw))
    bottom = min(bottom, float(cy + ch))
    new_w = right - left
    new_h = bottom - top
    if new_w <= 0 or new_h <= 0:
        return None

    # Re-normalize against the cropped patch (resize is uniform, so
    # normalized coords are preserved across the resize).
    new_xc = (left + right) / 2.0 - cx
    new_yc = (top + bottom) / 2.0 - cy
    return (new_xc / cw, new_yc / ch, new_w / cw, new_h / ch)


def _clone_annotation(
    ann: Annotation,
    new_bbox: Tuple[float, float, float, float],
    *,
    projected_from: Optional[str] = None,
) -> Annotation:
    """Return a copy of ``ann`` with the given bbox.

    When the annotation was projected from another channel, an
    ``projected_from`` attribute is added so external consumers can
    distinguish native vs projected labels.
    """
    new_attrs = dict(ann.attributes) if ann.attributes else {}
    if projected_from is not None:
        new_attrs["projected_from"] = projected_from
    return Annotation(
        class_id=ann.class_id,
        bbox=tuple(new_bbox),  # type: ignore[arg-type]
        source=ann.source,
        attributes=new_attrs,
        annotation_id=None,
        confidence=ann.confidence,
    )


__all__ = [
    "PERSISTENT_SOURCES",
    "build_exported_labels",
]
