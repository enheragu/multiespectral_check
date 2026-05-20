"""Attribute propagation between consecutive frames.

Propagates annotation attributes (role, age, state, etc.) from a source frame
to a target frame by matching annotations spatially.

Matching strategy (two-step):
  1. Direct IoU — if the bbox of a source annotation overlaps the target
     annotation enough (>= _IOU_THRESHOLD) and they share the same class,
     the attributes are considered transferable.
  2. Sparse Lucas-Kanade optical flow — when IoU is too low to be conclusive,
     a 3×3 grid of points inside the source bbox is tracked to the target frame
     to estimate the displacement. The predicted new bbox position is then
     compared with the target annotation via IoU.

Only attributes that are *absent* in the target are filled in; existing values
are never overwritten. Model metadata keys ("model", "model_version") are never
propagated.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from common.log_utils import log_debug, log_info

from .label_service import LabelService, _bbox_iou
from .label_types import Annotation

# ── Tuneable constants ────────────────────────────────────────────────────────

_IOU_THRESHOLD = 0.25        # direct IoU match
_LK_IOU_THRESHOLD = 0.20     # IoU match after LK-flow prediction
_LK_MAX_LEVEL = 3            # pyramid levels; level N handles ~winSize*2^N px displacement

# Keys never propagated: auto-detector metadata, our propagation marker,
# and per-frame geometric properties that don't transfer between images.
_SKIP_ATTR_KEYS: frozenset[str] = frozenset({
    "model", "model_version", "_propagated_attrs",
    "truncation", "occlusion",  # frame-specific: same person may not be occluded/truncated next frame
})

# Minimum number of successfully tracked LK points to trust the result
_LK_MIN_TRACKED = 3


# ── Internal helpers ──────────────────────────────────────────────────────────

def _has_meaningful_attrs(ann: Annotation) -> bool:
    """True if *ann* carries at least one user-set attribute value."""
    return any(
        v is not None and k not in _SKIP_ATTR_KEYS
        for k, v in ann.attributes.items()
    )


def _attrs_to_propagate(source: Annotation, target: Annotation) -> Dict[str, Any]:
    """Return source attributes that are absent (None or missing) in target."""
    present_in_target = {
        k for k, v in target.attributes.items()
        if v is not None and k not in _SKIP_ATTR_KEYS
    }
    return {
        k: v
        for k, v in source.attributes.items()
        if k not in _SKIP_ATTR_KEYS and k not in present_in_target and v is not None
    }


def _sample_bbox_points(
    bbox: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """Return a 3×3 grid of pixel coordinates inside *bbox* (normalized cx,cy,w,h)."""
    cx, cy, w, h = bbox
    x0 = (cx - w / 2) * img_w
    x1 = (cx + w / 2) * img_w
    y0 = (cy - h / 2) * img_h
    y1 = (cy + h / 2) * img_h
    xs = np.linspace(x0, x1, 3)
    ys = np.linspace(y0, y1, 3)
    pts = np.array(
        [[x, y] for y in ys for x in xs], dtype=np.float32
    ).reshape(-1, 1, 2)
    return pts


def _predict_bbox_via_lk(
    source_bbox: Tuple[float, float, float, float],
    source_gray: np.ndarray,
    target_gray: np.ndarray,
) -> Optional[Tuple[float, float, float, float]]:
    """Predict where *source_bbox* moved in *target_gray* using sparse LK flow.

    Returns the predicted bbox in normalised (cx, cy, w, h) coords, or None if
    tracking fails (too few good points).
    """
    try:
        import cv2
    except ImportError:
        return None

    h, w = source_gray.shape[:2]
    pts = _sample_bbox_points(source_bbox, w, h)

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=_LK_MAX_LEVEL,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        source_gray, target_gray, pts, None, **lk_params
    )

    if new_pts is None or status is None:
        return None

    mask = status.squeeze() == 1
    good_old = pts[mask]
    good_new = new_pts[mask]

    if len(good_old) < _LK_MIN_TRACKED:
        return None

    dx = float(np.median(good_new[:, 0, 0] - good_old[:, 0, 0])) / w
    dy = float(np.median(good_new[:, 0, 1] - good_old[:, 0, 1])) / h

    cx, cy, bw, bh = source_bbox
    return (cx + dx, cy + dy, bw, bh)


def _load_gray(path: Path) -> Optional[np.ndarray]:
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def propagate_attributes(
    source_base: str,
    target_base: str,
    channel: str,
    label_service: LabelService,
    get_image_path: Callable[[str, str], Optional[Path]],
) -> int:
    """Copy attributes from *source_base* annotations to matching *target_base* annotations.

    For each annotation in the target frame that is missing attributes, the
    function looks for the best-matching annotation in the source frame with the
    same class and filled attributes. Matching uses IoU first; if IoU is
    insufficient, sparse optical flow is used to predict the displacement.

    Attributes are never overwritten — only absent values are filled in.

    Args:
        source_base: Image base name of the frame that supplies attributes.
        target_base: Image base name of the frame that receives attributes.
        channel: "visible" or "lwir".
        label_service: The active LabelService instance.
        get_image_path: Callable(base, channel) → Path | None for image loading.

    Returns:
        Number of target annotations that had attributes propagated to them.
    """
    source_anns: List[Annotation] = label_service.get_annotations(source_base, channel)
    target_anns: List[Annotation] = label_service.get_annotations(target_base, channel)

    if not source_anns or not target_anns:
        return 0

    # Only use source annotations that actually have user-set attributes
    donors = [a for a in source_anns if _has_meaningful_attrs(a)]
    if not donors:
        return 0

    updated = 0
    source_gray: Optional[np.ndarray] = None
    target_gray: Optional[np.ndarray] = None
    lk_attempted = False  # load images lazily, at most once per call

    for target_ann in target_anns:
        if target_ann.annotation_id is None:
            continue

        candidates = [d for d in donors if d.class_id == target_ann.class_id]
        if not candidates:
            continue

        # ── Step 1: direct IoU match ──────────────────────────────────────
        best_iou = 0.0
        best_source: Optional[Annotation] = None
        for src in candidates:
            iou = _bbox_iou(src.bbox, target_ann.bbox)
            if iou > best_iou:
                best_iou = iou
                best_source = src

        if best_iou >= _IOU_THRESHOLD and best_source is not None:
            new_attrs = _attrs_to_propagate(best_source, target_ann)
        else:
            # ── Step 2: sparse optical flow ───────────────────────────────
            new_attrs = {}
            if not lk_attempted:
                lk_attempted = True
                src_path = get_image_path(source_base, channel)
                tgt_path = get_image_path(target_base, channel)
                if src_path and tgt_path:
                    source_gray = _load_gray(src_path)
                    target_gray = _load_gray(tgt_path)

            if source_gray is not None and target_gray is not None:
                best_lk_iou = 0.0
                best_lk_source: Optional[Annotation] = None
                for src in candidates:
                    predicted = _predict_bbox_via_lk(src.bbox, source_gray, target_gray)
                    if predicted is None:
                        continue
                    iou = _bbox_iou(predicted, target_ann.bbox)
                    if iou > best_lk_iou:
                        best_lk_iou = iou
                        best_lk_source = src

                if best_lk_iou >= _LK_IOU_THRESHOLD and best_lk_source is not None:
                    new_attrs = _attrs_to_propagate(best_lk_source, target_ann)
                    log_debug(
                        f"LK match ({best_lk_iou:.2f}) {source_base} → {target_base} "
                        f"ann_id={target_ann.annotation_id}",
                        "PROPAGATOR",
                    )


        if new_attrs:
            # Record which keys were propagated so the editor can highlight them
            existing_propagated = set(target_ann.attributes.get("_propagated_attrs") or [])
            new_attrs["_propagated_attrs"] = sorted(existing_propagated | set(new_attrs.keys()))
            ok = label_service.update_annotation(
                target_base, channel, target_ann.annotation_id,
                new_attributes=new_attrs,
            )
            if ok:
                updated += 1
                log_debug(
                    f"Propagated {list(new_attrs.keys())} from {source_base} to "
                    f"{target_base} ann_id={target_ann.annotation_id}",
                    "PROPAGATOR",
                )

    if updated:
        log_info(
            f"Attribute propagation: {source_base} → {target_base} "
            f"({channel}): {updated} annotation(s) updated",
            "PROPAGATOR",
        )

    return updated
