"""Ensemble detector — fuse results from multiple detection models.

Runs N ``DetectionModel`` instances on the same image and merges their
outputs using IoU-based duplicate suppression.  When two boxes from
different models overlap above a threshold *and* map to the same schema
class, they are merged into one detection with boosted confidence.

Design goals (DESIGN_PHILOSOPHY.md):
- Pure backend, no Qt dependencies.
- Encapsulates all fusion logic — callers just see a ``DetectionModel``.
- Pluggable: accepts any ``(DetectionModel, ClassMapper)`` pair.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from common.log_utils import log_debug, log_info

from .base import ClassMapper, DetectionModel, DetectionResult, ModelInfo


# ============================================================================
# IoU helpers
# ============================================================================

def _bbox_to_xyxy(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """Convert normalised centre-format ``(cx, cy, w, h)`` to ``(x1, y1, x2, y2)``."""
    x1 = cx - w / 2
    y1 = cy - h / 2
    return (x1, y1, x1 + w, y1 + h)


def _iou(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """Compute IoU between two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ============================================================================
# Merged detection helper
# ============================================================================

@dataclass
class _MergedBox:
    """Accumulates overlapping detections for fusion."""

    schema_class_id: int
    class_name: str
    xyxy: Tuple[float, float, float, float]
    best_confidence: float
    source_models: List[str] = field(default_factory=list)
    _confidences: List[float] = field(default_factory=list)

    @property
    def fused_confidence(self) -> float:
        """Noisy-OR fusion: ``1 - ∏(1 - c_i)``."""
        product = 1.0
        for c in self._confidences:
            product *= (1.0 - c)
        return 1.0 - product

    @property
    def bbox_centre(self) -> Tuple[float, float, float, float]:
        """Return ``(cx, cy, w, h)`` from internal xyxy."""
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


# ============================================================================
# EnsembleDetector
# ============================================================================

class EnsembleDetector(DetectionModel):
    """Run multiple detectors and fuse their outputs.

    Each sub-model is registered with its own ``ClassMapper`` so heterogeneous
    ID spaces (COCO for YOLO, schema-direct for GDINO) are handled
    transparently.

    Fusion strategy:
    1. Each sub-model's raw ``DetectionResult`` is mapped to **schema class IDs**
       via its ``ClassMapper``.
    2. Mapped results from all models are pooled.
    3. Greedy IoU suppression merges overlapping boxes of the **same schema
       class** and combines their confidences with noisy-OR.

    Attributes:
        iou_threshold: IoU above which two same-class boxes are merged.
    """

    def __init__(
        self,
        models: Optional[List[Tuple[DetectionModel, ClassMapper]]] = None,
        *,
        iou_threshold: float = 0.50,
    ) -> None:
        self._models: List[Tuple[DetectionModel, ClassMapper]] = list(models or [])
        self.iou_threshold = iou_threshold

    # ------------------------------------------------------------------
    # Sub-model management
    # ------------------------------------------------------------------

    def add_model(self, detector: DetectionModel, mapper: ClassMapper) -> None:
        """Register an additional sub-model."""
        self._models.append((detector, mapper))

    @property
    def num_models(self) -> int:
        return len(self._models)

    # ------------------------------------------------------------------
    # DetectionModel ABC
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> List[DetectionResult]:
        """Run all sub-models and fuse results."""
        if not self._models:
            return []

        # 1. Collect mapped detections from every sub-model ----------------
        all_mapped: List[Tuple[DetectionResult, int, str]] = []
        # Each element: (original_det, schema_class_id, model_name)

        for detector, mapper in self._models:
            model_name = detector.get_model_info().name
            raw = detector.detect(image, confidence_threshold)
            for det in raw:
                schema_id = mapper.to_schema(det.class_id)
                if schema_id is None:
                    continue  # unmappable class, skip
                all_mapped.append((det, schema_id, model_name))

        log_debug(
            f"Ensemble: {len(all_mapped)} raw detections from "
            f"{len(self._models)} models"
        )

        if not all_mapped:
            return []

        # 2. Greedy IoU merge (same schema class) -------------------------
        merged = self._fuse(all_mapped)

        # 3. Convert back to DetectionResult with schema IDs ---------------
        results: List[DetectionResult] = []
        for m in merged:
            results.append(DetectionResult(
                class_id=m.schema_class_id,
                class_name=m.class_name,
                bbox=m.bbox_centre,
                confidence=m.fused_confidence,
                attributes={
                    "source_models": m.source_models,
                    "multi_model": len(m.source_models) > 1,
                },
            ))

        log_debug(f"Ensemble: fused to {len(results)} detections")
        return results

    def _fuse(
        self,
        mapped: List[Tuple[DetectionResult, int, str]],
    ) -> List[_MergedBox]:
        """Greedy IoU merge on mapped detections.

        Iterates through every detection.  If it overlaps (IoU ≥ threshold)
        with an existing ``_MergedBox`` of the **same schema class**, the
        confidence is added to that box; otherwise a new box is created.
        """
        boxes: List[_MergedBox] = []

        for det, schema_id, model_name in mapped:
            det_xyxy = _bbox_to_xyxy(*det.bbox)
            matched = False

            for mb in boxes:
                if mb.schema_class_id != schema_id:
                    continue
                if _iou(det_xyxy, mb.xyxy) >= self.iou_threshold:
                    # Merge: keep the box with highest confidence,
                    # accumulate confidence list for noisy-OR.
                    mb._confidences.append(det.confidence)
                    if det.confidence > mb.best_confidence:
                        mb.best_confidence = det.confidence
                        mb.xyxy = det_xyxy  # adopt tighter box
                        mb.class_name = det.class_name
                    if model_name not in mb.source_models:
                        mb.source_models.append(model_name)
                    matched = True
                    break

            if not matched:
                boxes.append(_MergedBox(
                    schema_class_id=schema_id,
                    class_name=det.class_name,
                    xyxy=det_xyxy,
                    best_confidence=det.confidence,
                    source_models=[model_name],
                    _confidences=[det.confidence],
                ))

        return boxes

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_model_info(self) -> ModelInfo:
        names = [d.get_model_info().name for d, _ in self._models]
        return ModelInfo(
            name="ensemble(" + "+".join(names) + ")",
            version="1.0",
            source="ensemble",
            confidence_threshold=0.0,
            extra={"sub_models": names, "iou_threshold": self.iou_threshold},
        )

    @property
    def supported_class_ids(self) -> List[int]:
        """Union of all sub-model schema class IDs."""
        ids: set[int] = set()
        for _, mapper in self._models:
            ids.update(mapper.mapped_schema_classes)
        return sorted(ids)

    @property
    def class_names(self) -> Dict[int, str]:
        """Merge class names from all sub-models (last writer wins)."""
        names: Dict[int, str] = {}
        for det, mapper in self._models:
            for model_id, name in det.class_names.items():
                schema_id = mapper.to_schema(model_id)
                if schema_id is not None:
                    names[schema_id] = name
        return names

    def load(self) -> None:
        for det, _ in self._models:
            det.load()

    def unload(self) -> None:
        for det, _ in self._models:
            det.unload()

    @property
    def is_loaded(self) -> bool:
        return all(d.is_loaded for d, _ in self._models)
