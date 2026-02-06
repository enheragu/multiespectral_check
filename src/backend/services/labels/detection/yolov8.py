"""YOLOv8 detector implementation using ultralytics library.

This module provides a concrete DetectionModel implementation for YOLOv8.
Requires: pip install ultralytics

Usage:
    detector = YoloV8Detector("yolov8n.pt")  # nano model
    results = detector.detect(image)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from common.log_utils import log_debug, log_info

from .base import (
    ClassMapper,
    DetectionModel,
    DetectionResult,
    ModelInfo,
    get_default_coco_mapper,
)

# Lazy import container for ultralytics to avoid startup cost
_lazy_import_cache: Dict[str, Any] = {}


def _get_yolo_class():
    """Lazy import of ultralytics YOLO class."""
    if "YOLO" not in _lazy_import_cache:
        try:
            from ultralytics import YOLO
            _lazy_import_cache["YOLO"] = YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics not installed. Install with: pip install ultralytics"
            ) from exc
    return _lazy_import_cache["YOLO"]
    return _YOLO


class YoloV8Detector(DetectionModel):
    """YOLOv8 object detection model.

    Supports all YOLOv8 model sizes (n, s, m, l, x) and custom trained models.

    Attributes:
        model_path: Path to model weights (.pt file)
        confidence_threshold: Default confidence threshold for detections
        device: Device to run on ("cpu", "cuda", "cuda:0", etc.)
    """

    # Standard YOLOv8 COCO class names
    COCO_NAMES: Dict[int, str] = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
        5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
        10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
        14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
        20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
        25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
        30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
        34: "baseball bat", 35: "baseball glove", 36: "skateboard",
        37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
        41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
        46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
        51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
        56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
        60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
        65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
        69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
        74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
        78: "hair drier", 79: "toothbrush",
    }

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
        class_mapper: Optional[ClassMapper] = None,
    ) -> None:
        """Initialize YOLOv8 detector.

        Args:
            model_path: Path to model weights or model name (e.g., "yolov8n.pt")
            confidence_threshold: Default confidence threshold [0,1]
            device: Device to run on (None=auto, "cpu", "cuda", "cuda:0")
            class_mapper: Optional mapper for class ID conversion
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.class_mapper = class_mapper or get_default_coco_mapper()

        self._model: Any = None
        self._model_info: Optional[ModelInfo] = None

    def load(self) -> None:
        """Load the YOLO model into memory."""
        if self._model is not None:
            return  # Already loaded

        YOLO = _get_yolo_class()
        log_info(f"Loading YOLOv8 model: {self.model_path}", "DETECTION")

        self._model = YOLO(self.model_path)

        # Set device if specified
        if self.device:
            self._model.to(self.device)

        # Build model info
        model_name = Path(self.model_path).stem
        self._model_info = ModelInfo(
            name=model_name,
            version=self._get_model_version(),
            source="ultralytics",
            confidence_threshold=self.confidence_threshold,
            extra={"device": str(self._model.device) if hasattr(self._model, "device") else "unknown"},
        )

        log_info(f"YOLOv8 model loaded: {model_name}", "DETECTION")

    def unload(self) -> None:
        """Release model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_info = None
            log_debug("YOLOv8 model unloaded", "DETECTION")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> List[DetectionResult]:
        """Run object detection on an image.

        Args:
            image: Input image as numpy array (H, W, C) in BGR format
            confidence_threshold: Override default confidence threshold

        Returns:
            List of DetectionResult objects
        """
        # Auto-load if needed
        if not self.is_loaded:
            self.load()

        conf = confidence_threshold or self.confidence_threshold
        h, w = image.shape[:2]

        # Run inference
        results = self._model(
            image,
            conf=conf,
            verbose=False,
        )

        detections: List[DetectionResult] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                # Extract box data
                xyxy = boxes.xyxy[i].cpu().numpy()
                cls_id = int(boxes.cls[i].cpu().numpy())
                conf_score = float(boxes.conf[i].cpu().numpy())

                # Convert xyxy to normalized center format
                x1, y1, x2, y2 = xyxy
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h

                # Get class name
                class_name = self._get_class_name(cls_id)

                detections.append(DetectionResult(
                    class_id=cls_id,
                    class_name=class_name,
                    bbox=(x_center, y_center, width, height),
                    confidence=conf_score,
                ))

        log_debug(f"YOLOv8 detected {len(detections)} objects", "DETECTION")
        return detections

    def get_model_info(self) -> ModelInfo:
        """Return model metadata."""
        if self._model_info is None:
            # Return placeholder if not loaded
            return ModelInfo(
                name=Path(self.model_path).stem,
                version="unknown",
                source="ultralytics",
                confidence_threshold=self.confidence_threshold,
            )
        return self._model_info

    @property
    def supported_class_ids(self) -> List[int]:
        """Return list of COCO class IDs supported by this model."""
        return list(self.COCO_NAMES.keys())

    @property
    def class_names(self) -> Dict[int, str]:
        """Return COCO class names."""
        if self._model is not None and hasattr(self._model, "names"):
            return dict(self._model.names)
        return self.COCO_NAMES

    def _get_class_name(self, class_id: int) -> str:
        """Get class name for a class ID."""
        if self._model is not None and hasattr(self._model, "names"):
            return self._model.names.get(class_id, f"class_{class_id}")
        return self.COCO_NAMES.get(class_id, f"class_{class_id}")

    def _get_model_version(self) -> str:
        """Get ultralytics version."""
        try:
            import ultralytics
            return ultralytics.__version__
        except (ImportError, AttributeError):
            return "unknown"

    def detect_and_convert(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> List["Annotation"]:
        """Detect objects and convert to Annotations with source=AUTO.

        Convenience method that combines detect() + class mapping.

        Args:
            image: Input image
            confidence_threshold: Override confidence threshold

        Returns:
            List of Annotations with source=AUTO, mapped to schema classes
        """
        from backend.services.labels.label_types import Annotation, AnnotationSource

        detections = self.detect(image, confidence_threshold)
        annotations: List[Annotation] = []

        for det in detections:
            schema_class = self.class_mapper.to_schema(det.class_id)
            if schema_class is not None:
                annotations.append(Annotation(
                    class_id=schema_class,
                    bbox=det.bbox,
                    source=AnnotationSource.AUTO,
                    attributes={"confidence": det.confidence},
                ))

        return annotations
