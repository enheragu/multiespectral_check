"""Base classes for object detection models.

This module defines the abstract interface for detection models.
Concrete implementations should be in separate files (yolov8.py, etc).

Follows DESIGN_PHILOSOPHY.md:
- Clean abstraction for pluggable models
- No Qt dependencies (pure Python backend)
- Type annotations for clarity
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from backend.services.labels.label_types import Annotation


# =============================================================================
# Detection Result Types
# =============================================================================

@dataclass
class DetectionResult:
    """Result of a single object detection.

    Coordinates are normalized relative to the input image dimensions.

    Attributes:
        class_id: Detected class ID (model-specific, needs mapping to schema)
        class_name: Detected class name from model
        bbox: Bounding box as (x_center, y_center, width, height) normalized [0,1]
        confidence: Detection confidence score [0,1]
        attributes: Optional additional attributes from detection
    """
    class_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]
    confidence: float
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "attributes": dict(self.attributes) if self.attributes else {},
        }


@dataclass
class ModelInfo:
    """Metadata about a detection model for provenance tracking.

    Stored with auto-detected annotations to know which model produced them.
    """
    name: str
    version: str
    source: str
    confidence_threshold: float
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        result = {
            "name": self.name,
            "version": self.version,
            "source": self.source,
            "confidence_threshold": self.confidence_threshold,
        }
        if self.extra:
            result["extra"] = dict(self.extra)
        return result


# =============================================================================
# Detection Model ABC
# =============================================================================

class DetectionModel(ABC):
    """Abstract base class for object detection models.

    Implementations must provide:
    - detect(): Run detection on an image
    - get_model_info(): Return model metadata
    - supported_class_ids: Property returning supported class IDs
    - class_names: Property returning class ID to name mapping

    Optional to override:
    - load(): Load model weights (called before first detection)
    - unload(): Release model resources
    - is_loaded: Property to check if model is ready
    """

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> List[DetectionResult]:
        """Run object detection on an image.

        Args:
            image: Input image as numpy array (H, W, C) in BGR format
            confidence_threshold: Optional override for confidence threshold

        Returns:
            List of DetectionResult objects for detected objects
        """
        pass

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Return metadata about this model."""
        pass

    @property
    @abstractmethod
    def supported_class_ids(self) -> List[int]:
        """Return list of class IDs this model can detect."""
        pass

    @property
    @abstractmethod
    def class_names(self) -> Dict[int, str]:
        """Return mapping of class ID to class name."""
        pass

    def load(self) -> None:
        """Load model weights into memory. Override if needed."""
        pass

    def unload(self) -> None:
        """Release model resources. Override if needed."""
        pass

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for detection."""
        return True


# =============================================================================
# Class Mapper
# =============================================================================

class ClassMapper:
    """Maps between model class IDs and label schema class IDs.

    Example:
        YOLO COCO uses: 0=person, 2=car, 5=bus
        Our schema uses: 1=person, 2=car, 3=heavy_vehicle

        mapper = ClassMapper({0: 1, 2: 2, 5: 3})
        schema_id = mapper.to_schema(0)  # Returns 1 (person)
    """

    def __init__(
        self,
        model_to_schema: Optional[Dict[int, int]] = None,
        schema_to_model: Optional[Dict[int, int]] = None,
        model_to_attrs: Optional[Dict[int, Dict[str, str]]] = None,
    ) -> None:
        """Initialize mapper with one or both directions.

        Args:
            model_to_schema: {model_class_id: schema_class_id}
            schema_to_model: {schema_class_id: model_class_id}
            model_to_attrs: {model_class_id: {attr_name: attr_value}}
                Optional per-class inferred attributes.  When a model
                class ID is known to correspond to a specific schema
                attribute value (e.g. COCO "bird" → animal with
                ``type=bird``), this dict propagates that information
                into the converted Annotation.
        """
        self._model_to_schema = model_to_schema or {}
        self._schema_to_model = schema_to_model or {}
        self._model_to_attrs: Dict[int, Dict[str, str]] = model_to_attrs or {}

        # Build reverse mapping if only one direction provided
        if self._model_to_schema and not self._schema_to_model:
            self._schema_to_model = {v: k for k, v in self._model_to_schema.items()}
        elif self._schema_to_model and not self._model_to_schema:
            self._model_to_schema = {v: k for k, v in self._schema_to_model.items()}

    def to_schema(self, model_class_id: int) -> Optional[int]:
        """Convert model class ID to schema class ID."""
        return self._model_to_schema.get(model_class_id)

    def to_model(self, schema_class_id: int) -> Optional[int]:
        """Convert schema class ID to model class ID."""
        return self._schema_to_model.get(schema_class_id)

    def has_mapping(self, model_class_id: int) -> bool:
        """Check if model class ID has a schema mapping."""
        return model_class_id in self._model_to_schema

    @property
    def mapped_model_classes(self) -> List[int]:
        """Get list of model class IDs that have mappings."""
        return list(self._model_to_schema.keys())

    @property
    def mapped_schema_classes(self) -> List[int]:
        """Get list of schema class IDs that have mappings."""
        return list(self._schema_to_model.keys())

    def get_inferred_attrs(self, model_class_id: int) -> Dict[str, str]:
        """Return inferred schema attribute values for a model class ID.

        Returns a *copy* of the stored dict so callers can safely mutate
        the result.
        """
        return dict(self._model_to_attrs.get(model_class_id, {}))


# =============================================================================
# COCO Class Constants (used by YOLO and other COCO-trained models)
# =============================================================================

COCO_PERSON = 0
COCO_BICYCLE = 1
COCO_CAR = 2
COCO_MOTORCYCLE = 3
COCO_BUS = 5
COCO_TRUCK = 7
COCO_TRAFFIC_LIGHT = 9
COCO_STOP_SIGN = 11
COCO_PARKING_METER = 12
COCO_BENCH = 13
COCO_FIRE_HYDRANT = 10
COCO_BIRD = 14
COCO_CAT = 15
COCO_DOG = 16
COCO_BACKPACK = 24
COCO_HANDBAG = 26
COCO_SUITCASE = 28

# Our schema class IDs (from labels_multiespectral_dataset.yaml)
SCHEMA_PERSON = 1
SCHEMA_CAR = 2
SCHEMA_HEAVY_VEHICLE = 3
SCHEMA_CYCLE = 4
SCHEMA_ANIMAL = 6
SCHEMA_LOOSE_OBJECT = 7
SCHEMA_EMERGENCY_VEHICLE = 8
SCHEMA_CONSTRUCTION_ELEMENT = 9
SCHEMA_TRAFFIC_LIGHT = 10
SCHEMA_TRAFFIC_SIGN = 11
SCHEMA_STREET_FURNITURE = 12

# Default COCO -> Schema mapping
DEFAULT_COCO_TO_SCHEMA: Dict[int, int] = {
    COCO_PERSON: SCHEMA_PERSON,
    COCO_BICYCLE: SCHEMA_CYCLE,
    COCO_CAR: SCHEMA_CAR,
    COCO_MOTORCYCLE: SCHEMA_CYCLE,
    COCO_BUS: SCHEMA_HEAVY_VEHICLE,
    COCO_TRUCK: SCHEMA_HEAVY_VEHICLE,
    COCO_TRAFFIC_LIGHT: SCHEMA_TRAFFIC_LIGHT,
    COCO_FIRE_HYDRANT: SCHEMA_STREET_FURNITURE,
    COCO_STOP_SIGN: SCHEMA_TRAFFIC_SIGN,
    COCO_PARKING_METER: SCHEMA_STREET_FURNITURE,
    COCO_BENCH: SCHEMA_STREET_FURNITURE,
    COCO_BIRD: SCHEMA_ANIMAL,
    COCO_CAT: SCHEMA_ANIMAL,
    COCO_DOG: SCHEMA_ANIMAL,
    COCO_BACKPACK: SCHEMA_LOOSE_OBJECT,
    COCO_HANDBAG: SCHEMA_LOOSE_OBJECT,
    COCO_SUITCASE: SCHEMA_LOOSE_OBJECT,
}

# Inferred schema attributes per COCO class.
# When YOLO detects e.g. "bird" (COCO 14) we know it maps to
# SCHEMA_ANIMAL *and* that animal.type = "bird".
DEFAULT_COCO_TO_ATTRS: Dict[int, Dict[str, str]] = {
    COCO_BICYCLE: {"propulsion": "bicycle"},
    COCO_MOTORCYCLE: {"propulsion": "motorcycle"},
    COCO_BUS: {"class": "bus"},
    COCO_TRUCK: {"class": "truck"},
    COCO_FIRE_HYDRANT: {"type": "fire_hydrant"},
    COCO_STOP_SIGN: {"category": "stop"},
    COCO_PARKING_METER: {"type": "parking_meter"},
    COCO_BENCH: {"type": "bench"},
    COCO_BIRD: {"type": "bird"},
    COCO_CAT: {"type": "cat"},
    COCO_DOG: {"type": "dog"},
    COCO_BACKPACK: {"type": "bag"},
    COCO_HANDBAG: {"type": "bag"},
    COCO_SUITCASE: {"type": "bag"},
}


def get_default_coco_mapper() -> ClassMapper:
    """Get the default COCO (YOLO) to schema class mapper."""
    return ClassMapper(
        model_to_schema=DEFAULT_COCO_TO_SCHEMA,
        model_to_attrs=DEFAULT_COCO_TO_ATTRS,
    )


# =============================================================================
# Conversion Helpers
# =============================================================================

def detection_to_annotation(
    detection: DetectionResult,
    class_mapper: Optional[ClassMapper] = None,
) -> Optional["Annotation"]:
    """Convert a DetectionResult to an Annotation with source=AUTO.

    Args:
        detection: Detection result from a model
        class_mapper: Optional mapper to convert model class ID to schema ID

    Returns:
        Annotation with source=AUTO, or None if class mapping fails
    """
    from backend.services.labels.label_types import Annotation, AnnotationSource

    class_id = detection.class_id
    if class_mapper is not None:
        mapped_id = class_mapper.to_schema(class_id)
        if mapped_id is None:
            return None
        class_id = mapped_id

    # Merge detection attributes (raw_label, inferred attrs like type,
    # class, service…) and always include confidence.
    attrs: Dict[str, Any] = {"confidence": detection.confidence}
    if detection.attributes:
        attrs.update(detection.attributes)

    # Apply mapper-level inferred attributes (e.g. COCO bird → type=bird).
    # Lower priority: don't override attrs already set by the detector.
    if class_mapper is not None:
        mapper_attrs = class_mapper.get_inferred_attrs(detection.class_id)
        for k, v in mapper_attrs.items():
            if k not in attrs:
                attrs[k] = v

    return Annotation(
        class_id=class_id,
        bbox=detection.bbox,
        source=AnnotationSource.AUTO,
        attributes=attrs,
    )


def detections_to_annotations(
    detections: List[DetectionResult],
    class_mapper: Optional[ClassMapper] = None,
) -> List["Annotation"]:
    """Convert multiple DetectionResults to Annotations with source=AUTO.

    Skips detections that can't be mapped to schema classes.
    """
    annotations = []
    for det in detections:
        ann = detection_to_annotation(det, class_mapper)
        if ann is not None:
            annotations.append(ann)
    return annotations
