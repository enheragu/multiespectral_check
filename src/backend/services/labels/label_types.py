"""Label schema types and dataclasses for multiespectral annotation system.

This module defines the core types for the labeling system:
- LabelConfig: Schema configuration loaded from YAML
- ClassDefinition: Definition of a labelable class with attributes
- AttributeDefinition: Definition of an attribute (enum, bool, float)
- Annotation: A single annotation on an image
- ImageLabels: All annotations for one image file

Follows DESIGN_PHILOSOPHY.md:
- YAML-compatible types (list, dict, primitives)
- Single Source of Truth: derived data (projected labels) NOT stored
- Clean separation: schema config vs runtime annotations
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common.yaml_utils import load_yaml


# =============================================================================
# Attribute Types
# =============================================================================

class AttributeType(Enum):
    """Types of attributes that can be defined for a class."""
    ENUM = "enum"
    BOOL = "bool"
    FLOAT = "float"


@dataclass
class AttributeDefinition:
    """Definition of a class attribute from the schema.

    Attributes:
        name: Attribute name (e.g., "role", "state")
        attr_type: Type of attribute (enum, bool, float)
        options: For enum type, list of valid options
        range: For float type, [min, max] range
        description: Optional description
    """
    name: str
    attr_type: AttributeType
    options: Optional[List[str]] = None
    range: Optional[Tuple[float, float]] = None
    description: Optional[str] = None

    def validate_value(self, value: Any) -> bool:
        """Check if a value is valid for this attribute."""
        if value is None:
            return True  # None means "not set"

        if self.attr_type == AttributeType.ENUM:
            return self.options is not None and value in self.options
        elif self.attr_type == AttributeType.BOOL:
            return isinstance(value, bool)
        elif self.attr_type == AttributeType.FLOAT:
            if not isinstance(value, (int, float)):
                return False
            if self.range:
                return self.range[0] <= float(value) <= self.range[1]
            return True
        return False

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "AttributeDefinition":
        """Create from YAML dict."""
        attr_type = AttributeType(data.get("type", "enum"))
        options = data.get("options")
        raw_range = data.get("range")
        range_tuple = tuple(raw_range) if raw_range and len(raw_range) == 2 else None
        return cls(
            name=name,
            attr_type=attr_type,
            options=options,
            range=range_tuple,  # type: ignore[arg-type]
            description=data.get("description"),
        )


# =============================================================================
# Class Definition
# =============================================================================

@dataclass
class ClassDefinition:
    """Definition of a labelable class from the schema.

    Attributes:
        id: Unique numeric ID for the class (use class_id property)
        name: Human-readable name (e.g., "person", "car")
        group: Logical group ("dynamic" or "infrastructure")
        description: Optional description
        aliases: Alternative names for matching (e.g., ["pedestrian"] for person)
        attributes: Dict of attribute name -> AttributeDefinition
        min_confidence: Per-class minimum confidence threshold for auto-detection.
            Annotations below this threshold are discarded after detection.
            When *None*, the detector's global threshold is used.
    """
    id: int
    name: str
    group: str = "dynamic"
    description: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    attributes: Dict[str, AttributeDefinition] = field(default_factory=dict)
    min_confidence: Optional[float] = None

    @property
    def class_id(self) -> str:
        """Alias for id to match Annotation.class_id naming - always string."""
        return str(self.id)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassDefinition":
        """Create from YAML dict."""
        attrs = {}
        raw_attrs = data.get("attributes", {})
        if isinstance(raw_attrs, dict):
            for attr_name, attr_data in raw_attrs.items():
                attrs[attr_name] = AttributeDefinition.from_dict(attr_name, attr_data)

        return cls(
            id=data["id"],
            name=data["name"],
            group=data.get("group", "dynamic"),
            description=data.get("description"),
            aliases=data.get("aliases", []),
            attributes=attrs,
            min_confidence=data.get("min_confidence"),
        )

    def get_attribute_options(self, attr_name: str) -> Optional[List[str]]:
        """Get options for an enum attribute."""
        attr = self.attributes.get(attr_name)
        if attr and attr.attr_type == AttributeType.ENUM:
            return attr.options
        return None


# =============================================================================
# Label Configuration (Schema)
# =============================================================================

@dataclass
class LabelConfig:
    """Complete labeling schema configuration.

    Loaded from YAML config file. Defines available classes, their attributes,
    and universal attributes that apply to all annotations.

    Attributes:
        schema_version: Version string for migration support
        dataset_meta: Metadata about the dataset schema
        universal_attrs: Attributes that apply to ALL annotations
        classes: Dict of class_id (str) -> ClassDefinition
        classes_by_name: Dict of class_name -> ClassDefinition (for lookup)
    """
    schema_version: str
    dataset_meta: Dict[str, Any]
    universal_attrs: Dict[str, AttributeDefinition]
    classes: Dict[str, ClassDefinition]
    classes_by_name: Dict[str, ClassDefinition] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Build name lookup index."""
        self.classes_by_name = {c.name: c for c in self.classes.values()}

    @classmethod
    def from_yaml(cls, path: Path) -> "LabelConfig":
        """Load configuration from YAML file.

        Supports two formats:
        1. Full schema format (classes: [{id, name, ...}])
        2. COCO/YOLOv8 format (names: {id: name})
        """
        data = load_yaml(path)
        if not data:
            raise ValueError(f"Empty or invalid label config: {path}")

        # Parse universal attributes
        universal = {}
        raw_universal = data.get("universal_attrs", {})
        for attr_name, attr_data in raw_universal.items():
            universal[attr_name] = AttributeDefinition.from_dict(attr_name, attr_data)

        # Parse classes - support both formats
        classes: Dict[str, ClassDefinition] = {}

        # Format 1: Full schema with classes list
        if "classes" in data and data["classes"]:
            for class_data in data.get("classes", []):
                class_def = ClassDefinition.from_dict(class_data)
                classes[class_def.class_id] = class_def

        # Format 2: COCO/YOLOv8 format with names dict
        elif "names" in data and isinstance(data["names"], dict):
            for id_key, name in data["names"].items():
                class_def = ClassDefinition(
                    id=int(id_key),
                    name=str(name),
                    group="dynamic",
                )
                classes[class_def.class_id] = class_def

        return cls(
            schema_version=data.get("schema_version", "1.0.0"),
            dataset_meta=data.get("dataset_meta", {}),
            universal_attrs=universal,
            classes=classes,
        )

    def get_class(self, class_id: str) -> Optional[ClassDefinition]:
        """Get class definition by ID."""
        return self.classes.get(class_id)

    def get_class_by_name(self, name: str) -> Optional[ClassDefinition]:
        """Get class definition by name."""
        return self.classes_by_name.get(name)

    def get_all_class_names(self) -> List[str]:
        """Get sorted list of all class names."""
        return sorted(self.classes_by_name.keys())

    def get_classes_by_group(self, group: str) -> List[ClassDefinition]:
        """Get all classes in a group (e.g., 'dynamic', 'infrastructure')."""
        return [c for c in self.classes.values() if c.group == group]

    @property
    def bbox_format(self) -> str:
        """Get the bbox format from metadata."""
        return str(self.dataset_meta.get("bbox_format", "normalized_center_xywh"))


# =============================================================================
# Annotation Source (per-annotation tracking)
# =============================================================================

class AnnotationSource(Enum):
    """Source/origin of an annotation.

    Values:
        AUTO: Suggested by detection model, not yet reviewed by user
        REVIEWED: Auto-detected but manually reviewed and accepted
        MANUAL: Manually created from scratch (user drew the bbox)

    Workflow:
        1. Model runs detection -> annotations created with AUTO
        2. User reviews and accepts -> changes to REVIEWED
        3. User draws new bbox -> created with MANUAL
    """
    AUTO = "auto"           # Detected by model, pending review
    REVIEWED = "reviewed"   # Auto-detected, reviewed and accepted
    MANUAL = "manual"       # Hand-labeled from scratch


# =============================================================================
# Annotation (Runtime)
# =============================================================================

@dataclass
class Annotation:
    """A single annotation on an image.

    Represents one labeled object with bounding box and attributes.
    Coordinates are ALWAYS relative to the original image file on disk.

    Attributes:
        class_id: ID of the class (references LabelConfig.classes) - string for flexibility
        bbox: Bounding box as [x_center, y_center, width, height] normalized [0,1]
        source: Origin of annotation (auto/reviewed/manual)
        attributes: Dict of attribute values (universal + class-specific)
        annotation_id: Unique ID within the image (for editing)
        confidence: Detection confidence [0,1] if from auto-detection

    Source workflow:
        - AUTO: Created by detection model, needs user review
        - REVIEWED: User accepted an auto annotation (possibly with edits)
        - MANUAL: User created from scratch (drew bbox manually)
    """
    class_id: str
    bbox: Tuple[float, float, float, float]  # (x_center, y_center, width, height)
    source: AnnotationSource = AnnotationSource.MANUAL
    attributes: Dict[str, Any] = field(default_factory=dict)
    annotation_id: Optional[int] = None  # Assigned when added to ImageLabels
    confidence: float = 1.0  # Detection confidence

    def __post_init__(self) -> None:
        """Guarantee class_id is always stored as str."""
        if not isinstance(self.class_id, str):
            self.class_id = str(self.class_id)

    @property
    def x_center(self) -> float:
        """X center coordinate [0,1]."""
        return self.bbox[0]

    @property
    def y_center(self) -> float:
        """Y center coordinate [0,1]."""
        return self.bbox[1]

    @property
    def width(self) -> float:
        """Width [0,1]."""
        return self.bbox[2]

    @property
    def height(self) -> float:
        """Height [0,1]."""
        return self.bbox[3]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to YAML-compatible dict."""
        result: Dict[str, Any] = {
            "class_id": self.class_id,
            "bbox": list(self.bbox),
            "source": self.source.value,
        }
        if self.confidence < 1.0:
            result["confidence"] = self.confidence
        if self.attributes:
            result["attributes"] = dict(self.attributes)
        if self.annotation_id is not None:
            result["id"] = self.annotation_id
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Annotation":
        """Create from YAML dict."""
        bbox = data.get("bbox", [0, 0, 0, 0])
        source_str = data.get("source", "manual")
        try:
            source = AnnotationSource(source_str)
        except ValueError:
            source = AnnotationSource.MANUAL
        return cls(
            class_id=str(data["class_id"]),  # Ensure string
            bbox=tuple(bbox),  # type: ignore[arg-type]
            source=source,
            attributes=data.get("attributes", {}),
            annotation_id=data.get("id"),
            confidence=data.get("confidence", 1.0),
        )

    def mark_reviewed(self) -> None:
        """Mark this annotation as reviewed (from AUTO -> REVIEWED)."""
        if self.source == AnnotationSource.AUTO:
            self.source = AnnotationSource.REVIEWED

    def is_pending_review(self) -> bool:
        """Check if this annotation needs user review."""
        return self.source == AnnotationSource.AUTO

    def get_corners_normalized(self) -> Tuple[Tuple[float, float], ...]:
        """Get the 4 corners of the bbox as normalized coordinates.

        Returns corners in order: top-left, top-right, bottom-right, bottom-left
        """
        xc, yc, w, h = self.bbox
        half_w, half_h = w / 2, h / 2
        return (
            (xc - half_w, yc - half_h),  # top-left
            (xc + half_w, yc - half_h),  # top-right
            (xc + half_w, yc + half_h),  # bottom-right
            (xc - half_w, yc + half_h),  # bottom-left
        )


# =============================================================================
# Image Labels (Per-Image Container)
# =============================================================================

@dataclass
class ImageLabels:
    """All annotations for a single image file.

    Stored per dataset/channel (e.g., dataset/labels/visible/base_000123.yaml).
    Coordinates are ALWAYS relative to the original image file on disk.

    Each Annotation has its own `source` (auto/reviewed/manual) - the source
    is tracked per-annotation, not per-image.

    Attributes:
        image_file: Original image filename (e.g., "visible_000123.png")
        channel: "visible" or "lwir"
        model_info: If auto-detected, info about the model used
        annotations: List of Annotation objects
        _next_id: Counter for assigning annotation IDs
    """
    image_file: str
    channel: str  # "visible" | "lwir"
    model_info: Optional[Dict[str, Any]] = None
    annotations: List[Annotation] = field(default_factory=list)
    _next_id: int = field(default=1, repr=False)

    def add_annotation(self, annotation: Annotation) -> int:
        """Add annotation and assign ID. Returns the assigned ID."""
        annotation.annotation_id = self._next_id
        self._next_id += 1
        self.annotations.append(annotation)
        return annotation.annotation_id

    def remove_annotation(self, annotation_id: int) -> bool:
        """Remove annotation by ID. Returns True if found and removed."""
        for i, ann in enumerate(self.annotations):
            if ann.annotation_id == annotation_id:
                self.annotations.pop(i)
                return True
        return False

    def get_annotation(self, annotation_id: int) -> Optional[Annotation]:
        """Get annotation by ID."""
        for ann in self.annotations:
            if ann.annotation_id == annotation_id:
                return ann
        return None

    def find_annotation_at(
        self,
        x_norm: float,
        y_norm: float,
    ) -> Optional[Annotation]:
        """Find annotation containing the given normalized point.

        Returns the smallest (by area) annotation containing the point,
        or None if no annotation contains it.
        """
        candidates: List[Tuple[float, Annotation]] = []
        for ann in self.annotations:
            xc, yc, w, h = ann.bbox
            left = xc - w / 2
            right = xc + w / 2
            top = yc - h / 2
            bottom = yc + h / 2
            if left <= x_norm <= right and top <= y_norm <= bottom:
                area = w * h
                candidates.append((area, ann))

        if not candidates:
            return None
        # Return smallest containing box
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to YAML-compatible dict."""
        result: Dict[str, Any] = {
            "image_file": self.image_file,
            "channel": self.channel,
            "annotations": [ann.to_dict() for ann in self.annotations],
        }
        if self.model_info:
            result["model_info"] = self.model_info
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageLabels":
        """Create from YAML dict."""
        annotations = [Annotation.from_dict(a) for a in data.get("annotations", [])]

        # Reconstruct _next_id from existing annotations
        max_id = 0
        for ann in annotations:
            if ann.annotation_id is not None and ann.annotation_id > max_id:
                max_id = ann.annotation_id

        return cls(
            image_file=data.get("image_file", ""),
            channel=data.get("channel", "visible"),
            model_info=data.get("model_info"),
            annotations=annotations,
            _next_id=max_id + 1,
        )

    def is_empty(self) -> bool:
        """Check if there are no annotations."""
        return len(self.annotations) == 0

    def count_by_source(self) -> Dict[AnnotationSource, int]:
        """Count annotations by source type."""
        counts: Dict[AnnotationSource, int] = {
            AnnotationSource.AUTO: 0,
            AnnotationSource.REVIEWED: 0,
            AnnotationSource.MANUAL: 0,
        }
        for ann in self.annotations:
            counts[ann.source] += 1
        return counts

    def get_pending_review(self) -> List[Annotation]:
        """Get all annotations pending review (source=AUTO)."""
        return [ann for ann in self.annotations if ann.is_pending_review()]

    def mark_all_reviewed(self) -> int:
        """Mark all AUTO annotations as REVIEWED. Returns count updated."""
        count = 0
        for ann in self.annotations:
            if ann.source == AnnotationSource.AUTO:
                ann.mark_reviewed()
                count += 1
        return count
