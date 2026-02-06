"""Label services for multiespectral annotation system.

Provides:
- LabelService: Main orchestrator for all label operations
- LabelConfig: Schema configuration with classes and attributes
- LabelStorage: Persistence layer for labels per dataset/channel
- DetectionModel: Abstract interface for pluggable detection models
- SchemaConverter: Abstract interface for format conversions (COCO, YOLO, etc.)
- BBox transformation: Project labels between cameras (computed, not stored)

Use LabelService as the main entry point.

Module Structure:
    labels/
    ├── label_service.py    - Main orchestrator (use this!)
    ├── label_types.py      - Core types (Annotation, ImageLabels, etc.)
    ├── label_storage.py    - Persistence layer
    ├── bbox_transform.py   - Cross-channel projection
    ├── detection/          - Detection model interfaces
    │   ├── base.py         - DetectionModel ABC, ClassMapper
    │   └── yolov8.py       - YOLOv8 implementation
    └── converters/         - Format converters
        ├── base.py         - SchemaConverter/Exporter ABCs
        ├── coco.py         - COCO format converter
        └── yolo.py         - YOLO format converter/exporter
"""

# Main service - primary entry point
from .label_service import LabelService

# Core label types
from .label_types import (
    Annotation,
    AnnotationSource,
    AttributeDefinition,
    AttributeType,
    ClassDefinition,
    ImageLabels,
    LabelConfig,
)
from .label_storage import LabelStorage

# Detection interfaces
from .detection import (
    ClassMapper,
    DetectionModel,
    DetectionResult,
    ModelInfo,
    get_default_coco_mapper,
    detection_to_annotation,
    detections_to_annotations,
)

# Format converters
from .converters import (
    BatchConversionResult,
    ConversionResult,
    SchemaConverter,
    SchemaExporter,
    CocoToSchemaConverter,
    YoloToSchemaConverter,
    SchemaToYoloExporter,
)

# BBox transformation
from .bbox_transform import (
    project_bbox_to_other_channel,
    project_annotations_to_other_channel,
    project_bbox_with_homography,
)

__all__ = [
    # Main service
    "LabelService",
    # Core types
    "Annotation",
    "AnnotationSource",
    "AttributeDefinition",
    "AttributeType",
    "ClassDefinition",
    "ImageLabels",
    "LabelConfig",
    "LabelStorage",
    # Detection
    "ClassMapper",
    "DetectionModel",
    "DetectionResult",
    "ModelInfo",
    "get_default_coco_mapper",
    "detection_to_annotation",
    "detections_to_annotations",
    # Converters
    "BatchConversionResult",
    "ConversionResult",
    "SchemaConverter",
    "SchemaExporter",
    "CocoToSchemaConverter",
    "YoloToSchemaConverter",
    "SchemaToYoloExporter",
    # BBox transform
    "project_bbox_to_other_channel",
    "project_annotations_to_other_channel",
    "project_bbox_with_homography",
]
