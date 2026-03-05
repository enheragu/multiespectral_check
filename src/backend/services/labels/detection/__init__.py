"""Detection module - object detection models and utilities.

This module provides the abstract interface for detection models
and concrete implementations.

Contents:
- base: DetectionModel ABC, DetectionResult, ModelInfo, ClassMapper
- detector_factory: Single entry point for creating any detector
- yolov8: UltralyticsDetector (YOLOv5–v12, YOLO26, custom .pt)
- grounding_dino: GroundingDinoDetector (requires transformers + torch)
- ensemble_detector: EnsembleDetector (multi-model IoU fusion)
"""
from .base import (
    ClassMapper,
    DetectionModel,
    DetectionResult,
    ModelInfo,
    detection_to_annotation,
    detections_to_annotations,
    get_default_coco_mapper,
    # COCO constants for custom mappers
    COCO_PERSON, COCO_BICYCLE, COCO_CAR, COCO_MOTORCYCLE,
    COCO_BUS, COCO_TRUCK, COCO_TRAFFIC_LIGHT, COCO_FIRE_HYDRANT,
    COCO_STOP_SIGN, COCO_PARKING_METER, COCO_BENCH,
    COCO_BIRD, COCO_CAT, COCO_DOG,
    COCO_BACKPACK, COCO_HANDBAG, COCO_SUITCASE,
    SCHEMA_PERSON, SCHEMA_CAR, SCHEMA_HEAVY_VEHICLE, SCHEMA_CYCLE,
    SCHEMA_ANIMAL, SCHEMA_LOOSE_OBJECT, SCHEMA_EMERGENCY_VEHICLE,
    SCHEMA_CONSTRUCTION_ELEMENT, SCHEMA_TRAFFIC_LIGHT,
    SCHEMA_TRAFFIC_SIGN, SCHEMA_STREET_FURNITURE,
    DEFAULT_COCO_TO_SCHEMA, DEFAULT_COCO_TO_ATTRS,
)

__all__ = [
    # Core types
    "DetectionModel",
    "DetectionResult",
    "ModelInfo",
    "ClassMapper",
    # Conversion helpers
    "detection_to_annotation",
    "detections_to_annotations",
    "get_default_coco_mapper",
    # COCO class IDs
    "COCO_PERSON", "COCO_BICYCLE", "COCO_CAR", "COCO_MOTORCYCLE",
    "COCO_BUS", "COCO_TRUCK", "COCO_TRAFFIC_LIGHT", "COCO_FIRE_HYDRANT",
    "COCO_STOP_SIGN", "COCO_PARKING_METER", "COCO_BENCH",
    "COCO_BIRD", "COCO_CAT", "COCO_DOG",
    "COCO_BACKPACK", "COCO_HANDBAG", "COCO_SUITCASE",
    # Schema class IDs
    "SCHEMA_PERSON", "SCHEMA_CAR", "SCHEMA_HEAVY_VEHICLE", "SCHEMA_CYCLE",
    "SCHEMA_ANIMAL", "SCHEMA_LOOSE_OBJECT", "SCHEMA_EMERGENCY_VEHICLE",
    "SCHEMA_CONSTRUCTION_ELEMENT", "SCHEMA_TRAFFIC_LIGHT",
    "SCHEMA_TRAFFIC_SIGN", "SCHEMA_STREET_FURNITURE",
    # Mapping tables
    "DEFAULT_COCO_TO_SCHEMA", "DEFAULT_COCO_TO_ATTRS",
]
