"""Detection module - object detection models and utilities.

This module provides the abstract interface for detection models
and concrete implementations.

Contents:
- base: DetectionModel ABC, DetectionResult, ModelInfo, ClassMapper
- yolov8: YoloV8Detector implementation (requires ultralytics)
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
    COCO_BUS, COCO_TRUCK, COCO_TRAFFIC_LIGHT, COCO_STOP_SIGN,
    COCO_PARKING_METER, COCO_BENCH, COCO_BIRD, COCO_CAT, COCO_DOG, COCO_BACKPACK,
    SCHEMA_PERSON, SCHEMA_CAR, SCHEMA_HEAVY_VEHICLE, SCHEMA_CYCLE,
    SCHEMA_ANIMAL, SCHEMA_LOOSE_OBJECT, SCHEMA_TRAFFIC_LIGHT,
    SCHEMA_TRAFFIC_SIGN, SCHEMA_STREET_FURNITURE,
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
    "COCO_BUS", "COCO_TRUCK", "COCO_TRAFFIC_LIGHT", "COCO_STOP_SIGN",
    "COCO_PARKING_METER", "COCO_BENCH", "COCO_BIRD", "COCO_CAT", "COCO_DOG", "COCO_BACKPACK",
    # Schema class IDs
    "SCHEMA_PERSON", "SCHEMA_CAR", "SCHEMA_HEAVY_VEHICLE", "SCHEMA_CYCLE",
    "SCHEMA_ANIMAL", "SCHEMA_LOOSE_OBJECT", "SCHEMA_TRAFFIC_LIGHT",
    "SCHEMA_TRAFFIC_SIGN", "SCHEMA_STREET_FURNITURE",
]
