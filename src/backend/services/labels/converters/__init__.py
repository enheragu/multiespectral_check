"""Converters module - schema conversion between annotation formats.

This module provides converters for transforming annotations between
different formats (COCO, YOLO, our schema, etc.).

Contents:
- base: SchemaConverter ABC, SchemaExporter ABC, ConversionResult
- coco: CocoToSchemaConverter
- yolo: YoloToSchemaConverter, SchemaToYoloExporter
"""
from .base import (
    BatchConversionResult,
    ConversionResult,
    SchemaConverter,
    SchemaExporter,
)
from .coco import CocoToSchemaConverter
from .yolo import YoloToSchemaConverter, SchemaToYoloExporter

__all__ = [
    # Base types
    "ConversionResult",
    "BatchConversionResult",
    "SchemaConverter",
    "SchemaExporter",
    # Converters
    "CocoToSchemaConverter",
    "YoloToSchemaConverter",
    "SchemaToYoloExporter",
]
