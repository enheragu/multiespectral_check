"""Base classes for schema converters and exporters.

This module defines the abstract interfaces for converting annotations
between different formats.

Follows DESIGN_PHILOSOPHY.md:
- Clean abstraction for pluggable converters
- Separation of concerns from detection
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from backend.services.labels.label_types import Annotation, LabelConfig


# =============================================================================
# Conversion Result Types
# =============================================================================

@dataclass
class ConversionResult:
    """Result of converting a single annotation."""
    success: bool
    annotation: Optional["Annotation"] = None
    source_class: Optional[str] = None
    target_class: Optional[int] = None
    warning: Optional[str] = None
    error: Optional[str] = None


@dataclass
class BatchConversionResult:
    """Result of converting multiple annotations."""
    converted: List["Annotation"] = field(default_factory=list)
    warnings: List[Tuple[str, str]] = field(default_factory=list)
    errors: List[Tuple[str, str]] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stats:
            self.stats = {"total": 0, "converted": 0, "skipped": 0, "errors": 0}


# =============================================================================
# Abstract Converter Interface
# =============================================================================

class SchemaConverter(ABC):
    """Abstract base class for schema converters.

    Converters handle mapping from a source format (COCO, YOLO, etc.)
    to our target schema.
    """

    def __init__(self, target_schema: "LabelConfig") -> None:
        """Initialize converter with target schema."""
        self.target_schema = target_schema
        self._class_cache: Dict[Any, Optional[int]] = {}

    @property
    @abstractmethod
    def source_format_name(self) -> str:
        """Return name of source format (e.g., 'COCO', 'YOLO')."""
        pass

    @abstractmethod
    def convert_class(self, source_class: Any) -> Optional[int]:
        """Map source class ID/name to target schema class ID."""
        pass

    @abstractmethod
    def convert_bbox(
        self,
        bbox: Any,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[float, float, float, float]:
        """Convert bbox from source format to normalized center format."""
        pass

    def convert_attributes(
        self,
        source_attrs: Dict[str, Any],
        target_class_id: int,
    ) -> Dict[str, Any]:
        """Convert source attributes to target schema attributes."""
        return {}

    def convert_annotation(
        self,
        source_data: Dict[str, Any],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> ConversionResult:
        """Convert a single annotation from source format."""
        from backend.services.labels.label_types import Annotation

        source_class = self._extract_source_class(source_data)
        if source_class is None:
            return ConversionResult(
                success=False,
                source_class=str(source_data),
                error="Could not extract class from source data",
            )

        target_class = self.convert_class(source_class)
        if target_class is None:
            return ConversionResult(
                success=False,
                source_class=str(source_class),
                error=f"No mapping for source class '{source_class}'",
            )

        source_bbox = self._extract_source_bbox(source_data)
        if source_bbox is None:
            return ConversionResult(
                success=False,
                source_class=str(source_class),
                target_class=target_class,
                error="Could not extract bbox from source data",
            )

        try:
            bbox = self.convert_bbox(source_bbox, image_size)
        except Exception as e:
            return ConversionResult(
                success=False,
                source_class=str(source_class),
                target_class=target_class,
                error=f"Bbox conversion failed: {e}",
            )

        source_attrs = self._extract_source_attributes(source_data)
        attrs = self.convert_attributes(source_attrs, target_class)

        confidence = source_data.get("confidence") or source_data.get("score")
        if confidence is not None:
            attrs["confidence"] = float(confidence)

        annotation = Annotation(class_id=target_class, bbox=bbox, attributes=attrs)

        return ConversionResult(
            success=True,
            annotation=annotation,
            source_class=str(source_class),
            target_class=target_class,
        )

    def convert_batch(
        self,
        source_annotations: List[Dict[str, Any]],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> BatchConversionResult:
        """Convert multiple annotations."""
        result = BatchConversionResult()
        result.stats["total"] = len(source_annotations)

        for source_data in source_annotations:
            conv = self.convert_annotation(source_data, image_size)

            if conv.success and conv.annotation:
                result.converted.append(conv.annotation)
                result.stats["converted"] += 1
                if conv.warning:
                    result.warnings.append((conv.source_class or "?", conv.warning))
            elif conv.error:
                result.errors.append((conv.source_class or "?", conv.error))
                result.stats["errors"] += 1
            else:
                result.stats["skipped"] += 1

        return result

    def _extract_source_class(self, source_data: Dict[str, Any]) -> Optional[Any]:
        """Extract class identifier from source data."""
        return (
            source_data.get("category_id") or
            source_data.get("class_id") or
            source_data.get("class") or
            source_data.get("label")
        )

    def _extract_source_bbox(self, source_data: Dict[str, Any]) -> Optional[Any]:
        """Extract bbox from source data."""
        return source_data.get("bbox") or source_data.get("box")

    def _extract_source_attributes(self, source_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract attributes from source data."""
        return dict(source_data.get("attributes", {}))


# =============================================================================
# Abstract Exporter Interface
# =============================================================================

class SchemaExporter(ABC):
    """Abstract base class for exporting our annotations to external formats."""

    def __init__(self, source_schema: "LabelConfig") -> None:
        self.source_schema = source_schema

    @property
    @abstractmethod
    def target_format_name(self) -> str:
        """Return name of target format (e.g., 'COCO', 'YOLO')."""
        pass

    @abstractmethod
    def export_annotation(
        self,
        annotation: "Annotation",
        image_size: Tuple[int, int],
    ) -> Any:
        """Export single annotation to target format."""
        pass

    def export_batch(
        self,
        annotations: List["Annotation"],
        image_size: Tuple[int, int],
    ) -> List[Any]:
        """Export multiple annotations."""
        return [self.export_annotation(a, image_size) for a in annotations]
