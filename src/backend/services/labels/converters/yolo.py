"""YOLO format converter and exporter.

Handles conversion between YOLO format and our schema.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .base import BatchConversionResult, SchemaConverter, SchemaExporter

if TYPE_CHECKING:
    from backend.services.labels.label_types import Annotation, LabelConfig


class YoloToSchemaConverter(SchemaConverter):
    """Converter from YOLO format to our schema.

    YOLO format:
    - class_id: 0-indexed integer
    - bbox: [cx, cy, w, h] normalized 0-1

    Since YOLO format matches our schema, this converter primarily
    handles class ID mapping.
    """

    def __init__(
        self,
        target_schema: "LabelConfig",
        yolo_names: Optional[List[str]] = None,
        class_mapping: Optional[Dict[int, int]] = None,
    ) -> None:
        """
        Args:
            target_schema: Target label configuration
            yolo_names: Optional list of YOLO class names (from data.yaml)
            class_mapping: Optional explicit YOLO class_id → schema class_id mapping
        """
        super().__init__(target_schema)
        self.yolo_names = yolo_names or []
        self._class_id_mapping = class_mapping or {}

        if yolo_names and not class_mapping:
            self._class_id_mapping = self._build_name_based_mapping()

    @property
    def source_format_name(self) -> str:
        return "YOLO"

    def _build_name_based_mapping(self) -> Dict[int, int]:
        """Build mapping by matching YOLO names to schema names."""
        mapping: Dict[int, int] = {}

        for yolo_idx, yolo_name in enumerate(self.yolo_names):
            yolo_normalized = yolo_name.lower().replace(" ", "_")

            for schema_class in self.target_schema.classes.values():
                schema_name = schema_class.name.lower().replace(" ", "_")
                schema_aliases = [a.lower() for a in schema_class.aliases]

                if (
                    yolo_normalized == schema_name or
                    yolo_name.lower() in schema_aliases or
                    yolo_normalized in schema_aliases
                ):
                    mapping[yolo_idx] = schema_class.class_id
                    break

        return mapping

    def convert_class(self, source_class: Any) -> Optional[int]:
        """Map YOLO class_id to schema class_id."""
        if source_class in self._class_cache:
            return self._class_cache[source_class]

        result: Optional[int] = None

        if isinstance(source_class, int):
            if source_class in self._class_id_mapping:
                result = self._class_id_mapping[source_class]
            elif source_class in self.target_schema.classes:
                # Direct class_id match
                result = source_class

        elif isinstance(source_class, str):
            source_normalized = source_class.lower().replace(" ", "_")
            for schema_class in self.target_schema.classes.values():
                schema_name = schema_class.name.lower().replace(" ", "_")
                if source_normalized == schema_name:
                    result = schema_class.class_id
                    break

        self._class_cache[source_class] = result
        return result

    def convert_bbox(
        self,
        bbox: Any,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[float, float, float, float]:
        """Convert YOLO bbox - already in normalized [cx,cy,w,h] format."""
        if len(bbox) >= 4:
            return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        raise ValueError(f"Invalid YOLO bbox: {bbox}")

    @classmethod
    def parse_yolo_line(cls, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single line from YOLO annotation file."""
        parts = line.strip().split()
        if len(parts) < 5:
            return None

        try:
            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            confidence = float(parts[5]) if len(parts) > 5 else None

            result: Dict[str, Any] = {
                "class_id": class_id,
                "bbox": [cx, cy, w, h],
            }
            if confidence is not None:
                result["confidence"] = confidence

            return result
        except (ValueError, IndexError):
            return None

    def convert_from_file(self, file_path: Path) -> BatchConversionResult:
        """Convert annotations from a YOLO .txt file."""
        annotations = []

        if file_path.exists():
            with open(file_path) as f:
                for line in f:
                    parsed = self.parse_yolo_line(line)
                    if parsed:
                        annotations.append(parsed)

        return self.convert_batch(annotations)


class SchemaToYoloExporter(SchemaExporter):
    """Exporter from our schema to YOLO format.

    Exports annotations to YOLO .txt format:
    - One line per annotation: class_id cx cy w h
    - All values normalized 0-1
    """

    def __init__(
        self,
        source_schema: "LabelConfig",
        class_mapping: Optional[Dict[int, int]] = None,
    ) -> None:
        """
        Args:
            source_schema: Source label configuration
            class_mapping: Optional schema class_id → YOLO class_id mapping
        """
        super().__init__(source_schema)
        self._class_mapping = class_mapping or {}

    @property
    def target_format_name(self) -> str:
        return "YOLO"

    def map_class(self, schema_class_id: int) -> int:
        """Map schema class_id to YOLO class_id."""
        return self._class_mapping.get(schema_class_id, schema_class_id)

    def export_annotation(
        self,
        annotation: "Annotation",
        image_size: Tuple[int, int],
    ) -> str:
        """Export single annotation to YOLO format string."""
        yolo_class = self.map_class(annotation.class_id)
        cx, cy, w, h = annotation.bbox

        confidence = annotation.attributes.get("confidence")
        if confidence is not None:
            return f"{yolo_class} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {confidence:.4f}"
        return f"{yolo_class} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    def export_to_file(
        self,
        annotations: List["Annotation"],
        file_path: Path,
        image_size: Tuple[int, int],
    ) -> None:
        """Export annotations to a YOLO .txt file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [self.export_annotation(a, image_size) for a in annotations]

        with open(file_path, "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

    def export_data_yaml(
        self,
        output_path: Path,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
    ) -> None:
        """Export data.yaml file for YOLO training."""
        import yaml

        names = {
            self.map_class(class_id): cls.name
            for class_id, cls in self.source_schema.classes.items()
        }

        data = {
            "names": names,
            "nc": len(names),
        }

        if train_path:
            data["train"] = train_path
        if val_path:
            data["val"] = val_path
        if test_path:
            data["test"] = test_path

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
