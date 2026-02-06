"""COCO format converter.

Converts COCO-format annotations to our schema.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from .base import SchemaConverter

if TYPE_CHECKING:
    from backend.services.labels.label_types import LabelConfig

# COCO category ID to name mapping (80 classes)
COCO_CATEGORIES: Dict[int, str] = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup",
    48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana",
    53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot",
    58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair",
    63: "couch", 64: "potted plant", 65: "bed", 67: "dining table",
    70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote",
    76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven",
    80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock",
    86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier",
    90: "toothbrush",
}


class CocoToSchemaConverter(SchemaConverter):
    """Converter from COCO format to our schema.

    COCO uses:
    - category_id: 1-indexed, non-contiguous (1-90 with gaps)
    - bbox: [x, y, width, height] in pixels (top-left origin)

    Our schema uses:
    - class_id: 0-indexed based on LabelConfig.classes
    - bbox: [cx, cy, w, h] normalized 0-1
    """

    def __init__(
        self,
        target_schema: "LabelConfig",
        class_mapping: Optional[Dict[str, int]] = None,
        strict: bool = False,
    ) -> None:
        """
        Args:
            target_schema: Target label configuration
            class_mapping: Optional explicit COCO name → schema class_id mapping
            strict: If True, fail on unmapped classes; if False, try fuzzy match
        """
        super().__init__(target_schema)
        self.strict = strict

        if class_mapping:
            self._class_name_mapping = class_mapping
        else:
            self._class_name_mapping = self._build_default_mapping()

    @property
    def source_format_name(self) -> str:
        return "COCO"

    def _build_default_mapping(self) -> Dict[str, int]:
        """Build default mapping by matching COCO names to schema names."""
        mapping: Dict[str, int] = {}

        for coco_id, coco_name in COCO_CATEGORIES.items():
            coco_normalized = coco_name.lower().replace(" ", "_")

            for schema_class in self.target_schema.classes.values():
                schema_name = schema_class.name.lower().replace(" ", "_")
                schema_aliases = [a.lower() for a in schema_class.aliases]

                if (
                    coco_normalized == schema_name or
                    coco_name.lower() in schema_aliases or
                    coco_normalized in schema_aliases
                ):
                    mapping[coco_name] = schema_class.class_id
                    break

        return mapping

    def convert_class(self, source_class: Any) -> Optional[int]:
        """Map COCO category_id to schema class_id."""
        if source_class in self._class_cache:
            return self._class_cache[source_class]

        result: Optional[int] = None

        if isinstance(source_class, int):
            coco_name = COCO_CATEGORIES.get(source_class)
            if coco_name and coco_name in self._class_name_mapping:
                result = self._class_name_mapping[coco_name]

        elif isinstance(source_class, str):
            if source_class in self._class_name_mapping:
                result = self._class_name_mapping[source_class]

        self._class_cache[source_class] = result
        return result

    def convert_bbox(
        self,
        bbox: Any,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[float, float, float, float]:
        """Convert COCO bbox [x,y,w,h] to normalized [cx,cy,w,h]."""
        if image_size is None:
            raise ValueError("image_size required for COCO bbox conversion")

        img_w, img_h = image_size
        x, y, w, h = bbox

        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h

        return (cx, cy, nw, nh)

    def convert_attributes(
        self,
        source_attrs: Dict[str, Any],
        target_class_id: int,
    ) -> Dict[str, Any]:
        """Convert COCO attributes."""
        attrs: Dict[str, Any] = {}

        if "iscrowd" in source_attrs and source_attrs["iscrowd"]:
            attrs["crowd"] = True

        if "area" in source_attrs:
            attrs["area"] = float(source_attrs["area"])

        if "segmentation" in source_attrs:
            attrs["has_segmentation"] = True

        return attrs
