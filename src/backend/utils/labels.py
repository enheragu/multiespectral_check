"""YOLO label IO helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

@dataclass
class YoloBox:
    cls: str
    x_center: float
    y_center: float
    width: float
    height: float


def parse_yolo_file(path: Path) -> List[YoloBox]:
    boxes: List[YoloBox] = []
    if not path.exists():
        return boxes
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return boxes
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = parts[0]
        try:
            x_c, y_c, w, h = map(float, parts[1:5])
        except ValueError:
            continue
        boxes.append(YoloBox(cls, x_c, y_c, w, h))
    return boxes


def write_yolo_file(path: Path, boxes: List[YoloBox]) -> None:
    lines = [f"{b.cls} {b.x_center:.6f} {b.y_center:.6f} {b.width:.6f} {b.height:.6f}" for b in boxes]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def class_color(cls_name) -> Tuple[int, int, int]:
    """Deterministic RGB colour from a class name (or anything stringable)."""
    name = cls_name if isinstance(cls_name, str) else str(cls_name)
    seed = sum(ord(c) for c in name) or 1
    r = (seed * 37) % 255
    g = (seed * 57) % 255
    b = (seed * 97) % 255
    return r, g, b

