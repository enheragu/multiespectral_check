"""Encapsulates YOLO label IO, class maps, and per-image label caching for the viewer.

Resolves dataset or preferred YAMLs, builds controllers, and exposes helpers to read, cache, and
mutate per-image label overlays used by the UI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt6.QtGui import QColor

from backend.utils.labels import YoloBox, class_color, load_class_map, parse_yolo_file, write_yolo_file
from backend.services.labels.labeling_controller import build_controller, LabelingController


LabelOverlay = Tuple[str, float, float, float, float, QColor]


class LabelWorkflow:
    def __init__(self, dataset_root: Path, default_yaml: Optional[Path], prefs: Dict[str, str]) -> None:
        self.dataset_root = dataset_root
        self.default_yaml = default_yaml
        self.prefs = dict(prefs)
        self.yaml_path = self._resolve_label_yaml_path()
        self.class_map: Dict[str, str] = load_class_map(self.yaml_path)
        self._label_cache: Dict[Tuple[str, str], Tuple[float, Tuple[Tuple[str, float, float, float, float], ...]]] = {}
        self.controller: Optional[LabelingController] = None

    # --- YAML handling ---
    def _dataset_label_yaml_path(self) -> Path:
        return self.dataset_root / "labels" / "labels.yaml"

    def _resolve_label_yaml_path(self) -> Optional[Path]:
        dataset_copy = self._dataset_label_yaml_path()
        if dataset_copy.exists():
            return dataset_copy
        pref_yaml = self.prefs.get("label_yaml")
        if pref_yaml:
            path = Path(pref_yaml)
            if path.exists():
                return path
        if self.default_yaml and self.default_yaml.exists():
            return self.default_yaml
        return None

    def set_label_yaml(self, path: Path) -> None:
        self.yaml_path = path
        self.class_map = load_class_map(path)
        self._label_cache.clear()

    def copy_yaml_to_dataset(self, source: Path) -> None:
        dst = self._dataset_label_yaml_path()
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            write = True
            if dst.exists():
                # Overwrite only if different to avoid churn
                write = dst.read_text(encoding="utf-8") != source.read_text(encoding="utf-8")
            if write:
                dst.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            # Non-fatal
            return

    # --- Controller ---
    def ensure_controller(self) -> bool:
        if self.controller:
            return True
        prefs = dict(self.prefs)
        if self.yaml_path:
            prefs["label_yaml"] = str(self.yaml_path)
        self.controller = build_controller(self.dataset_root, prefs)
        return self.controller is not None

    def rebuild_controller(self) -> None:
        self.controller = None
        self.ensure_controller()

    # --- Paths and caching ---
    def label_path(self, base: str, channel: str) -> Path:
        return self.dataset_root / "labels" / channel / f"{channel}_{base}.txt"

    def _cached_boxes(self, base: str, channel: str, path: Path) -> Tuple[Tuple[str, float, float, float, float], ...]:
        cache_key = (base, channel)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        cached = self._label_cache.get(cache_key)
        if cached and cached[0] == mtime:
            return cached[1]
        parsed = parse_yolo_file(path)
        boxes_raw = tuple((b.cls, b.x_center, b.y_center, b.width, b.height) for b in parsed)
        self._label_cache[cache_key] = (mtime, boxes_raw)
        return boxes_raw

    def read_overlay_boxes(self, base: str, channel: str) -> List[LabelOverlay]:
        path = self.label_path(base, channel)
        if not path.exists():
            return []
        boxes_raw = self._cached_boxes(base, channel, path)
        result: List[LabelOverlay] = []
        for cls_id, x_c, y_c, w, h in boxes_raw:
            cls_name = self.class_map.get(cls_id, cls_id)
            display = f"{cls_id}: {cls_name}" if cls_id != cls_name else cls_name
            r, g, b = class_color(cls_name)
            result.append((display, x_c, y_c, w, h, QColor(r, g, b)))
        return result

    def read_raw_boxes(self, base: str, channel: str) -> List[YoloBox]:
        path = self.label_path(base, channel)
        if not path.exists():
            return []
        return parse_yolo_file(path)

    def label_signature(self, base: str, channel: str, boxes: List[LabelOverlay]) -> Optional[Tuple]:
        if not boxes:
            return None
        path = self.label_path(base, channel)
        try:
            mtime = path.stat().st_mtime if path.exists() else 0.0
        except Exception:
            mtime = 0.0
        compact = tuple((cls, round(x, 4), round(y, 4), round(w, 4), round(h, 4)) for cls, x, y, w, h, _ in boxes)
        return (mtime, compact)

    def clear_labels(self, base: str) -> None:
        for channel in ("lwir", "visible"):
            path = self.label_path(base, channel)
            if path.exists():
                path.unlink(missing_ok=True)
            self._label_cache.pop((base, channel), None)

    def append_box(self, base: str, channel: str, cls_id: str, x_center: float, y_center: float, width: float, height: float) -> None:
        path = self.label_path(base, channel)
        boxes = parse_yolo_file(path)
        boxes.append(YoloBox(cls_id, x_center, y_center, width, height))
        write_yolo_file(path, boxes)
        self._label_cache.pop((base, channel), None)

    def delete_box_at(self, base: str, channel: str, x_norm: float, y_norm: float) -> Optional[str]:
        path = self.label_path(base, channel)
        if not path.exists():
            return None
        boxes = parse_yolo_file(path)
        if not boxes:
            return None
        target_idx = -1
        best_dist = 1e9
        for idx, box in enumerate(boxes):
            left = box.x_center - box.width / 2
            right = box.x_center + box.width / 2
            top = box.y_center - box.height / 2
            bottom = box.y_center + box.height / 2
            if left <= x_norm <= right and top <= y_norm <= bottom:
                target_idx = idx
                break
            dist = (box.x_center - x_norm) ** 2 + (box.y_center - y_norm) ** 2
            if dist < best_dist:
                best_dist = dist
                target_idx = idx
        if target_idx < 0:
            return None
        removed = boxes.pop(target_idx)
        write_yolo_file(path, boxes)
        self._label_cache.pop((base, channel), None)
        cls_name = self.class_map.get(removed.cls, removed.cls)
        return f"{removed.cls}: {cls_name}" if removed.cls != cls_name else removed.cls

    def find_label_display_at(self, base: str, channel: str, x_norm: float, y_norm: float) -> Optional[str]:
        boxes = self.read_raw_boxes(base, channel)
        if not boxes:
            return None
        target_idx = -1
        best_dist = 1e9
        for idx, box in enumerate(boxes):
            left = box.x_center - box.width / 2
            right = box.x_center + box.width / 2
            top = box.y_center - box.height / 2
            bottom = box.y_center + box.height / 2
            if left <= x_norm <= right and top <= y_norm <= bottom:
                target_idx = idx
                break
            dist = (box.x_center - x_norm) ** 2 + (box.y_center - y_norm) ** 2
            if dist < best_dist:
                best_dist = dist
                target_idx = idx
        if target_idx < 0:
            return None
        box = boxes[target_idx]
        cls_name = self.class_map.get(box.cls, box.cls)
        return f"{box.cls}: {cls_name}" if box.cls != cls_name else cls_name

    def update_prefs(self, **kwargs: str) -> None:
        self.prefs.update({k: v for k, v in kwargs.items() if v is not None})
        self.rebuild_controller()

    def class_id_for_value(self, value: str) -> Optional[str]:
        value = value.strip()
        if ":" in value:
            leading = value.split(":", 1)[0].strip()
            if leading:
                value = leading
        if value in self.class_map:
            return value
        for cls_id, name in self.class_map.items():
            if value.lower() == str(name).lower():
                return cls_id
        if self.class_map:
            return None
        return value or None

    def class_choices(self) -> List[str]:
        return [f"{cls_id}: {name}" for cls_id, name in sorted(self.class_map.items(), key=lambda kv: kv[0])]

    def clear_cache(self) -> None:
        self._label_cache.clear()