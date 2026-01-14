"""Labeling controller for running YOLO models and reading/writing labels."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    import torch  # type: ignore
except Exception:  # noqa: BLE001
    torch = None  # type: ignore

from backend.utils.labels import YoloBox, load_class_map, parse_yolo_file, write_yolo_file


@dataclass
class LabelingConfig:
    model_path: Optional[Path] = None
    labels_yaml: Optional[Path] = None
    input_mode: str = "visible"  # visible | lwir
    conf_threshold: float = 0.25
    img_size: int = 640


class LabelingController:
    def __init__(self, dataset_root: Path, config: LabelingConfig) -> None:
        self.dataset_root = dataset_root
        self.config = config
        self._model = None
        self.class_map = load_class_map(config.labels_yaml)

    def load_model(self) -> None:
        if self._model or not self.config.model_path:
            return
        if torch is None:
            raise RuntimeError("PyTorch is not available")
        try:
            self._model = torch.jit.load(str(self.config.model_path)) if self.config.model_path.suffix.endswith(".pt") else None
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to load model: {exc}")

    def label_path(self, base: str, channel: str) -> Path:
        return self.dataset_root / "labels" / channel / f"{channel}_{base}.txt"

    def read_labels(self, base: str, channel: str) -> List[YoloBox]:
        return parse_yolo_file(self.label_path(base, channel))

    def write_labels(self, base: str, channel: str, boxes: List[YoloBox]) -> None:
        write_yolo_file(self.label_path(base, channel), boxes)

    def clear_labels(self, base: str) -> None:
        for channel in ("lwir", "visible"):
            path = self.label_path(base, channel)
            if path.exists():
                path.unlink(missing_ok=True)

    def run_inference_on_image(self, image_path: Path) -> List[YoloBox]:
        # Placeholder: actual inference would use self._model
        # Here we return empty, but keep the interface ready.
        return []

    def run_single(self, base: str, channel: str, image_path: Path) -> List[YoloBox]:
        boxes = self.run_inference_on_image(image_path)
        self.write_labels(base, channel, boxes)
        return boxes

    def class_name(self, cls_id: str) -> str:
        return self.class_map.get(cls_id, cls_id)


def build_controller(dataset_root: Optional[Path], prefs: Dict[str, str]) -> Optional[LabelingController]:
    if not dataset_root:
        return None
    model_path = Path(prefs["label_model"]) if prefs.get("label_model") else None
    labels_yaml = Path(prefs["label_yaml"]) if prefs.get("label_yaml") else None
    config = LabelingConfig(
        model_path=model_path,
        labels_yaml=labels_yaml,
        input_mode=prefs.get("label_input_mode", "visible"),
        conf_threshold=float(prefs.get("label_conf", 0.25)),
        img_size=int(prefs.get("label_img_size", 640)),
    )
    return LabelingController(Path(dataset_root), config)
