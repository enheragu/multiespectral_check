"""Detector factory — single source of truth for available detection models.

Encapsulates all model-specific knowledge: what models exist, what they
need, how to instantiate them.  Both ``LabelService`` and the frontend
talk **only** to this module; neither knows about YOLO or Grounding DINO
internals.

Supported families:
- **Grounding DINO** (HuggingFace) — open-vocabulary zero-shot detector.
- **YOLO** (ultralytics) — YOLO26, YOLOv8, or any custom .pt weights.

Follows DESIGN_PHILOSOPHY.md:
- Pure backend (no Qt dependencies)
- Whoever produces data, encapsulates the behaviour
- One operation = one function that does everything
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common.log_utils import log_info

from .base import ClassMapper, DetectionModel, get_default_coco_mapper

# Re-export for convenience so callers don't need to import base directly
__all__ = [
    "ModelSpec",
    "list_available_models",
    "create_detector",
    "needs_file_path",
]


# ============================================================================
# Model specifications
# ============================================================================

@dataclass(frozen=True)
class ModelSpec:
    """Immutable description of a loadable detection model.

    Attributes:
        key: Unique identifier used by ``create_detector`` (e.g. ``"gdino-base"``).
        display_name: Human-readable name shown in UI combo boxes.
        needs_file: If *True* the frontend must collect a file path from the user
                    (e.g. a ``.pt`` weights file for YOLO).  If *False* the model
                    is auto-downloaded / self-contained.
        needs_config: If *True* a ``LabelConfig`` must be loaded before this model
                      can be instantiated (e.g. Grounding DINO needs class names).
        description: One-line tooltip for the UI.
    """
    key: str
    display_name: str
    needs_file: bool = False
    needs_config: bool = False
    description: str = ""


# Registry ---------------------------------------------------------------
# Add new models here — nothing else needs to change.

_MODEL_REGISTRY: List[ModelSpec] = [
    # -- Grounding DINO (HuggingFace, open-vocabulary) ---------------------
    ModelSpec(
        key="gdino-base",
        display_name="Grounding DINO — base (recommended)",
        needs_config=True,
        description=(
            "Open-vocabulary zero-shot detector.  Builds text prompts from "
            "label schema.  Auto-downloads weights (~1 GB) on first use."
        ),
    ),
    ModelSpec(
        key="gdino-tiny",
        display_name="Grounding DINO — tiny (faster)",
        needs_config=True,
        description=(
            "Lighter variant of Grounding DINO.  Faster inference, "
            "slightly lower accuracy."
        ),
    ),
    # -- YOLO26 (ultralytics, auto-download, COCO-80) ---------------------
    ModelSpec(
        key="yolo26l",
        display_name="YOLO26 — large  (55.0 mAP, recommended)",
        description=(
            "YOLO26 large: 55.0 mAP on COCO.  NMS-free, end-to-end.  "
            "Auto-downloads weights (~50 MB) on first use."
        ),
    ),
    ModelSpec(
        key="yolo26s",
        display_name="YOLO26 — small  (48.6 mAP, fast)",
        description=(
            "YOLO26 small: 48.6 mAP on COCO.  Faster inference, "
            "smaller footprint.  Auto-downloads on first use."
        ),
    ),
    # -- Custom YOLO (bring your own .pt) ---------------------------------
    ModelSpec(
        key="yolo-custom",
        display_name="YOLO — custom .pt file",
        needs_file=True,
        description=(
            "Load any ultralytics-compatible weights (.pt).  "
            "Works with YOLOv5–v12, YOLO26, and fine-tuned models."
        ),
    ),
    # -- Ensemble (multi-model fusion) ------------------------------------
    ModelSpec(
        key="ensemble-gdino-yolo26",
        display_name="Ensemble — GDINO base + YOLO26 large",
        needs_config=True,
        description=(
            "Runs Grounding DINO and YOLO26 together.  Fuses overlapping "
            "boxes by IoU for higher confidence on shared classes."
        ),
    ),
]

_SPEC_BY_KEY: Dict[str, ModelSpec] = {s.key: s for s in _MODEL_REGISTRY}

# Ensemble recipes: composite key → list of sub-model keys to combine.
_ENSEMBLE_RECIPES: Dict[str, List[str]] = {
    "ensemble-gdino-yolo26": ["gdino-base", "yolo26l"],
}

# HuggingFace model-id lookup for Grounding DINO variants
_GDINO_HF_IDS: Dict[str, str] = {
    "gdino-base": "IDEA-Research/grounding-dino-base",
    "gdino-tiny": "IDEA-Research/grounding-dino-tiny",
}

# Ultralytics auto-download model names for YOLO variants
_YOLO_AUTO_MODELS: Dict[str, str] = {
    "yolo26l": "yolo26l.pt",
    "yolo26s": "yolo26s.pt",
}


# ============================================================================
# Public API
# ============================================================================

def list_available_models() -> List[ModelSpec]:
    """Return specs for every registered model (order = display order)."""
    return list(_MODEL_REGISTRY)


def needs_file_path(key: str) -> bool:
    """Return *True* if the model identified by *key* needs a file path."""
    spec = _SPEC_BY_KEY.get(key)
    return spec.needs_file if spec else False


def create_detector(
    key: str,
    *,
    label_config: Any = None,
    file_path: Optional[str] = None,
    confidence_threshold: Optional[float] = None,
    image_size: Optional[Tuple[int, int]] = None,
) -> Tuple[DetectionModel, ClassMapper]:
    """Instantiate a detector and its matching ``ClassMapper``.

    This is the **only** function that imports concrete detector classes.

    Args:
        key: Model key from ``ModelSpec.key``.
        label_config: A ``LabelConfig`` instance (required for GDINO).
        file_path: Path to weights file (required for YOLO).
        confidence_threshold: Override the model's default threshold.
        image_size: Dataset image dimensions ``(width, height)``.
                    Forwarded to detectors that support resolution
                    override (e.g. Grounding DINO).

    Returns:
        ``(detector, class_mapper)`` ready to pass to
        ``LabelService.set_detector()``.

    Raises:
        ValueError: Unknown key or missing required arguments.
    """
    if key not in _SPEC_BY_KEY:
        raise ValueError(
            f"Unknown model key {key!r}. "
            f"Available: {[s.key for s in _MODEL_REGISTRY]}"
        )

    spec = _SPEC_BY_KEY[key]

    if spec.needs_file and not file_path:
        raise ValueError(f"Model {key!r} requires a file_path argument.")
    if spec.needs_config and label_config is None:
        raise ValueError(
            f"Model {key!r} requires a loaded LabelConfig. "
            "Load a labels YAML first."
        )

    # ----- Grounding DINO variants ----------------------------------------
    if key.startswith("gdino"):
        return _create_gdino(key, label_config, confidence_threshold, image_size)

    # ----- YOLO (ultralytics) — auto-download or custom .pt ---------------
    if key in _YOLO_AUTO_MODELS:
        return _create_yolo(_YOLO_AUTO_MODELS[key], confidence_threshold)
    if key == "yolo-custom":
        return _create_yolo(file_path, confidence_threshold)  # type: ignore[arg-type]

    # ----- Ensemble (multi-model fusion) ----------------------------------
    if key in _ENSEMBLE_RECIPES:
        return _create_ensemble(
            key, label_config, file_path, confidence_threshold, image_size,
        )

    raise ValueError(f"No factory handler for key {key!r}.")


# ============================================================================
# Private builders
# ============================================================================

def _create_gdino(
    key: str,
    label_config: Any,
    confidence_threshold: Optional[float],
    image_size: Optional[Tuple[int, int]] = None,
) -> Tuple[DetectionModel, ClassMapper]:
    from .grounding_dino import GroundingDinoDetector

    model_id = _GDINO_HF_IDS[key]
    kwargs: Dict[str, Any] = {"model_id": model_id}
    if confidence_threshold is not None:
        kwargs["confidence_threshold"] = confidence_threshold

    # Use image dimensions to set processing resolution so the
    # model sees the full native image instead of downscaling to 1333px.
    if image_size is not None:
        longest_edge = max(image_size)
        # Only override if the image is actually larger than the default
        if longest_edge > 1333:
            kwargs["max_image_size"] = longest_edge
            log_info(
                f"GDINO: image size {image_size[0]}×{image_size[1]} → "
                f"max_image_size={longest_edge}"
            )

    detector = GroundingDinoDetector(**kwargs)
    detector.configure_from_schema(label_config)

    mapper = GroundingDinoDetector.get_identity_mapper(
        list(detector.supported_class_ids)
    )
    log_info(f"Created Grounding DINO detector: {model_id}")
    return detector, mapper


def _create_yolo(
    model_path: str,
    confidence_threshold: Optional[float],
) -> Tuple[DetectionModel, ClassMapper]:
    """Build any ultralytics-based detector (YOLO26, YOLOv8, custom, …)."""
    from .yolov8 import YoloV8Detector

    kwargs: Dict[str, Any] = {"model_path": model_path}
    if confidence_threshold is not None:
        kwargs["confidence_threshold"] = confidence_threshold

    detector = YoloV8Detector(**kwargs)
    mapper = get_default_coco_mapper()
    log_info(f"Created YOLO detector: {model_path}")
    return detector, mapper


def _create_ensemble(
    key: str,
    label_config: Any,
    file_path: Optional[str],
    confidence_threshold: Optional[float],
    image_size: Optional[Tuple[int, int]] = None,
) -> Tuple[DetectionModel, ClassMapper]:
    """Build an ensemble from a recipe of sub-model keys."""
    from .ensemble_detector import EnsembleDetector

    sub_keys = _ENSEMBLE_RECIPES[key]
    pairs: List[Tuple[DetectionModel, ClassMapper]] = []
    for sk in sub_keys:
        det, mapper = create_detector(
            sk,
            label_config=label_config,
            file_path=file_path,
            confidence_threshold=confidence_threshold,
            image_size=image_size,
        )
        pairs.append((det, mapper))

    ensemble = EnsembleDetector(models=pairs)
    # Identity mapper — the ensemble already outputs schema IDs.
    from .base import ClassMapper as _CM
    all_ids = ensemble.supported_class_ids
    identity_mapper = _CM(model_to_schema={i: i for i in all_ids})

    log_info(f"Created ensemble detector: {key} ({len(pairs)} sub-models)")
    return ensemble, identity_mapper
