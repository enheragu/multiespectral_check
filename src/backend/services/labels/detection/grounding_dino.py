"""Grounding DINO zero-shot detector via HuggingFace Transformers.

Open-vocabulary object detection using text prompts.  Runs locally
on GPU (or CPU) – no cloud API required.

Requires:
    pip install transformers torch

Supported models (HuggingFace hub):
    - IDEA-Research/grounding-dino-base   (larger, higher accuracy)
    - IDEA-Research/grounding-dino-tiny   (smaller, faster)

Usage:
    detector = GroundingDinoDetector()
    detector.set_class_prompts({1: "person", 2: "car", 3: "bus . truck"})
    detector.load()
    results = detector.detect(image)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from common.log_utils import log_debug, log_info, log_warning

from .base import (
    ClassMapper,
    DetectionModel,
    DetectionResult,
    ModelInfo,
)

# ---------------------------------------------------------------------------
# Lazy-import helpers (avoid heavy torch/transformers import at startup)
# ---------------------------------------------------------------------------

_lazy_cache: Dict[str, Any] = {}


def _get_transformers():
    """Lazy import of transformers classes."""
    if "AutoProcessor" not in _lazy_cache:
        try:
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )
            _lazy_cache["AutoProcessor"] = AutoProcessor
            _lazy_cache["AutoModel"] = AutoModelForZeroShotObjectDetection
        except ImportError as exc:
            raise ImportError(
                "transformers not installed. Install with: "
                "pip install transformers torch"
            ) from exc
    return _lazy_cache["AutoProcessor"], _lazy_cache["AutoModel"]


def _get_torch():
    """Lazy import of torch."""
    if "torch" not in _lazy_cache:
        try:
            import torch
            _lazy_cache["torch"] = torch
        except ImportError as exc:
            raise ImportError(
                "torch not installed. Install with: pip install torch"
            ) from exc
    return _lazy_cache["torch"]


def _get_pil_image():
    """Lazy import of PIL.Image."""
    if "PILImage" not in _lazy_cache:
        try:
            from PIL import Image as PILImage
            _lazy_cache["PILImage"] = PILImage
        except ImportError as exc:
            raise ImportError(
                "Pillow not installed. Install with: pip install Pillow"
            ) from exc
    return _lazy_cache["PILImage"]


# ---------------------------------------------------------------------------
# Geometry helpers (module-level, used by cross-chunk NMS)
# ---------------------------------------------------------------------------

def _bbox_to_xyxy_norm(
    bbox: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """Convert normalised centre ``(cx, cy, w, h)`` to ``(x1, y1, x2, y2)``."""
    cx, cy, w, h = bbox
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _iou_xyxy(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """IoU between two ``(x1, y1, x2, y2)`` boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class GroundingDinoDetector(DetectionModel):
    """Grounding DINO zero-shot object detection model.

    Uses text prompts constructed from label schema class names to detect
    objects in a fully open-vocabulary manner.  Runs on local GPU/CPU via
    HuggingFace Transformers.

    Workflow
    --------
    1. Instantiate with model ID and thresholds.
    2. Call ``set_class_prompts()`` to define which classes to detect and the
       text phrases that describe them.
    3. Call ``load()`` (or let the first ``detect()`` auto-load).
    4. Call ``detect(image)`` to get ``DetectionResult`` objects whose
       ``class_id`` values are **schema** class IDs (not COCO IDs).

    Because the text prompts are derived from the label schema, the returned
    ``DetectionResult.class_id`` is already the schema class ID.  Use an
    identity ``ClassMapper`` so that ``detections_to_annotations`` passes
    the IDs through unchanged.

    Attributes
    ----------
    model_id : str
        HuggingFace model identifier.
    confidence_threshold : float
        Minimum box confidence score (``box_threshold`` in GDINO parlance).
    text_threshold : float
        Minimum text-matching score.  Lower values are more permissive.
    device : str | None
        ``"cuda"``, ``"cpu"`` or ``None`` (auto-detect).
    max_image_size : int | None
        Override the processor's default longest-edge resize limit
        (default 1333 px).  Higher values improve small-object recall
        at the cost of VRAM and latency.  ``None`` keeps the default.
    """

    # Known HuggingFace model IDs ↔ short display names
    KNOWN_MODELS: Dict[str, str] = {
        "IDEA-Research/grounding-dino-tiny": "gdino-tiny",
        "IDEA-Research/grounding-dino-base": "gdino-base",
    }

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        confidence_threshold: float = 0.30,
        text_threshold: float = 0.20,
        device: Optional[str] = None,
        max_image_size: Optional[int] = None,
    ) -> None:
        self.model_id = model_id
        self.confidence_threshold = confidence_threshold
        self.text_threshold = text_threshold
        self.device = device
        self.max_image_size = max_image_size

        # class_id (schema) → prompt text  (set via set_class_prompts)
        self._class_prompts: Dict[int, str] = {}
        # Reverse: lowered prompt text → schema class_id
        self._prompt_to_class_id: Dict[str, int] = {}
        # Reverse: lowered prompt text → inferred attribute values.
        # Example: "sedan" → {"type": "sedan"}  (we know the YAML attr).
        self._prompt_to_attrs: Dict[str, Dict[str, str]] = {}

        # Model state (populated by load())
        self._model: Any = None
        self._processor: Any = None
        self._model_info: Optional[ModelInfo] = None
        self._resolved_device: Optional[str] = None

    # ------------------------------------------------------------------
    # Class prompt management
    # ------------------------------------------------------------------

    def set_class_prompts(self, prompts: Dict[int, str]) -> None:
        """Define which classes to detect and the text that describes them.

        Args:
            prompts: ``{schema_class_id: text_phrase}``.
                Example::

                    {1: "person", 2: "car", 3: "bus . truck"}

                Each *value* is passed verbatim to Grounding DINO as part of
                the full text prompt.  Values **must not** contain a trailing
                period – the detector joins them with ``" . "``.
        """
        self._class_prompts = dict(prompts)
        self._build_prompt_index()
        log_debug(
            f"GDINO prompts set: {len(prompts)} classes – "
            f"{self._full_prompt_text()!r}",
            "DETECTION",
        )

    def configure_from_schema(self, config: Any) -> None:
        """One-call setup: build prompts + attribute inference table.

        Call this instead of ``build_prompts_from_config`` +
        ``set_class_prompts`` separately.  In addition to prompts it
        builds ``_prompt_to_attrs`` which maps each sub-phrase to the
        schema attribute it originated from, enabling automatic
        attribute inference at detection time.

        Example:  GDINO returns ``"sedan"`` → we know class=car **and**
        ``type=sedan`` because "sedan" came from ``car.attributes.type``.
        """
        prompts = self.build_prompts_from_config(config)
        self.set_class_prompts(prompts)
        self._build_attr_inference_table(config)

    def _build_attr_inference_table(self, config: Any) -> None:
        """Populate ``_prompt_to_attrs`` from the label schema.

        For each sub-type attribute option that made it into the prompt,
        record which attribute name it belongs to.  This lets us
        auto-fill attribute values on detected objects.
        """
        self._prompt_to_attrs = {}

        for class_def in config.classes.values():
            for attr_name, attr in (class_def.attributes or {}).items():
                if attr_name not in self._SUBTYPE_ATTRS:
                    continue
                if not (hasattr(attr, "options") and attr.options):
                    continue
                for opt in attr.options:
                    if not isinstance(opt, str):
                        continue
                    key = opt.replace("_", " ").strip().lower()
                    # Only record if this sub-phrase is actually in our
                    # prompt index AND belongs to the same class (it may
                    # have been registered under a different class due to
                    # aliases).
                    if key in self._prompt_to_class_id and \
                       self._prompt_to_class_id[key] == class_def.id:
                        self._prompt_to_attrs[key] = {attr_name: opt}

            # Aliases that also double as attribute values (e.g. "bus"
            # is both an alias of heavy_vehicle AND class=bus).
            for alias in (class_def.aliases or []):
                key = alias.replace("_", " ").strip().lower()
                if key not in self._prompt_to_attrs:
                    # Check if the alias matches any sub-type option
                    for attr_name, attr in (class_def.attributes or {}).items():
                        if attr_name not in self._SUBTYPE_ATTRS:
                            continue
                        if not (hasattr(attr, "options") and attr.options):
                            continue
                        opts_lower = {
                            str(o).replace("_", " ").lower(): o
                            for o in attr.options
                            if isinstance(o, str)
                        }
                        if key in opts_lower:
                            self._prompt_to_attrs[key] = {
                                attr_name: opts_lower[key]
                            }
                            break

        # Register visual synonyms that aren't YAML options but carry
        # inferred attributes (e.g. "woman" → perceived_gender=female).
        for class_id, synonyms in self._VISUAL_SYNONYMS.items():
            for term, attrs in synonyms:
                key = term.strip().lower()
                if key in self._prompt_to_class_id and \
                   self._prompt_to_class_id[key] == class_id:
                    if key not in self._prompt_to_attrs:
                        self._prompt_to_attrs[key] = dict(attrs)

        log_debug(
            f"Attribute inference table: {len(self._prompt_to_attrs)} entries",
            "DETECTION",
        )

    def _infer_attributes(self, label_text: str) -> Dict[str, str]:
        """Infer schema attribute values from a GDINO raw label.

        Returns a dict of ``{attr_name: attr_value}`` that can be merged
        into the ``DetectionResult.attributes``.  Returns empty dict if
        no inference is possible.
        """
        key = label_text.strip().lower()
        # Exact match
        if key in self._prompt_to_attrs:
            return dict(self._prompt_to_attrs[key])
        # Containment match (GDINO sometimes returns partial labels)
        for prompt_key, attrs in self._prompt_to_attrs.items():
            if key in prompt_key or prompt_key in key:
                return dict(attrs)
        return {}

    def _build_prompt_index(self) -> None:
        """Rebuild the reverse text→class_id lookup."""
        self._prompt_to_class_id = {}
        for class_id, text in self._class_prompts.items():
            # A prompt may contain sub-phrases separated by " . " when a
            # single schema class maps to several GDINO categories (e.g.
            # heavy_vehicle → "bus . truck").  We index each sub-phrase.
            for part in text.split("."):
                key = part.strip().lower()
                if key:
                    self._prompt_to_class_id[key] = class_id

    def _full_prompt_text(self) -> str:
        """Build the full text prompt sent to GDINO.

        All sub-phrases from every class are joined with ``" . "``
        and terminated with a trailing ``" ."``.
        """
        parts: List[str] = []
        for text in self._class_prompts.values():
            for sub in text.split("."):
                sub = sub.strip()
                if sub:
                    parts.append(sub)
        if not parts:
            return ""
        return " . ".join(parts) + " ."

    # Maximum tokens the GDINO text encoder accepts.  The original model
    # was trained with a BERT tokeniser capped at 256 sub-word tokens
    # (including [CLS] and [SEP]).  We leave a small margin.
    _MAX_PROMPT_TOKENS: int = 245

    def _build_prompt_chunks(self) -> List[str]:
        """Split per-class prompts into chunks that fit the token limit.

        Each chunk is a self-contained ``" . "``-joined prompt string.
        The tokeniser is used to count actual sub-word tokens, falling
        back to a character-based heuristic if the model is not loaded yet.
        """
        all_subs: List[str] = []
        for text in self._class_prompts.values():
            for sub in text.split("."):
                sub = sub.strip()
                if sub:
                    all_subs.append(sub)

        if not all_subs:
            return []

        # Build the single full prompt; if it fits, return as-is.
        full = " . ".join(all_subs) + " ."
        if self._estimate_tokens(full) <= self._MAX_PROMPT_TOKENS:
            return [full]

        # Greedy bin-packing into chunks
        chunks: List[str] = []
        current_subs: List[str] = []
        for sub in all_subs:
            candidate = " . ".join(current_subs + [sub]) + " ."
            if current_subs and self._estimate_tokens(candidate) > self._MAX_PROMPT_TOKENS:
                # Flush current chunk
                chunks.append(" . ".join(current_subs) + " .")
                current_subs = [sub]
            else:
                current_subs.append(sub)
        if current_subs:
            chunks.append(" . ".join(current_subs) + " .")

        log_debug(
            f"GDINO prompt split into {len(chunks)} chunks "
            f"(~{[self._estimate_tokens(c) for c in chunks]} tokens each)",
            "DETECTION",
        )
        return chunks

    def _estimate_tokens(self, text: str) -> int:
        """Return the number of tokens *text* would produce.

        Uses the loaded tokeniser if available, otherwise falls back to a
        conservative heuristic (~4.5 chars per token for English text).
        """
        if self._processor is not None:
            try:
                enc = self._processor.tokenizer(
                    text, add_special_tokens=True, truncation=False,
                    verbose=False,
                )
                return len(enc["input_ids"])
            except Exception:
                pass
        # Fallback: 1 token every ~4 chars (conservative)
        return len(text) // 4 + 2  # +2 for [CLS]/[SEP]

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Download (if needed) and load the model into GPU/CPU memory."""
        if self._model is not None:
            return  # Already loaded

        torch = _get_torch()
        AutoProcessor, AutoModel = _get_transformers()

        # Resolve device
        if self.device:
            device = self.device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._resolved_device = device

        log_info(
            f"Loading Grounding DINO: {self.model_id} on {device}", "DETECTION"
        )

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id).to(device)

        short_name = self.KNOWN_MODELS.get(self.model_id, self.model_id)
        self._model_info = ModelInfo(
            name=short_name,
            version=self._get_transformers_version(),
            source="huggingface-transformers",
            confidence_threshold=self.confidence_threshold,
            extra={
                "model_id": self.model_id,
                "device": device,
                "text_threshold": self.text_threshold,
                "max_image_size": self.max_image_size or "default (1333)",
            },
        )
        log_info(f"Grounding DINO loaded: {short_name} on {device}", "DETECTION")

    def unload(self) -> None:
        """Release model from memory."""
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            self._model_info = None
            self._resolved_device = None

            torch = _get_torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            log_debug("Grounding DINO model unloaded", "DETECTION")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> List[DetectionResult]:
        """Run open-vocabulary detection on a single image.

        Args:
            image: ``(H, W, C)`` numpy array in **BGR** format (OpenCV
                   convention used throughout the app).
            confidence_threshold: Override ``self.confidence_threshold``.

        Returns:
            List of ``DetectionResult`` whose ``class_id`` is the *schema*
            class ID (not a COCO ID).
        """
        if not self._class_prompts:
            log_warning(
                "GDINO detect() called with no class prompts – returning []",
                "DETECTION",
            )
            return []

        # Auto-load if needed
        if not self.is_loaded:
            self.load()

        torch = _get_torch()
        PILImage = _get_pil_image()

        box_thr = confidence_threshold or self.confidence_threshold
        chunks = self._build_prompt_chunks()

        if not chunks:
            return []

        h, w = image.shape[:2]

        # Convert BGR numpy → RGB PIL (required by the processor)
        image_rgb = image[:, :, ::-1] if image.ndim == 3 and image.shape[2] >= 3 else image
        pil_image = PILImage.fromarray(np.ascontiguousarray(image_rgb))

        detections: List[DetectionResult] = []

        # Build optional size override for higher-resolution processing
        proc_kwargs: Dict[str, Any] = {}
        if self.max_image_size is not None:
            # Scale shortest_edge proportionally to longest_edge
            default_short, default_long = 800, 1333
            ratio = self.max_image_size / default_long
            proc_kwargs["size"] = {
                "shortest_edge": int(default_short * ratio),
                "longest_edge": self.max_image_size,
            }

        for chunk_prompt in chunks:
            # Tokenise + encode
            inputs = self._processor(
                images=pil_image, text=chunk_prompt, return_tensors="pt",
                **proc_kwargs,
            )
            inputs = {k: v.to(self._resolved_device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self._model(**inputs)

            # Post-process → list of dicts with "scores", "labels", "boxes"
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs["input_ids"],
                threshold=box_thr,
                text_threshold=self.text_threshold,
                target_sizes=[(h, w)],
            )

            if not results:
                continue

            result = results[0]  # single image
            boxes = result["boxes"]       # tensor (N, 4)  xyxy absolute
            scores = result["scores"]     # tensor (N,)
            # transformers ≥4.51 renamed "labels" → "text_labels"
            labels = result.get("text_labels") or result.get("labels", [])

            for i in range(len(scores)):
                label_text = labels[i] if i < len(labels) else ""
                score = float(scores[i].cpu())
                xyxy = boxes[i].cpu().numpy()

                class_id = self._match_label_to_class_id(label_text)
                if class_id is None:
                    log_debug(
                        f"GDINO label {label_text!r} has no schema mapping – skipped",
                        "DETECTION",
                    )
                    continue

                # xyxy absolute → normalised center (cx, cy, w, h)
                x1, y1, x2, y2 = xyxy
                cx = float((x1 + x2) / 2) / w
                cy = float((y1 + y2) / 2) / h
                bw = float(x2 - x1) / w
                bh = float(y2 - y1) / h

                class_name = self._class_prompts.get(class_id, label_text)

                # Build attributes: raw label + inferred schema attrs
                det_attrs: Dict[str, Any] = {"raw_label": label_text}
                det_attrs.update(self._infer_attributes(label_text))

                detections.append(DetectionResult(
                    class_id=class_id,
                    class_name=class_name,
                    bbox=(cx, cy, bw, bh),
                    confidence=score,
                    attributes=det_attrs,
                ))

        # ── Cross-chunk NMS ──
        # When prompts are split across chunks the same physical object
        # may be detected in multiple chunks (e.g. "person" in chunk 1
        # and "man" in chunk 2, both mapping to class_id=1).  We suppress
        # same-class duplicates keeping the higher-confidence box.
        if len(chunks) > 1:
            before = len(detections)
            detections = self._cross_chunk_nms(detections)
            suppressed = before - len(detections)
            if suppressed:
                log_debug(
                    f"GDINO cross-chunk NMS removed {suppressed} duplicate(s), "
                    f"{len(detections)} remain",
                    "DETECTION",
                )

        log_debug(
            f"GDINO detected {len(detections)} objects "
            f"across {len(chunks)} prompt chunk(s)",
            "DETECTION",
        )
        return detections

    # ------------------------------------------------------------------
    # Cross-chunk NMS
    # ------------------------------------------------------------------

    @staticmethod
    def _cross_chunk_nms(
        detections: List[DetectionResult],
        iou_threshold: float = 0.70,
    ) -> List[DetectionResult]:
        """Suppress same-class duplicates from multi-chunk inference.

        When the prompt is split into chunks, the same physical object
        can appear in multiple chunks with the same (or very similar)
        bounding box.  This greedy NMS keeps the highest-confidence
        detection and discards lower-scored duplicates **only when they
        share the same class_id** and overlap above *iou_threshold*.

        A high IoU threshold (0.70) is used intentionally: two *distinct*
        instances of the same class standing close together rarely exceed
        0.70 IoU, while a true duplicate from a different prompt chunk
        will have IoU > 0.90.
        """
        if not detections:
            return detections

        # Sort by confidence descending — keep the best first
        ranked = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep: List[DetectionResult] = []
        suppressed = [False] * len(ranked)

        for i, det_i in enumerate(ranked):
            if suppressed[i]:
                continue
            keep.append(det_i)
            box_i = _bbox_to_xyxy_norm(det_i.bbox)
            for j in range(i + 1, len(ranked)):
                if suppressed[j]:
                    continue
                det_j = ranked[j]
                # Only suppress same-class boxes
                if det_j.class_id != det_i.class_id:
                    continue
                box_j = _bbox_to_xyxy_norm(det_j.bbox)
                if _iou_xyxy(box_i, box_j) >= iou_threshold:
                    suppressed[j] = True

        return keep

    # ------------------------------------------------------------------
    # Label → class-ID matching
    # ------------------------------------------------------------------

    def _match_label_to_class_id(self, label: str) -> Optional[int]:
        """Map a GDINO output label string to a schema class ID.

        Matching strategy (first wins):
        1. Exact match against indexed prompt sub-phrases.
        2. Containment match – the label contains a known prompt, or vice
           versa (handles partial matches returned by GDINO).
        """
        key = label.strip().lower()

        # 1. Exact
        if key in self._prompt_to_class_id:
            return self._prompt_to_class_id[key]

        # 2. Containment (short labels inside prompts or vice-versa)
        for prompt_key, cid in self._prompt_to_class_id.items():
            if key in prompt_key or prompt_key in key:
                return cid

        return None

    # ------------------------------------------------------------------
    # ABC required properties / helpers
    # ------------------------------------------------------------------

    def get_model_info(self) -> ModelInfo:
        if self._model_info is not None:
            return self._model_info
        short_name = self.KNOWN_MODELS.get(self.model_id, self.model_id)
        return ModelInfo(
            name=short_name,
            version="unknown",
            source="huggingface-transformers",
            confidence_threshold=self.confidence_threshold,
        )

    @property
    def supported_class_ids(self) -> List[int]:
        """Return the schema class IDs that have prompts configured."""
        return list(self._class_prompts.keys())

    @property
    def class_names(self) -> Dict[int, str]:
        """Return schema class ID → prompt text."""
        return dict(self._class_prompts)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def get_identity_mapper(class_ids: List[int]) -> ClassMapper:
        """Build a ClassMapper where model IDs == schema IDs (pass-through).

        Use this when calling ``LabelService.set_detector()`` so that
        ``detections_to_annotations`` preserves the schema IDs returned
        by this detector.
        """
        return ClassMapper(model_to_schema={cid: cid for cid in class_ids})

    # Attribute names whose enum options describe *visual sub-types* that
    # Grounding DINO can recognise.  Everything else (state, material,
    # condition, location, activity, gender…) is post-detection metadata
    # and would only waste precious token budget.
    _SUBTYPE_ATTRS: frozenset[str] = frozenset(
        {"type", "class", "propulsion", "service", "category",
         "role", "age_group"}
    )

    # Max enum options to include per sub-type attribute.  Generous limit
    # so that classes with many visual sub-types (street_furniture has 9)
    # get full coverage in the prompt.
    _MAX_SUBTYPE_OPTIONS: int = 15

    # When a class has at least this many specific terms (aliases +
    # sub-type options + visual synonyms), the generic class name is
    # dropped from the prompt.  This forces GDINO to return a specific
    # sub-type label (e.g. "bench" instead of "street furniture") so
    # that _infer_attributes() can resolve the attribute automatically.
    _MIN_SPECIFIC_TO_DROP_GENERIC: int = 3

    # Extra visual synonyms that help detection but don't appear in the
    # YAML schema.  These are appended to the class prompt and their
    # inferred attributes are registered for reverse mapping.
    # Format: {schema_class_id: [(term, {attr: val}), ...]}
    _VISUAL_SYNONYMS: Dict[int, List[Tuple[str, Dict[str, str]]]] = {
        1: [  # person
            ("man",   {"perceived_gender": "male",   "age_group": "adult"}),
            ("woman", {"perceived_gender": "female", "age_group": "adult"}),
            ("boy",   {"perceived_gender": "male",   "age_group": "child"}),
            ("girl",  {"perceived_gender": "female", "age_group": "child"}),
        ],
        14: [  # vegetation
            ("tree",         {"type": "tree_canopy"}),
            ("palm tree",    {"type": "tree_canopy"}),
            ("shrub",        {"type": "bush"}),
            ("potted plant", {"type": "potted_plant"}),
        ],
    }

    @staticmethod
    def build_prompts_from_config(config: Any) -> Dict[int, str]:
        """Build a ``{class_id: prompt_text}`` dict from a ``LabelConfig``.

        The prompt is optimised for Grounding DINO's 256-token text limit
        and for **attribute inference**: we want the model to return a
        *specific* sub-type term (``bench``, ``trash bag``) rather than
        an abstract class name (``street furniture``, ``loose object``).

        Strategy:

        * **Aliases** → each becomes a sub-phrase (``bus . truck``).
        * **Sub-type attribute options** → from attributes that name
          *what the object is* (``type``, ``class``, ``propulsion``,
          ``service``, ``category``), e.g. ``sedan . suv . van``.
        * **Visual synonyms** → extra detection terms (``man``, ``woman``).
        * **Generic class name** → included ONLY when there aren't
          enough specific terms (< ``_MIN_SPECIFIC_TO_DROP_GENERIC``).
          Dropping it forces GDINO to return specific sub-types, which
          map cleanly to schema attributes via ``_infer_attributes()``.
        * Options that duplicate another class's name/alias are skipped
          to avoid cross-class detection confusion.

        Sub-phrases are deduplicated and joined with ``" . "``.
        """
        # Build a global set of all class names + aliases (lowercased)
        # so that sub-type options don't collide across classes.
        _global_names: dict[str, int] = {}  # lowered term → owning class_id
        for class_def in config.classes.values():
            _global_names[class_def.name.replace("_", " ").lower()] = class_def.id
            for alias in (class_def.aliases or []):
                _global_names[alias.replace("_", " ").lower()] = class_def.id

        # Terms that are never useful for open-vocabulary detection
        _skip = {"unknown", "none", "other"}

        prompts: Dict[int, str] = {}
        for class_def in config.classes.values():
            parts: List[str] = []

            # 1. Primary name (human-readable)
            parts.append(class_def.name.replace("_", " "))

            # 2. Aliases ("bus", "truck", …)
            for alias in (class_def.aliases or []):
                parts.append(alias.replace("_", " "))

            # 3. Only *sub-type* attribute options, with collision guard
            for attr_name, attr in (class_def.attributes or {}).items():
                if attr_name not in GroundingDinoDetector._SUBTYPE_ATTRS:
                    continue
                if not (hasattr(attr, "options") and attr.options):
                    continue
                added = 0
                for opt in attr.options:
                    if added >= GroundingDinoDetector._MAX_SUBTYPE_OPTIONS:
                        break
                    if not isinstance(opt, str):
                        continue
                    clean = opt.replace("_", " ")
                    low = clean.lower()
                    if low in _skip:
                        continue
                    # Skip if this term is the name/alias of a *different* class
                    owner = _global_names.get(low)
                    if owner is not None and owner != class_def.id:
                        continue
                    parts.append(clean)
                    added += 1

            # Append visual synonyms for this class (detection-boosting
            # terms not in the YAML, e.g. "man", "woman" for person).
            for term, _attrs in GroundingDinoDetector._VISUAL_SYNONYMS.get(
                class_def.id, [],
            ):
                low = term.lower()
                owner = _global_names.get(low)
                if owner is not None and owner != class_def.id:
                    continue  # owned by another class – skip
                parts.append(term)

            # Deduplicate while preserving order, lowercase comparison
            seen: set[str] = set()
            unique: List[str] = []
            for p in parts:
                key = p.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    unique.append(p.strip())

            # Drop the generic class name when enough specific terms
            # exist.  This forces GDINO to return "bench" instead of
            # "street furniture", enabling attribute inference.
            generic = class_def.name.replace("_", " ").strip()
            specific = [u for u in unique if u.lower() != generic.lower()]
            if len(specific) >= GroundingDinoDetector._MIN_SPECIFIC_TO_DROP_GENERIC:
                unique = specific

            prompts[class_def.id] = " . ".join(unique)
        return prompts

    @staticmethod
    def _get_transformers_version() -> str:
        try:
            import transformers
            return transformers.__version__
        except (ImportError, AttributeError):
            return "unknown"
