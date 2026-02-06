"""Label Service - Orchestrates label operations for multiespectral datasets.

This is the main entry point for labelling functionality, coordinating:
- LabelStorage: Persistence layer
- DetectionModel: Auto-detection
- BBox transforms: Cross-channel projection
- Source workflow: AUTO → REVIEWED → MANUAL

Follows DESIGN_PHILOSOPHY.md:
- Single source of truth (LabelStorage._cache)
- Pure backend (no Qt dependencies)
- Pluggable detection models
- Lazy loading of heavy dependencies

Usage:
    service = LabelService(dataset_path)
    service.load_config("config/labels_multiespectral_dataset.yaml")

    # Auto-detect on visible channel
    annotations = service.auto_detect("image_001", "visible", image)

    # Project to LWIR
    lwir_annotations = service.project_to_channel("image_001", "visible", "lwir")

    # User reviews
    service.mark_reviewed("image_001", "visible", annotation_indices=[0, 1])

    # Save
    service.save_all()
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from common.log_utils import log_debug, log_info, log_warning

from .label_types import (
    Annotation,
    AnnotationSource,
    ImageLabels,
    LabelConfig,
)
from .label_storage import LabelStorage
from .detection import (
    ClassMapper,
    DetectionModel,
    DetectionResult,
    detections_to_annotations,
    get_default_coco_mapper,
)
from .bbox_transform import (
    project_annotations_to_other_channel,
    project_bbox_to_other_channel,
)

if TYPE_CHECKING:
    from backend.services.calibration.calibration_controller import CalibrationController


# =============================================================================
# Label Service
# =============================================================================

class LabelService:
    """Orchestrates all label operations for a dataset.

    Responsibilities:
    - Manage label configuration (schema)
    - Coordinate storage operations
    - Run auto-detection with pluggable models
    - Project labels between channels
    - Handle source workflow (AUTO → REVIEWED → MANUAL)

    Thread Safety:
    - NOT thread-safe. Use from main thread or with external locking.
    - Detection can be run in background thread, results merged on main thread.
    """

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        """Initialize label service.

        Args:
            dataset_path: Path to dataset root (can be set later via set_dataset)
            config_path: Path to label config YAML (can be set later via load_config)
        """
        self._dataset_path: Optional[Path] = None
        self._config: Optional[LabelConfig] = None
        self._storage: Optional[LabelStorage] = None

        # Detection model (lazy loaded)
        self._detector: Optional[DetectionModel] = None
        self._class_mapper: Optional[ClassMapper] = None

        # Calibration data for projection (injected)
        self._homography_visible_to_lwir: Optional[np.ndarray] = None
        self._homography_lwir_to_visible: Optional[np.ndarray] = None
        self._visible_size: Optional[Tuple[int, int]] = None
        self._lwir_size: Optional[Tuple[int, int]] = None

        # Initialize if paths provided
        if dataset_path:
            self.set_dataset(dataset_path)
        if config_path:
            self.load_config(config_path)

    # =========================================================================
    # Configuration
    # =========================================================================

    def load_config(self, config_path: Path) -> None:
        """Load label configuration from YAML file."""
        self._config = LabelConfig.from_yaml(config_path)
        log_info(f"Loaded label config: {len(self._config.classes)} classes")

        # Re-initialize storage with new config if dataset is set
        if self._dataset_path and self._config:
            self._init_storage()

    def set_dataset(self, dataset_path: Path) -> None:
        """Set the active dataset path."""
        self._dataset_path = Path(dataset_path)

        # Initialize storage if config is loaded
        if self._config:
            self._init_storage()

    def _init_storage(self) -> None:
        """Initialize label storage for current dataset and config."""
        if not self._dataset_path or not self._config:
            return

        self._storage = LabelStorage(dataset_path=self._dataset_path)
        self._storage.set_config(self._config)
        log_debug(f"Label storage initialized for {self._dataset_path}")

    @property
    def config(self) -> Optional[LabelConfig]:
        """Get current label configuration."""
        return self._config

    @property
    def is_ready(self) -> bool:
        """Check if service is ready for operations."""
        return self._storage is not None and self._config is not None

    # =========================================================================
    # Detection Model
    # =========================================================================

    def set_detector(
        self,
        detector: DetectionModel,
        class_mapper: Optional[ClassMapper] = None,
    ) -> None:
        """Set the detection model to use for auto-detection.

        Args:
            detector: DetectionModel instance
            class_mapper: Optional mapper from model classes to schema classes.
                         If None, uses default COCO mapper.
        """
        self._detector = detector
        self._class_mapper = class_mapper or get_default_coco_mapper()
        log_info(f"Detector set: {detector.get_model_info().name}")

    def load_yolov8_detector(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        class_mapper: Optional[ClassMapper] = None,
    ) -> None:
        """Convenience method to load YOLOv8 detector.

        Args:
            model_path: Path to model weights or model name
            confidence_threshold: Detection confidence threshold
            class_mapper: Optional class mapper (defaults to COCO mapper)
        """
        from .detection.yolov8 import YoloV8Detector

        detector = YoloV8Detector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
        )
        self.set_detector(detector, class_mapper)

    @property
    def has_detector(self) -> bool:
        """Check if a detection model is configured."""
        return self._detector is not None

    # =========================================================================
    # Calibration / Projection Setup
    # =========================================================================

    def set_projection_params(
        self,
        homography_visible_to_lwir: np.ndarray,
        homography_lwir_to_visible: np.ndarray,
        visible_size: Tuple[int, int],
        lwir_size: Tuple[int, int],
    ) -> None:
        """Set homographies for cross-channel bbox projection.

        Args:
            homography_visible_to_lwir: 3x3 homography matrix
            homography_lwir_to_visible: 3x3 homography matrix (inverse)
            visible_size: (width, height) of visible images
            lwir_size: (width, height) of LWIR images
        """
        self._homography_visible_to_lwir = homography_visible_to_lwir
        self._homography_lwir_to_visible = homography_lwir_to_visible
        self._visible_size = visible_size
        self._lwir_size = lwir_size
        log_debug("Projection parameters set")

    def load_projection_from_calibration(
        self,
        calibration_controller: "CalibrationController",
    ) -> bool:
        """Load projection parameters from calibration controller.

        Args:
            calibration_controller: Controller with computed calibration

        Returns:
            True if projection params were loaded successfully
        """
        try:
            # Get homography from calibration
            # This depends on how calibration stores the data
            calib_data = calibration_controller.get_calibration_data()
            if not calib_data:
                log_warning("No calibration data available for projection")
                return False

            # Extract homography - this may need adjustment based on actual structure
            H_v2l = calib_data.get("homography_visible_to_lwir")
            H_l2v = calib_data.get("homography_lwir_to_visible")

            if H_v2l is None or H_l2v is None:
                log_warning("Calibration missing homography matrices")
                return False

            visible_size = calib_data.get("visible_image_size")
            lwir_size = calib_data.get("lwir_image_size")

            if not visible_size or not lwir_size:
                log_warning("Calibration missing image sizes")
                return False

            self.set_projection_params(
                np.array(H_v2l),
                np.array(H_l2v),
                tuple(visible_size),
                tuple(lwir_size),
            )
            return True

        except Exception as e:
            log_warning(f"Failed to load projection from calibration: {e}")
            return False

    @property
    def can_project(self) -> bool:
        """Check if cross-channel projection is available."""
        return (
            self._homography_visible_to_lwir is not None and
            self._homography_lwir_to_visible is not None and
            self._visible_size is not None and
            self._lwir_size is not None
        )

    # =========================================================================
    # Label Operations - Read
    # =========================================================================

    def get_labels(
        self,
        image_base: str,
        channel: str,
    ) -> Optional[ImageLabels]:
        """Get labels for an image.

        Args:
            image_base: Image base name (without extension)
            channel: Channel name ('visible' or 'lwir')

        Returns:
            ImageLabels or None if storage not initialized
        """
        if not self._storage:
            return None
        return self._storage.load_labels(channel, image_base)

    def get_annotations(
        self,
        image_base: str,
        channel: str,
    ) -> List[Annotation]:
        """Get annotations for an image.

        Args:
            image_base: Image base name
            channel: Channel name

        Returns:
            List of annotations (empty if none)
        """
        labels = self.get_labels(image_base, channel)
        return labels.annotations if labels else []

    def get_annotations_by_source(
        self,
        image_base: str,
        channel: str,
        source: AnnotationSource,
    ) -> List[Annotation]:
        """Get annotations filtered by source.

        Args:
            image_base: Image base name
            channel: Channel name
            source: Filter by this source (AUTO, REVIEWED, MANUAL)

        Returns:
            List of matching annotations
        """
        labels = self.get_labels(image_base, channel)
        if not labels:
            return []
        return [a for a in labels.annotations if a.source == source]

    def count_annotations(
        self,
        image_base: str,
        channel: str,
    ) -> Dict[str, int]:
        """Count annotations by source.

        Returns:
            Dict with keys 'auto', 'reviewed', 'manual', 'total'
        """
        labels = self.get_labels(image_base, channel)
        if not labels:
            return {"auto": 0, "reviewed": 0, "manual": 0, "total": 0}

        counts = labels.count_by_source()
        return {
            "auto": counts[AnnotationSource.AUTO],
            "reviewed": counts[AnnotationSource.REVIEWED],
            "manual": counts[AnnotationSource.MANUAL],
            "total": len(labels.annotations),
        }

    def has_labels(self, image_base: str, channel: str) -> bool:
        """Check if image has any labels."""
        labels = self.get_labels(image_base, channel)
        return labels is not None and len(labels.annotations) > 0

    def has_unreviewed(self, image_base: str, channel: str) -> bool:
        """Check if image has unreviewed (AUTO) annotations."""
        labels = self.get_labels(image_base, channel)
        if not labels:
            return False
        return any(a.source == AnnotationSource.AUTO for a in labels.annotations)

    # =========================================================================
    # Label Operations - Write
    # =========================================================================

    def add_annotation(
        self,
        image_base: str,
        channel: str,
        annotation: Annotation,
    ) -> None:
        """Add a single annotation.

        Args:
            image_base: Image base name
            channel: Channel name
            annotation: Annotation to add
        """
        if not self._storage:
            raise RuntimeError("LabelService not initialized")

        # LabelStorage.add_annotation handles load + mark_dirty
        self._storage.add_annotation(channel, image_base, annotation)

    def add_annotations(
        self,
        image_base: str,
        channel: str,
        annotations: List[Annotation],
    ) -> None:
        """Add multiple annotations.

        Args:
            image_base: Image base name
            channel: Channel name
            annotations: Annotations to add
        """
        if not self._storage:
            raise RuntimeError("LabelService not initialized")

        for ann in annotations:
            self._storage.add_annotation(channel, image_base, ann)

    def set_annotations(
        self,
        image_base: str,
        channel: str,
        annotations: List[Annotation],
    ) -> None:
        """Replace all annotations for an image.

        Args:
            image_base: Image base name
            channel: Channel name
            annotations: New annotations (replaces existing)
        """
        if not self._storage:
            raise RuntimeError("LabelService not initialized")

        labels = self._storage.load_labels(channel, image_base)
        labels.annotations = list(annotations)
        self._storage.mark_dirty(channel, image_base)

    def remove_annotation(
        self,
        image_base: str,
        channel: str,
        index: int,
    ) -> Optional[Annotation]:
        """Remove annotation by index.

        Returns:
            Removed annotation or None if index invalid
        """
        if not self._storage:
            raise RuntimeError("LabelService not initialized")

        labels = self._storage.load_labels(channel, image_base)
        if not labels or index < 0 or index >= len(labels.annotations):
            return None

        removed = labels.annotations.pop(index)
        self._storage.mark_dirty(channel, image_base)
        # Save immediately for manual deletions
        self._storage.save_labels(channel, image_base)
        return removed

    def update_annotation(
        self,
        image_base: str,
        channel: str,
        annotation_id: Optional[int],
        new_class_id: Optional[str] = None,
        new_attributes: Optional[Dict[str, Any]] = None,
        new_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> bool:
        """Update an existing annotation's class, bbox, and/or attributes.

        Args:
            image_base: Image base name
            channel: Channel name
            annotation_id: ID of annotation to update (None uses index search)
            new_class_id: New class ID (None keeps existing)
            new_attributes: New attributes dict (merged with existing)
            new_bbox: New bounding box (x_center, y_center, width, height) normalized

        Returns:
            True if annotation was updated, False if not found
        """
        if not self._storage:
            raise RuntimeError("LabelService not initialized")

        labels = self._storage.load_labels(channel, image_base)
        if not labels:
            return False

        # Find annotation by ID or index
        target_ann = None
        target_idx = None
        for idx, ann in enumerate(labels.annotations):
            if ann.annotation_id == annotation_id:
                target_ann = ann
                target_idx = idx
                break

        if target_ann is None:
            # Fallback: try annotation_id as direct index
            if annotation_id is not None and 0 <= annotation_id < len(labels.annotations):
                target_ann = labels.annotations[annotation_id]
                target_idx = annotation_id
            else:
                return False

        # Update class if provided
        if new_class_id is not None:
            target_ann.class_id = new_class_id

        # Update bbox if provided
        if new_bbox is not None:
            # Annotation.bbox is a tuple, we need to create a new dataclass instance
            # or update the field directly if it's mutable
            from dataclasses import replace
            labels.annotations[target_idx] = replace(target_ann, bbox=new_bbox)
            target_ann = labels.annotations[target_idx]

        # Merge attributes
        if new_attributes:
            if target_ann.attributes is None:
                target_ann.attributes = {}
            target_ann.attributes.update(new_attributes)

        self._storage.mark_dirty(channel, image_base)
        # Save immediately for manual edits
        self._storage.save_labels(channel, image_base)
        log_debug(f"Updated annotation {annotation_id} in {channel}/{image_base}", "LABELS")
        return True

    def clear_annotations(
        self,
        image_base: str,
        channel: str,
        source_filter: Optional[AnnotationSource] = None,
    ) -> int:
        """Clear annotations, optionally filtered by source.

        Args:
            image_base: Image base name
            channel: Channel name
            source_filter: If set, only clear annotations with this source

        Returns:
            Number of annotations removed
        """
        if not self._storage:
            raise RuntimeError("LabelService not initialized")

        labels = self._storage.load_labels(channel, image_base)
        if not labels:
            return 0

        if source_filter is None:
            count = len(labels.annotations)
            labels.annotations.clear()
        else:
            original = labels.annotations
            labels.annotations = [a for a in original if a.source != source_filter]
            count = len(original) - len(labels.annotations)

        if count > 0:
            self._storage.mark_dirty(channel, image_base)
        return count

    # =========================================================================
    # Source Workflow
    # =========================================================================

    def mark_reviewed(
        self,
        image_base: str,
        channel: str,
        indices: Optional[List[int]] = None,
    ) -> int:
        """Mark AUTO annotations as REVIEWED.

        Args:
            image_base: Image base name
            channel: Channel name
            indices: Specific indices to mark, or None for all AUTO

        Returns:
            Number of annotations marked
        """
        if not self._storage:
            raise RuntimeError("LabelService not initialized")

        labels = self._storage.load_labels(channel, image_base)
        if not labels:
            return 0

        count = 0
        for i, ann in enumerate(labels.annotations):
            if ann.source == AnnotationSource.AUTO:
                if indices is None or i in indices:
                    ann.source = AnnotationSource.REVIEWED
                    count += 1

        if count > 0:
            self._storage.mark_dirty(channel, image_base)
        return count

    def mark_all_reviewed(
        self,
        image_base: str,
        channel: str,
    ) -> int:
        """Mark all AUTO annotations as REVIEWED."""
        return self.mark_reviewed(image_base, channel, indices=None)

    def reject_annotation(
        self,
        image_base: str,
        channel: str,
        index: int,
    ) -> Optional[Annotation]:
        """Reject (remove) an AUTO annotation.

        This removes the annotation entirely. Use for false positives.

        Returns:
            Removed annotation or None
        """
        labels = self.get_labels(image_base, channel)
        if not labels or index < 0 or index >= len(labels.annotations):
            return None

        ann = labels.annotations[index]
        if ann.source != AnnotationSource.AUTO:
            log_warning(f"Rejecting non-AUTO annotation at index {index}")

        return self.remove_annotation(image_base, channel, index)

    def add_manual_annotation(
        self,
        image_base: str,
        channel: str,
        class_id: int,
        bbox: Tuple[float, float, float, float],
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Annotation:
        """Add a user-drawn manual annotation.

        Args:
            image_base: Image base name
            channel: Channel name
            class_id: Class ID from schema
            bbox: Normalized bbox (cx, cy, w, h)
            attributes: Optional additional attributes

        Returns:
            Created annotation
        """
        annotation = Annotation(
            class_id=class_id,
            bbox=bbox,
            source=AnnotationSource.MANUAL,
            attributes=attributes or {},
        )
        self.add_annotation(image_base, channel, annotation)
        return annotation

    # =========================================================================
    # Auto Detection
    # =========================================================================

    def auto_detect(
        self,
        image_base: str,
        channel: str,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
        replace_existing_auto: bool = True,
    ) -> List[Annotation]:
        """Run auto-detection on an image.

        Args:
            image_base: Image base name
            channel: Channel name
            image: Image as numpy array (BGR format)
            confidence_threshold: Override detector threshold
            replace_existing_auto: If True, removes existing AUTO annotations first

        Returns:
            List of new AUTO annotations added
        """
        if not self._detector:
            raise RuntimeError("No detector configured. Call set_detector() first.")
        if not self._storage:
            raise RuntimeError("LabelService not initialized")

        # Run detection
        detections = self._detector.detect(image, confidence_threshold)
        log_debug(f"Detection found {len(detections)} objects")

        # Convert to annotations with source=AUTO
        annotations = detections_to_annotations(detections, self._class_mapper)
        log_debug(f"Mapped to {len(annotations)} annotations")

        # Add model info to attributes
        model_info = self._detector.get_model_info()
        for ann in annotations:
            ann.attributes["model"] = model_info.name
            ann.attributes["model_version"] = model_info.version

        # Optionally clear existing AUTO annotations
        if replace_existing_auto:
            self.clear_annotations(image_base, channel, AnnotationSource.AUTO)

        # Add new annotations
        self.add_annotations(image_base, channel, annotations)

        return annotations

    def auto_detect_batch(
        self,
        image_bases: List[str],
        channel: str,
        image_loader: Callable[[str, str], Optional[np.ndarray]],
        confidence_threshold: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, int]:
        """Run auto-detection on multiple images.

        Args:
            image_bases: List of image base names
            channel: Channel to detect on
            image_loader: Function(base, channel) -> image array
            confidence_threshold: Override detector threshold
            progress_callback: Optional callback(current, total, base)

        Returns:
            Dict mapping image_base to number of detections
        """
        if not self._detector:
            raise RuntimeError("No detector configured")

        results: Dict[str, int] = {}
        total = len(image_bases)

        for i, base in enumerate(image_bases):
            if progress_callback:
                progress_callback(i, total, base)

            image = image_loader(base, channel)
            if image is None:
                log_warning(f"Could not load image for {base}/{channel}")
                results[base] = 0
                continue

            annotations = self.auto_detect(
                base, channel, image, confidence_threshold
            )
            results[base] = len(annotations)

        if progress_callback:
            progress_callback(total, total, "Done")

        return results

    # =========================================================================
    # Cross-Channel Projection
    # =========================================================================

    def project_to_channel(
        self,
        image_base: str,
        source_channel: str,
        target_channel: str,
        mark_as_auto: bool = True,
    ) -> List[Annotation]:
        """Project annotations from one channel to another.

        Args:
            image_base: Image base name
            source_channel: Source channel ('visible' or 'lwir')
            target_channel: Target channel
            mark_as_auto: If True, projected annotations are marked AUTO

        Returns:
            List of projected annotations added to target channel
        """
        if not self.can_project:
            raise RuntimeError("Projection not configured. Call set_projection_params()")
        if not self._storage:
            raise RuntimeError("LabelService not initialized")

        # Get source annotations
        source_labels = self.get_labels(image_base, source_channel)
        if not source_labels or not source_labels.annotations:
            return []

        # Determine direction
        if source_channel == "visible" and target_channel == "lwir":
            H = self._homography_visible_to_lwir
            src_size = self._visible_size
            dst_size = self._lwir_size
        elif source_channel == "lwir" and target_channel == "visible":
            H = self._homography_lwir_to_visible
            src_size = self._lwir_size
            dst_size = self._visible_size
        else:
            raise ValueError(f"Invalid channel pair: {source_channel} -> {target_channel}")

        # Project annotations
        projected = project_annotations_to_other_channel(
            annotations=source_labels.annotations,
            homography=H,
            src_size=src_size,
            dst_size=dst_size,
        )

        # Optionally change source
        if mark_as_auto:
            for ann in projected:
                ann.source = AnnotationSource.AUTO
                ann.attributes["projected_from"] = source_channel

        # Add to target channel
        self.add_annotations(image_base, target_channel, projected)

        return projected

    def project_single_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        source_channel: str,
        target_channel: str,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Project a single bbox between channels.

        Useful for real-time preview during annotation.

        Args:
            bbox: Normalized bbox (cx, cy, w, h)
            source_channel: Source channel
            target_channel: Target channel

        Returns:
            Projected bbox or None if projection fails
        """
        if not self.can_project:
            return None

        if source_channel == "visible" and target_channel == "lwir":
            H = self._homography_visible_to_lwir
            src_size = self._visible_size
            dst_size = self._lwir_size
        elif source_channel == "lwir" and target_channel == "visible":
            H = self._homography_lwir_to_visible
            src_size = self._lwir_size
            dst_size = self._visible_size
        else:
            return None

        return project_bbox_to_other_channel(bbox, H, src_size, dst_size)

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, image_base: str, channel: str) -> bool:
        """Save labels for a specific image.

        Returns:
            True if saved successfully
        """
        if not self._storage:
            return False
        return self._storage.save_labels(channel, image_base)

    def save_all(self, force: bool = False) -> int:
        """Save all dirty labels.

        Args:
            force: If True, save all labels regardless of dirty state (not implemented)

        Returns:
            Number of files saved
        """
        if not self._storage:
            return 0
        return self._storage.save_all_dirty()

    def save_all_sync(self) -> int:
        """Synchronously save all dirty labels.

        Use this before shutdown to ensure all data is persisted.
        """
        if not self._storage:
            return 0
        return self._storage.save_all_dirty()

    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved label changes."""
        if not self._storage:
            return False
        return self._storage.has_dirty()

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get labelling statistics for the dataset.

        Returns:
            Dict with stats like total_annotations, by_class, by_source, etc.
        """
        if not self._storage:
            return {}

        stats: Dict[str, Any] = {
            "images_with_labels": 0,
            "total_annotations": 0,
            "by_source": {"auto": 0, "reviewed": 0, "manual": 0},
            "by_channel": {"visible": 0, "lwir": 0},
            "by_class": {},
        }

        # LabelStorage cache key is (channel, base)
        for (channel, image_base), labels in self._storage._cache.items():
            if not labels.annotations:
                continue

            stats["images_with_labels"] += 1
            stats["by_channel"][channel] = stats["by_channel"].get(channel, 0) + len(labels.annotations)

            for ann in labels.annotations:
                stats["total_annotations"] += 1

                # By source
                source_key = ann.source.value
                stats["by_source"][source_key] = stats["by_source"].get(source_key, 0) + 1

                # By class
                class_id = ann.class_id
                stats["by_class"][class_id] = stats["by_class"].get(class_id, 0) + 1

        return stats

    # =========================================================================
    # Cleanup
    # =========================================================================

    def shutdown(self) -> None:
        """Shutdown service, saving any pending changes."""
        if self._storage:
            self._storage.save_all_dirty()
            log_info("LabelService shutdown complete")

        # Unload detector if needed
        if self._detector:
            self._detector.unload()
            self._detector = None

    # =========================================================================
    # UI Helpers - Overlay and Interaction
    # =========================================================================

    def get_overlay_boxes(
        self,
        image_base: str,
        channel: str,
    ) -> List[Tuple[str, float, float, float, float, Tuple[int, int, int]]]:
        """Get annotations formatted for UI overlay rendering.

        Returns list of tuples: (display_name, x_center, y_center, width, height, (r, g, b))
        All coordinates normalized [0,1].
        """
        from backend.utils.labels import class_color

        annotations = self.get_annotations(image_base, channel)
        result: List[Tuple[str, float, float, float, float, Tuple[int, int, int]]] = []

        for ann in annotations:
            # Get class name from config or use ID
            cls_name = ann.class_id
            if self._config:
                cls_def = self._config.classes.get(ann.class_id)
                if cls_def:
                    cls_name = cls_def.name

            # Display format: "id: name" if different, else just name
            display = f"{ann.class_id}: {cls_name}" if ann.class_id != cls_name else cls_name

            # Get color from class name
            color = class_color(cls_name)

            result.append((display, ann.x_center, ann.y_center, ann.width, ann.height, color))

        return result

    def label_file_path(self, image_base: str, channel: str) -> Path:
        """Get the path where labels for this image/channel would be stored."""
        if not self._dataset_path:
            raise RuntimeError("Dataset path not set")
        return self._dataset_path / "labels" / channel / f"{channel}_{image_base}.yaml"

    def label_signature(
        self,
        image_base: str,
        channel: str,
    ) -> Optional[Tuple[Any, ...]]:
        """Get a signature for cache invalidation.

        Returns tuple of (mtime, compact_boxes) or None if no labels.
        Used by overlay cache to detect changes.
        """
        labels = self.get_labels(image_base, channel)
        if not labels or not labels.annotations:
            return None

        path = self.label_file_path(image_base, channel)
        try:
            mtime = path.stat().st_mtime if path.exists() else 0.0
        except OSError:
            mtime = 0.0

        # Compact representation for comparison
        compact = tuple(
            (a.class_id, round(a.x_center, 4), round(a.y_center, 4),
             round(a.width, 4), round(a.height, 4))
            for a in labels.annotations
        )
        return (mtime, compact)

    def find_annotation_at(
        self,
        image_base: str,
        channel: str,
        x_norm: float,
        y_norm: float,
    ) -> Optional[Tuple[int, Annotation]]:
        """Find annotation containing or nearest to a point.

        Args:
            image_base: Image base name
            channel: Channel name
            x_norm: X coordinate [0,1]
            y_norm: Y coordinate [0,1]

        Returns:
            Tuple of (index, annotation) or None if no annotations
        """
        annotations = self.get_annotations(image_base, channel)
        if not annotations:
            return None

        # First try to find one that contains the point
        for idx, ann in enumerate(annotations):
            left = ann.x_center - ann.width / 2
            right = ann.x_center + ann.width / 2
            top = ann.y_center - ann.height / 2
            bottom = ann.y_center + ann.height / 2
            if left <= x_norm <= right and top <= y_norm <= bottom:
                return (idx, ann)

        # Fallback: find nearest by center
        best_idx = 0
        best_dist = float('inf')
        for idx, ann in enumerate(annotations):
            dist = (ann.x_center - x_norm) ** 2 + (ann.y_center - y_norm) ** 2
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        return (best_idx, annotations[best_idx])

    def delete_annotation_at(
        self,
        image_base: str,
        channel: str,
        x_norm: float,
        y_norm: float,
    ) -> Optional[str]:
        """Delete annotation at or near a point.

        Returns:
            Display name of deleted annotation, or None if nothing deleted
        """
        result = self.find_annotation_at(image_base, channel, x_norm, y_norm)
        if not result:
            return None

        idx, ann = result
        removed = self.remove_annotation(image_base, channel, idx)
        if not removed:
            return None

        # Format display name
        cls_name = removed.class_id
        if self._config:
            cls_def = self._config.classes.get(removed.class_id)
            if cls_def:
                cls_name = cls_def.name

        return f"{removed.class_id}: {cls_name}" if removed.class_id != cls_name else cls_name

    def add_manual_box(
        self,
        image_base: str,
        channel: str,
        class_id: str,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a manually-drawn bounding box.

        Convenience method that creates a MANUAL annotation.
        Saves immediately to ensure persistence.

        Args:
            image_base: Image base name
            channel: Channel name
            class_id: Class ID
            x_center, y_center, width, height: Normalized bbox coordinates
            attributes: Optional dict of attributes (occlusion, truncation, etc.)
        """
        ann = Annotation(
            class_id=class_id,
            bbox=(x_center, y_center, width, height),
            source=AnnotationSource.MANUAL,
            confidence=1.0,
            attributes=attributes or {},
        )
        self.add_annotation(image_base, channel, ann)
        # Save immediately for manual edits
        if self._storage:
            self._storage.save_labels(channel, image_base)

    def clear_labels(self, image_base: str) -> None:
        """Clear all labels for an image (both channels)."""
        for channel in ("visible", "lwir"):
            self.clear_annotations(image_base, channel)

    def clear_cache(self) -> None:
        """Clear internal caches (for reload scenarios)."""
        if self._storage:
            self._storage._cache.clear()
            self._storage._dirty.clear()

    # =========================================================================
    # Class Management - UI helpers
    # =========================================================================

    def class_choices(self) -> List[str]:
        """Get list of class choices for UI dropdown.

        Returns:
            List of "class_id: class_name" strings
        """
        if not self._config:
            return []
        return [
            f"{cls_def.class_id}: {cls_def.name}"
            for cls_def in sorted(
                self._config.classes.values(),
                key=lambda c: c.class_id
            )
        ]

    def class_id_for_value(self, value: str) -> Optional[str]:
        """Resolve a user input to a class ID.

        Handles formats like:
        - "0" (direct ID)
        - "person" (class name)
        - "0: person" (combo format)

        Returns:
            The class_id or None if not found
        """
        if not self._config:
            # No config - just return value as-is if non-empty
            return value.strip() if value and value.strip() else None

        value = value.strip()
        if not value:
            return None

        # Handle "id: name" format
        if ":" in value:
            leading = value.split(":", 1)[0].strip()
            if leading and leading in self._config.classes:
                return leading

        # Direct ID match
        if value in self._config.classes:
            return value

        # Name match (case-insensitive)
        value_lower = value.lower()
        for cls_id, cls_def in self._config.classes.items():
            if cls_def.name.lower() == value_lower:
                return cls_id
            # Also check aliases
            for alias in cls_def.aliases:
                if alias.lower() == value_lower:
                    return cls_id

        return None

    def copy_config_to_dataset(self, source_path: Path) -> None:
        """Copy a config YAML to the dataset's labels directory."""
        if not self._dataset_path:
            return

        dst = self._dataset_path / "labels" / "labels.yaml"
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Only write if different
            if dst.exists():
                if dst.read_text(encoding="utf-8") == source_path.read_text(encoding="utf-8"):
                    return
            dst.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass  # Non-fatal
