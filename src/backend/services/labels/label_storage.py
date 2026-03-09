"""Label storage and persistence for multiespectral annotation system.

This module handles loading/saving labels per dataset and channel.

WHY SEPARATE FROM cache_data?
    Unlike marks/calibration (1 YAML file for entire dataset), labels are
    stored as individual files per image. This allows:
    - Efficient partial loading (only load labels when viewing an image)
    - Git-friendly diffs (one file per image, not monolithic)
    - Scalability (datasets can have thousands of labeled images)

    Therefore, LabelStorage has its OWN cache dict as source of truth,
    following the same dirty-tracking pattern as the main cache system.

Storage structure:
    workspace/
    ├── labels_config.yaml          # Workspace-wide label schema config
    └── dataset/
        └── labels/
            ├── visible/
            │   └── {base}.yaml     # Labels for visible images
            └── lwir/
                └── {base}.yaml     # Labels for LWIR images

Config resolution:
    1. ``{workspace}/labels_config.yaml``  (single source per workspace)
    2. ``config/labels_multiespectral_dataset.yaml``  (repo default)

Source of Truth:
    _cache: Dict[(channel, base) -> ImageLabels] is the in-memory truth.
    Disk files are loaded on-demand and saved when dirty.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from common.log_utils import log_debug, log_info, log_warning
from common.yaml_utils import load_yaml, save_yaml

from backend.services.labels.label_types import (
    Annotation,
    AnnotationSource,
    ImageLabels,
)

# -------------------------------------------------------------------------
# Workspace-level config
# -------------------------------------------------------------------------
WORKSPACE_CONFIG_FILENAME = "labels_config.yaml"


def find_labels_config(start_dir: Path, max_levels: int = 5) -> Optional[Path]:
    """Walk up from *start_dir* looking for ``labels_config.yaml``.

    Returns the first match found, or ``None`` if none within *max_levels*
    parent directories.  This is the primary mechanism for
    ``_load_class_name_map`` to locate the workspace config without
    requiring an explicit path.
    """
    current = start_dir.resolve()
    for _ in range(max_levels):
        candidate = current / WORKSPACE_CONFIG_FILENAME
        if candidate.exists():
            return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


class LabelStorage:
    """Manages label persistence for a single dataset.

    Source of Truth: _cache dict (not disk files).
    Pattern: Same dirty-tracking as main cache system.

    Thread Safety: NOT thread-safe. Use from main thread only.
    """

    LABELS_DIR = "labels"

    def __init__(self, dataset_path: Path) -> None:
        """Initialize storage for a dataset."""
        self.dataset_path = dataset_path
        self._labels_dir = dataset_path / self.LABELS_DIR

        # SOURCE OF TRUTH: in-memory cache
        self._cache: Dict[Tuple[str, str], ImageLabels] = {}
        self._dirty: Set[Tuple[str, str]] = set()

    # =========================================================================
    # Label I/O
    # =========================================================================

    def _label_path(self, channel: str, base: str) -> Path:
        """Get path to label file for a specific image."""
        return self._labels_dir / channel / f"{base}.yaml"

    def load_labels(self, channel: str, base: str) -> ImageLabels:
        """Load labels for an image.

        Returns existing labels or creates empty ImageLabels if none exist.

        Args:
            channel: "visible" or "lwir"
            base: Image base name (without extension)

        Returns:
            ImageLabels object (may be empty)
        """
        cache_key = (channel, base)

        # Check memory cache first
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try to load from disk
        path = self._label_path(channel, base)
        if path.exists():
            try:
                data = load_yaml(path)
                if data:
                    labels = ImageLabels.from_dict(data)
                    self._cache[cache_key] = labels
                    return labels
            except Exception as e:
                log_warning(f"Failed to load labels from {path}: {e}", "LABELS")

        # Create empty labels
        labels = ImageLabels(
            image_file=f"{channel}_{base}.png",
            channel=channel,
        )
        self._cache[cache_key] = labels
        return labels

    def save_labels(self, channel: str, base: str, labels: Optional[ImageLabels] = None) -> bool:
        """Save labels for an image.

        AUTO annotations are **never** persisted.  Only REVIEWED and
        MANUAL annotations are written to disk.  If no persistent
        annotations remain the file is deleted.

        Args:
            channel: "visible" or "lwir"
            base: Image base name
            labels: Optional ImageLabels to save (uses cache if None)

        Returns:
            True if save successful
        """
        cache_key = (channel, base)
        path = self._label_path(channel, base)

        # Get labels from cache if not provided
        if labels is None:
            labels = self._cache.get(cache_key)
            if labels is None:
                return True  # Nothing to save

        # Build a persistent-only snapshot (exclude AUTO)
        persistent = [a for a in labels.annotations if a.source != AnnotationSource.AUTO]

        # Delete file if no persistent annotations remain
        if not persistent:
            if path.exists():
                try:
                    path.unlink()
                    log_debug(f"Deleted labels file (no persistent annotations): {path}", "LABELS")
                except Exception as e:
                    log_warning(f"Failed to delete {path}: {e}", "LABELS")
                    return False
            # Keep the in-memory cache intact (AUTO labels stay alive)
            self._dirty.discard(cache_key)
            return True

        # Save persistent annotations to disk
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Build dict with only persistent annotations
            data = labels.to_dict()
            data["annotations"] = [a.to_dict() for a in persistent]
            save_yaml(path, data)
            self._dirty.discard(cache_key)
            log_debug(f"Saved {len(persistent)} labels to {path} "
                      f"({len(labels.annotations) - len(persistent)} AUTO skipped)", "LABELS")
            return True
        except Exception as e:
            log_warning(f"Failed to save labels to {path}: {e}", "LABELS")
            return False

    def save_all_dirty(self) -> int:
        """Save all dirty labels.

        Returns number of files saved.
        """
        saved = 0
        for channel, base in list(self._dirty):
            if self.save_labels(channel, base):
                saved += 1
        return saved

    # =========================================================================
    # Dirty Tracking
    # =========================================================================

    def mark_dirty(self, channel: str, base: str) -> None:
        """Mark labels as dirty (needs saving)."""
        self._dirty.add((channel, base))

    def is_dirty(self, channel: str, base: str) -> bool:
        """Check if labels are dirty."""
        return (channel, base) in self._dirty

    def has_dirty(self) -> bool:
        """Check if any labels are dirty."""
        return len(self._dirty) > 0

    def get_dirty_count(self) -> int:
        """Get number of dirty label files."""
        return len(self._dirty)

    # =========================================================================
    # Annotation Operations
    # =========================================================================

    def add_annotation(
        self,
        channel: str,
        base: str,
        annotation: Annotation,
    ) -> int:
        """Add an annotation to an image.

        Args:
            channel: "visible" or "lwir"
            base: Image base name
            annotation: Annotation to add

        Returns:
            Assigned annotation ID
        """
        labels = self.load_labels(channel, base)
        ann_id = labels.add_annotation(annotation)
        self.mark_dirty(channel, base)
        return ann_id

    def remove_annotation(
        self,
        channel: str,
        base: str,
        annotation_id: int,
    ) -> bool:
        """Remove an annotation from an image.

        Returns True if annotation was found and removed.
        """
        labels = self.load_labels(channel, base)
        if labels.remove_annotation(annotation_id):
            self.mark_dirty(channel, base)
            return True
        return False

    def update_annotation(
        self,
        channel: str,
        base: str,
        annotation_id: int,
        **updates: Any,
    ) -> bool:
        """Update annotation attributes.

        Args:
            channel: "visible" or "lwir"
            base: Image base name
            annotation_id: ID of annotation to update
            **updates: Attribute updates (class_id, bbox, attributes)

        Returns:
            True if annotation was found and updated
        """
        labels = self.load_labels(channel, base)
        ann = labels.get_annotation(annotation_id)
        if ann is None:
            return False

        if "class_id" in updates:
            ann.class_id = updates["class_id"]
        if "bbox" in updates:
            ann.bbox = tuple(updates["bbox"])  # type: ignore
        if "attributes" in updates:
            ann.attributes.update(updates["attributes"])

        self.mark_dirty(channel, base)
        return True

    def clear_labels(self, channel: str, base: str) -> None:
        """Remove all labels for an image."""
        cache_key = (channel, base)
        path = self._label_path(channel, base)

        # Remove from cache
        self._cache.pop(cache_key, None)
        self._dirty.discard(cache_key)

        # Delete file
        if path.exists():
            try:
                path.unlink()
            except Exception as e:
                log_warning(f"Failed to delete {path}: {e}", "LABELS")

    # =========================================================================
    # Queries
    # =========================================================================

    def get_labeled_bases(self, channel: str) -> List[str]:
        """Get list of bases that have labels for a channel."""
        channel_dir = self._labels_dir / channel
        if not channel_dir.exists():
            return []

        bases = []
        for path in channel_dir.rglob("*.yaml"):
            # For collections, labels live in subdirs (e.g. child_key/base.yaml)
            rel = path.relative_to(channel_dir)
            bases.append(str(rel.with_suffix("")))
        return sorted(bases)

    def get_annotation_count(self, channel: str, base: str) -> int:
        """Get number of annotations for an image."""
        labels = self.load_labels(channel, base)
        return len(labels.annotations)

    def get_total_annotation_count(self) -> Dict[str, int]:
        """Get total annotation counts by channel."""
        counts: Dict[str, int] = {"visible": 0, "lwir": 0}
        for channel in ("visible", "lwir"):
            for base in self.get_labeled_bases(channel):
                counts[channel] += self.get_annotation_count(channel, base)
        return counts

    def has_labels(self, channel: str, base: str) -> bool:
        """Check if an image has any labels."""
        cache_key = (channel, base)
        if cache_key in self._cache:
            return not self._cache[cache_key].is_empty()
        return self._label_path(channel, base).exists()

    # =========================================================================
    # Cache Management
    # =========================================================================

    def clear_cache(self) -> None:
        """Clear in-memory cache (does not affect disk)."""
        # Save dirty entries first
        self.save_all_dirty()
        self._cache.clear()

    def invalidate(self, channel: str, base: str) -> None:
        """Invalidate cache entry (forces reload on next access)."""
        cache_key = (channel, base)
        self._cache.pop(cache_key, None)
        self._dirty.discard(cache_key)
