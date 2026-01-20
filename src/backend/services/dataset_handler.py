"""DatasetHandler: Manages cache and stats for a single dataset with debounced saving."""
from __future__ import annotations

import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from PyQt6.QtCore import QMetaObject, QObject, Qt, QTimer, pyqtSignal

from backend.services.stats_manager import DatasetStats, empty_stats_dict
from backend.services.summary_derivation import derive_summary_from_entry
from common.dict_helpers import get_dict_path
from common.log_utils import log_debug, log_error, log_info, log_warning
from common.timing import timed
from common.yaml_utils import get_timestamp_fields, load_yaml, save_yaml
from config import get_config

if TYPE_CHECKING:
    from backend.services.dataset_session import DatasetSession
    from backend.services.workspace_manager import WorkspaceManager


class SummaryCache:
    """Lightweight summary stored in .summary_cache.yaml for fast workspace scanning.

    Uses the unified stats dict format from derive_summary_from_entry().
    Structure:
        dataset_info:
            note: str
            last_updated: float
            last_updated_str: str
        stats:
            img: {total, removed}
            tagged: {user: {...}, auto: {...}}
            removed: {user: {...}, auto: {...}}
            calibration: {total, user: {...}, auto: {...}, outlier: {...}}
            sweep: {duplicates, quality, patterns}
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        """Initialize from dict data or with defaults.

        Args:
            data: Complete summary dict (from yaml.safe_load or derive_summary_from_entry).
                  If None, uses empty defaults.
        """
        if data is None:
            self._data = {
                "dataset_info": {
                    "note": "",
                    "last_updated": 0.0,
                    "last_updated_str": "",
                },
                "stats": empty_stats_dict(),
            }
        else:
            self._data = deepcopy(data)
            # Ensure stats exists
            if "stats" not in self._data:
                self._data["stats"] = empty_stats_dict()

    # =========================================================================
    # Core access
    # =========================================================================

    @property
    def data(self) -> Dict[str, Any]:
        """Get internal data dict."""
        return self._data

    def to_dict(self) -> Dict[str, Any]:
        """Return a copy for serialization."""
        return deepcopy(self._data)

    # =========================================================================
    # Dataset info accessors
    # =========================================================================

    @property
    def dataset_info(self) -> Dict[str, Any]:
        return self._data.get("dataset_info", {})

    def get_note(self) -> str:
        return get_dict_path(self._data, "dataset_info.note", "", str) or ""

    def set_note(self, note: str) -> None:
        if "dataset_info" not in self._data:
            self._data["dataset_info"] = {}
        self._data["dataset_info"]["note"] = note

    def update_timestamp(self) -> None:
        """Update last_updated timestamp."""
        now = time.time()
        if "dataset_info" not in self._data:
            self._data["dataset_info"] = {}
        self._data["dataset_info"]["last_updated"] = now
        self._data["dataset_info"]["last_updated_str"] = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")

    # =========================================================================
    # Sweep flags
    # =========================================================================

    def get_sweep_duplicates_done(self) -> bool:
        return get_dict_path(self._data, "stats.sweep.duplicates", False, bool) or False

    def set_sweep_duplicates_done(self, value: bool) -> None:
        self._ensure_stats()
        self._data["stats"]["sweep"]["duplicates"] = value

    def get_sweep_quality_done(self) -> bool:
        return get_dict_path(self._data, "stats.sweep.quality", False, bool) or False

    def set_sweep_quality_done(self, value: bool) -> None:
        self._ensure_stats()
        self._data["stats"]["sweep"]["quality"] = value

    def get_sweep_patterns_done(self) -> bool:
        return get_dict_path(self._data, "stats.sweep.patterns", False, bool) or False

    def set_sweep_patterns_done(self, value: bool) -> None:
        self._ensure_stats()
        self._data["stats"]["sweep"]["patterns"] = value

    def _ensure_stats(self) -> None:
        """Ensure stats structure exists."""
        if "stats" not in self._data:
            self._data["stats"] = empty_stats_dict()
        if "sweep" not in self._data["stats"]:
            self._data["stats"]["sweep"] = {"duplicates": False, "quality": False, "patterns": False}

    # =========================================================================
    # Conversion to/from DatasetStats
    # =========================================================================

    def to_stats(self) -> DatasetStats:
        """Convert to DatasetStats using the internal stats dict directly."""
        stats_data = self._data.get("stats", {})
        return DatasetStats(stats_data)

    @staticmethod
    def from_stats(stats: DatasetStats, note: str = "") -> 'SummaryCache':
        """Create SummaryCache from DatasetStats."""
        summary = SummaryCache()
        summary._data["dataset_info"] = {
            "note": note,
            **get_timestamp_fields(),
        }
        summary._data["stats"] = stats.to_dict()
        return summary


class DatasetHandler(QObject):
    """Handler for a single dataset - manages cache loading, saving, and stats."""

    # Signal emitted when save status changes ("dirty", "saving", "saved", "error")
    saveStatusChanged = pyqtSignal(str)

    def __init__(self, dataset_path: Path, workspace_manager: Optional['WorkspaceManager'] = None) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.workspace_manager = workspace_manager
        self.session: Optional[DatasetSession] = None
        self.summary: Optional[SummaryCache] = None

        # Debounce timer for saving
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(2000)  # 2 second debounce
        self._save_timer.timeout.connect(self._flush_cache)

        self._dirty = False
        self._save_status = "saved"
        self._is_collection = self._check_is_collection()

    def _check_is_collection(self) -> bool:
        """Check if this dataset is a collection (has subdirs with images)."""
        try:
            for child in self.dataset_path.iterdir():
                if child.is_dir() and not child.name.startswith("."):
                    # Check if child has images
                    lwir = child / "lwir"
                    visible = child / "visible"
                    if (lwir.exists() and any(lwir.glob("*.png"))) or \
                       (visible.exists() and any(visible.glob("*.jpg"))):
                        return True
        except OSError:
            pass
        return False

    @property
    def is_collection(self) -> bool:
        return self._is_collection

    @timed
    def load_summary(self, *, force_rebuild: bool = False) -> SummaryCache:
        """Load or rebuild summary cache.

        Priority:
        1. Return cached summary if available and not force_rebuild
        2. Try to rebuild from .image_labels.yaml (authoritative source)
        3. Load from .summary_cache.yaml if exists
        4. Return empty summary
        """
        if self.summary is not None and not force_rebuild:
            return self.summary

        # Try to rebuild from full cache (authoritative source)
        full_cache_path = self.dataset_path / ".image_labels.yaml"
        if full_cache_path.exists():
            log_info("✓ Found .image_labels.yaml - rebuilding from full cache", "HANDLER")
            self.summary = self._rebuild_summary_from_full_cache()
            return self.summary

        # Fallback: Load from .summary_cache.yaml
        config = get_config()
        summary_path = self.dataset_path / config.summary_cache_filename
        if summary_path.exists():
            try:
                data = load_yaml(summary_path)
                if isinstance(data, dict):
                    self.summary = SummaryCache(data)
                    log_info(f"✓ Loaded summary from {config.summary_cache_filename}", "HANDLER")
                    return self.summary
            except Exception as e:
                log_warning(f"Failed to load summary cache: {e}", "HANDLER")

        # No cache available - return empty
        log_info("No cache found - returning empty summary", "HANDLER")
        self.summary = SummaryCache()
        return self.summary

    def _rebuild_summary_from_full_cache(self) -> SummaryCache:
        """Rebuild summary from .image_labels.yaml (authoritative source)."""
        full_cache_path = self.dataset_path / ".image_labels.yaml"
        try:
            cache_entry = load_yaml(full_cache_path)
        except Exception as e:
            log_error(f"Failed to load full cache: {e}", "HANDLER")
            return SummaryCache()

        if not isinstance(cache_entry, dict):
            log_warning("Full cache is not a dict", "HANDLER")
            return SummaryCache()

        # Count real pairs from filesystem (total_pairs is NOT persisted, always derived)
        real_count = self._count_pairs_from_filesystem()
        cache_entry['total_pairs'] = real_count

        log_debug(f"Loaded cache for {self.dataset_path.name}: pairs={real_count}")

        # ✅ DERIVE summary from entry (single source of truth)
        summary_dict = derive_summary_from_entry(cache_entry)
        summary = SummaryCache(summary_dict)

        # Log for debugging
        stats = summary.to_stats()
        log_info(f"✅ DERIVED summary for {self.dataset_path.name}: "
                 f"total_pairs={stats.total_pairs}, "
                 f"tagged_manual={stats.tagged_manual}, tagged_auto={stats.tagged_auto}, "
                 f"calib_marked={stats.calibration_marked}", "HANDLER")

        log_info(f"✓ Rebuilt summary: pairs={stats.total_pairs}, "
                 f"calib={stats.calibration_marked}/{stats.calibration_both}, "
                 f"manual={stats.tagged_manual}, auto={stats.tagged_auto}", "HANDLER")

        return summary

    def _count_pairs_from_filesystem(self) -> int:
        """Count actual image pairs from filesystem.

        Handles both formats:
        - With prefix: lwir_000000.png / visible_000000.jpg
        - Without prefix: 000000.png / 000000.jpg
        """
        lwir_dir = self.dataset_path / "lwir"
        visible_dir = self.dataset_path / "visible"

        if not lwir_dir.exists() or not visible_dir.exists():
            return 0

        def _extract_bases(folder: Path) -> set:
            bases = set()
            # Accept both .png and .jpg
            for f in folder.iterdir():
                if f.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                stem = f.stem
                # Strip prefix if present (lwir_xxx or visible_xxx)
                for prefix in ("lwir_", "visible_"):
                    if stem.startswith(prefix):
                        stem = stem[len(prefix):]
                        break
                bases.add(stem)
            return bases

        lwir_bases = _extract_bases(lwir_dir)
        visible_bases = _extract_bases(visible_dir)
        # Use intersection for actual pairs (both sides must exist)
        return len(lwir_bases & visible_bases)

    def mark_dirty(self) -> None:
        """Mark cache dirty and restart debounce timer."""
        log_debug(f"mark_dirty() called for {self.dataset_path.name}, timer will fire in 2s", "HANDLER")
        self._dirty = True
        self._save_status = "pending"

        try:
            self.saveStatusChanged.emit("pending")
        except RuntimeError:
            pass

        QMetaObject.invokeMethod(
            self._save_timer,
            "start",
            Qt.ConnectionType.QueuedConnection
        )

        if self.workspace_manager:
            self.workspace_manager.notify_dataset_changed(self.dataset_path)

    def _flush_cache(self) -> None:
        """Actually save summary and full cache to disk. Collections are skipped."""
        if not self._dirty:
            log_debug(f"_flush_cache called but not dirty for {self.dataset_path.name}", "HANDLER")
            return

        if self._is_collection:
            log_debug(f"{self.dataset_path.name} is a COLLECTION - skipping cache save", "HANDLER")
            self._dirty = False
            self._save_status = "saved"
            self.saveStatusChanged.emit("saved")
            return

        self._save_status = "saving"
        self.saveStatusChanged.emit("saving")

        log_info(f"*** FLUSHING CACHE for {self.dataset_path.name} ***", "HANDLER")

        # Update summary from session if loaded
        if self.session is not None:
            log_debug("Updating summary from active session", "HANDLER")
            self.update_summary_from_session()

        if self.summary is not None:
            config = get_config()
            summary_path = self.dataset_path / config.summary_cache_filename
            try:
                self.summary.update_timestamp()
                save_yaml(summary_path, self.summary.to_dict())
                log_info(f"✓ Saved summary to {config.summary_cache_filename}", "HANDLER")
                self._save_status = "saved"
            except Exception as e:
                log_error(f"Failed to save summary cache: {e}", "HANDLER")
                self._save_status = "error"

        self._dirty = False
        self.saveStatusChanged.emit(self._save_status)

    def force_flush(self) -> None:
        """Force immediate save (called on app close)."""
        if self._save_timer.isActive():
            self._save_timer.stop()
        self._flush_cache()

    def update_summary_from_session(self) -> None:
        """Update summary cache from active session state."""
        if self.session is None or self.summary is None:
            return

        # Rebuild from full cache to get latest state
        self.summary = self._rebuild_summary_from_full_cache()

    def update_from_workspace_info(self, info: 'WorkspaceDatasetInfo') -> None:
        """Update summary from WorkspaceDatasetInfo (used during workspace refresh)."""
        self.summary = SummaryCache.from_stats(info.stats, note=info.note)
        self.mark_dirty()


# Import at end to avoid circular imports
from backend.services.workspace_inspector import WorkspaceDatasetInfo
