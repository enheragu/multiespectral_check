"""DatasetHandler: Manages cache and stats for a single dataset with debounced saving."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING, Any
import time

from PyQt6.QtCore import QTimer, QObject, pyqtSignal

from backend.services.dataset_session import DatasetSession
from backend.services.stats_manager import DatasetStats
from backend.services.cache_service import load_dataset_cache_file
from backend.services.summary_derivation import derive_summary_from_entry
from common.dict_helpers import get_dict_path
from common.timing import timed
from common.log_utils import log_debug, log_info, log_warning, log_error
import yaml

if TYPE_CHECKING:
    from backend.services.workspace_manager import WorkspaceManager


SUMMARY_CACHE_FILENAME = ".summary_cache.yaml"


@dataclass(init=False)
class SummaryCache:
    """Lightweight summary stored in .summary_cache.yaml for fast workspace scanning.

    Structure follows a clean hierarchy:
    - dataset_info: General metadata and sweep status
    - img_number: Image counts
    - removed_reasons: Reasons for deleted images
    - tagged_user_to_delete_reasons: User-marked images by reason
    - tagged_auto_to_delete_reasons: Auto-detected issues by reason
    - calibration: Calibration board detection and outliers
    """
    dataset_info: Dict[str, Any] = field(default_factory=lambda: {
        "note": "",
        "last_updated": 0.0,
        "auto_sweeps": {
            "sweep_missing_done": False,
            "sweep_duplicates_done": False,
            "sweep_patterns_done": False,
            "sweep_quality_done": False,
        }
    })
    img_number: Dict[str, int] = field(default_factory=lambda: {
        "num_pairs": 0,
        "removed_pairs": 0,
        "tagged_user_to_delete": 0,
        "tagged_auto_to_delete": 0,
    })
    removed_reasons: Dict[str, int] = field(default_factory=dict)
    removed_user_reasons: Dict[str, int] = field(default_factory=dict)
    removed_auto_reasons: Dict[str, int] = field(default_factory=dict)
    pattern_matches: Dict[str, int] = field(default_factory=dict)  # pattern_name -> count
    tagged_user_to_delete_reasons: Dict[str, int] = field(default_factory=lambda: {
        "user_marked": 0,
        "duplicate": 0,
        "missing_pair": 0,
        "sync": 0,
        "blurry": 0,
        "motion_blur": 0,
        "pattern": 0,
    })
    tagged_auto_to_delete_reasons: Dict[str, int] = field(default_factory=lambda: {
        "duplicate": 0,
        "missing_pair": 0,
        "sync": 0,
        "blurry": 0,
        "motion_blur": 0,
        "pattern": 0,
    })
    calibration: Dict[str, int] = field(default_factory=lambda: {
        "marked": 0,
        "found_both_chessboard": 0,
        "found_only_lwir_chessboard": 0,
        "found_only_visible_chessboard": 0,
        "found_none_chessboard": 0,
        "outlier_discarded_lwir": 0,
        "outlier_discarded_visible": 0,
        "outlier_discarded_stereo": 0,
    })

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        """Initialize from dict data or with defaults.

        Args:
            data: Complete dict with all fields (from yaml.safe_load). If None, uses defaults.
        """
        if data is None:
            # Default initialization
            self.dataset_info = {
                "note": "",
                "last_updated": 0.0,
                "sweep_flags": {
                    "missing": False,
                    "duplicates": False,
                    "patterns": False,
                    "quality": False,
                },
            }
            self.img_number = {
                "num_pairs": 0,
                "removed_pairs": 0,
                "tagged_user_to_delete": 0,
                "tagged_auto_to_delete": 0,
            }
            self.removed_reasons = {}
            self.removed_user_reasons = {}
            self.removed_auto_reasons = {}
            self.pattern_matches = {}
            self.tagged_user_to_delete_reasons = {
                "user_marked": 0,
                "duplicate": 0,
                "missing_pair": 0,
                "sync": 0,
                "blurry": 0,
                "motion_blur": 0,
                "pattern": 0,
            }
            self.tagged_auto_to_delete_reasons = {
                "duplicate": 0,
                "missing_pair": 0,
                "sync": 0,
                "blurry": 0,
                "motion_blur": 0,
                "pattern": 0,
            }
            self.calibration = {
                "marked": 0,
                "found_both_chessboard": 0,
                "found_only_lwir_chessboard": 0,
                "found_only_visible_chessboard": 0,
                "found_none_chessboard": 0,
                "outlier_discarded_lwir": 0,
                "outlier_discarded_visible": 0,
                "outlier_discarded_stereo": 0,
            }
        else:
            # Load from data dict
            self.dataset_info = data.get("dataset_info", {})
            self.img_number = data.get("img_number", {})
            self.removed_reasons = data.get("removed_reasons", {})
            self.removed_user_reasons = data.get("removed_user_reasons", {})
            self.removed_auto_reasons = data.get("removed_auto_reasons", {})
            self.pattern_matches = data.get("pattern_matches", {})
            self.tagged_user_to_delete_reasons = data.get("tagged_user_to_delete_reasons", {})
            self.tagged_auto_to_delete_reasons = data.get("tagged_auto_to_delete_reasons", {})
            self.calibration = data.get("calibration", {})

    # Getters para acceso externo seguro
    def get_note(self) -> str:
        """Get note from dataset_info."""
        return str(self.dataset_info.get("note", ""))

    def get_sweep_duplicates_done(self) -> bool:
        """Check if duplicates sweep is done."""
        return bool(get_dict_path(self.dataset_info, "sweep_flags.duplicates", False))

    def set_sweep_duplicates_done(self, value: bool) -> None:
        """Set duplicates sweep done status."""
        if "sweep_flags" not in self.dataset_info:
            self.dataset_info["sweep_flags"] = {}
        self.dataset_info["sweep_flags"]["duplicates"] = value

    def get_sweep_missing_done(self) -> bool:
        """Check if missing pairs sweep is done."""
        return bool(get_dict_path(self.dataset_info, "sweep_flags.missing", False))

    def set_sweep_missing_done(self, value: bool) -> None:
        """Set missing pairs sweep done status."""
        if "sweep_flags" not in self.dataset_info:
            self.dataset_info["sweep_flags"] = {}
        self.dataset_info["sweep_flags"]["missing"] = value

    def get_sweep_quality_done(self) -> bool:
        """Check if quality sweep is done."""
        return bool(get_dict_path(self.dataset_info, "sweep_flags.quality", False))

    def get_sweep_patterns_done(self) -> bool:
        """Check if patterns sweep is done."""
        return bool(get_dict_path(self.dataset_info, "sweep_flags.patterns", False))

    # Métodos internos de modificación
    def update_timestamp(self) -> None:
        """Update last_updated timestamp."""
        self.dataset_info["last_updated"] = time.time()


    def to_stats(self) -> DatasetStats:
        """Convert summary to DatasetStats - acceso directo a campos."""
        # Get sweep flags from sweep_flags structure (ensure bool)
        sweep_dups = bool(get_dict_path(self.dataset_info, "sweep_flags.duplicates", False))
        sweep_qual = bool(get_dict_path(self.dataset_info, "sweep_flags.quality", False))
        sweep_pats = bool(get_dict_path(self.dataset_info, "sweep_flags.patterns", False))

        # Calculate calibration_partial from components
        calib_partial = (
            self.calibration.get("found_only_lwir_chessboard", 0) +
            self.calibration.get("found_only_visible_chessboard", 0)
        )

        return DatasetStats(
            total_pairs=self.img_number.get("num_pairs", 0),
            removed_total=self.img_number.get("removed_pairs", 0),
            tagged_manual=self.img_number.get("tagged_user_to_delete", 0),
            tagged_auto=self.img_number.get("tagged_auto_to_delete", 0),
            removed_by_reason=self.removed_reasons.copy(),
            tagged_by_reason=self.tagged_user_to_delete_reasons.copy(),
            tagged_auto_by_reason=self.tagged_auto_to_delete_reasons.copy(),
            pattern_matches=self.pattern_matches.copy(),
            calibration_marked=self.calibration.get("marked", 0),
            calibration_both=self.calibration.get("found_both_chessboard", 0),
            calibration_partial=calib_partial,
            calibration_missing=self.calibration.get("found_none_chessboard", 0),
            outlier_lwir=self.calibration.get("outlier_discarded_lwir", 0),
            outlier_visible=self.calibration.get("outlier_discarded_visible", 0),
            outlier_stereo=self.calibration.get("outlier_discarded_stereo", 0),
            sweep_duplicates_done=sweep_dups,
            sweep_quality_done=sweep_qual,
            sweep_patterns_done=sweep_pats,
        )

    @staticmethod
    def from_stats(stats: DatasetStats, note: str = "") -> 'SummaryCache':
        """Create summary from DatasetStats efficiently."""
        summary = SummaryCache()

        # Direct field assignment - cada cual gestiona su información
        summary.img_number = {
            "num_pairs": stats.total_pairs,
            "removed_pairs": stats.removed_total,
            "tagged_user_to_delete": stats.tagged_manual,
            "tagged_auto_to_delete": stats.tagged_auto,
        }
        summary.removed_reasons = stats.removed_by_reason.copy()
        summary.removed_user_reasons = stats.removed_user_by_reason.copy()
        summary.removed_auto_reasons = stats.removed_auto_by_reason.copy()
        summary.tagged_user_to_delete_reasons = stats.tagged_by_reason.copy()
        summary.tagged_auto_to_delete_reasons = stats.tagged_auto_by_reason.copy()
        summary.pattern_matches = {k[8:]: v for k, v in stats.tagged_auto_by_reason.items() if k.startswith("pattern:")}

        summary.calibration = {
            "marked": stats.calibration_marked,
            "found_both_chessboard": stats.calibration_both,
            "found_only_lwir_chessboard": stats.calibration_partial // 2,
            "found_only_visible_chessboard": stats.calibration_partial - (stats.calibration_partial // 2),
            "found_none_chessboard": stats.calibration_missing,
            "outlier_discarded_lwir": stats.outlier_lwir,
            "outlier_discarded_visible": stats.outlier_visible,
            "outlier_discarded_stereo": stats.outlier_stereo,
        }

        summary.dataset_info = {
            "note": note,
            "last_updated": time.time(),
            "sweep_flags": {
                "duplicates": stats.sweep_duplicates_done,
                "quality": stats.sweep_quality_done,
                "patterns": stats.sweep_patterns_done,
            }
        }

        return summary


class DatasetHandler(QObject):
    """
    Manages single dataset cache, stats, and debounced saving.

    Features:
    - Lazy loading of full cache (DatasetSession) only when needed
    - Fast summary loading via .summary_cache.yaml (ONLY for leaf datasets, NOT collections)
    - Debounced saving (2 seconds) to avoid excessive I/O
    - Force flush on dataset change or GUI close
    - Collections are virtual: they aggregate from children without their own cache
    """

    # Signal emitted when save status changes
    saveStatusChanged = pyqtSignal(str)  # "saved", "pending", or "saving"

    def __init__(self, dataset_path: Path, workspace_manager: Optional['WorkspaceManager'] = None) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.workspace_manager = workspace_manager
        self.summary: Optional[SummaryCache] = None
        self.session: Optional[DatasetSession] = None
        self._save_status = "saved"  # "saved", "pending", "saving"
        self._is_collection = self._detect_collection()
        self._cached_pair_count: tuple[int, float] | None = None  # (count, timestamp)

        # Debounced save timer
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(2000)  # 2 seconds debounce
        self._save_timer.timeout.connect(self._flush_cache)
        self._dirty = False

    def _detect_collection(self) -> bool:
        """Check if this path is a collection (has subdirectories with caches)."""
        from backend.services.collection import Collection
        return Collection.is_collection_dir(self.dataset_path)

    def connect_session(self, session: 'DatasetSession') -> None:
        """Connect a DatasetSession to this handler so changes trigger saves."""
        self.session = session

        # Hook into session's mark_cache_dirty to trigger our debounced save
        original_mark_dirty = session.mark_cache_dirty

        def mark_dirty_wrapper() -> None:
            original_mark_dirty()
            self.mark_dirty()  # Trigger debounced save in handler
            log_debug("Session marked dirty, debounced save triggered", "HANDLER")

        session.mark_cache_dirty = mark_dirty_wrapper  # type: ignore[method-assign]

        log_debug(f"Connected session for {self.dataset_path.name}", "HANDLER")

    @timed
    def load_summary(self, *, force_rebuild: bool = False) -> SummaryCache:
        """
        Load lightweight summary from .summary_cache.yaml.
        Falls back to rebuilding from .reviewer_cache.yaml if missing or invalid.
        Always validates against filesystem to detect stale caches.

        For COLLECTIONS: Returns empty summary (collections don't have their own cache).

        CRITICAL: If .reviewer_cache.yaml exists, ALWAYS rebuild from it to ensure calibration
        and other complex data is accurate. .summary_cache.yaml is only used as last resort.
        """
        log_debug(f"=== START load_summary for {self.dataset_path.name} ===", "HANDLER")
        log_debug(f"  is_collection={self._is_collection}", "HANDLER")
        log_debug(f"  force_rebuild={force_rebuild}", "HANDLER")
        log_debug(f"  cached_summary_exists={self.summary is not None}", "HANDLER")

        # Collections only store note in their summary cache
        if self._is_collection:
            log_debug(f"{self.dataset_path.name} is a COLLECTION - loading note only", "HANDLER")

            # Try to load existing note from summary cache
            summary_path = self.dataset_path / SUMMARY_CACHE_FILENAME
            if summary_path.exists():
                try:
                    with summary_path.open("r", encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}

                    self.summary = SummaryCache(data)
                    note = self.summary.get_note()
                    log_debug(f"Loaded note for collection: {note[:50]}...", "HANDLER")
                except Exception as e:
                    log_warning(f"Failed to load collection note: {e}", "HANDLER")
                    self.summary = SummaryCache()
            else:
                self.summary = SummaryCache()

            return self.summary

        if self.summary is not None and not force_rebuild:
            return self.summary

        # PRIORITY 1: If .reviewer_cache.yaml exists, ALWAYS use it (has complete data including calibration)
        from backend.services.cache_service import DATASET_CACHE_FILENAME
        cache_path = self.dataset_path / DATASET_CACHE_FILENAME
        log_debug(f"Checking for full cache: {cache_path.exists()}", "HANDLER")
        if cache_path.exists():
            log_info("✓ Found .reviewer_cache.yaml - rebuilding from full cache", "HANDLER")
            self.summary = self._rebuild_summary_from_full_cache()
            log_info(f"✓ Rebuilt summary: pairs={self.summary.img_number.get('num_pairs', 0)}, "
                      f"calib={self.summary.calibration.get('marked', 0)}/{self.summary.calibration.get('found_both_chessboard', 0)}, "
                      f"manual={self.summary.img_number.get('tagged_user_to_delete', 0)}, auto={self.summary.img_number.get('tagged_auto_to_delete', 0)}", "HANDLER")
            # Save to .summary_cache.yaml for faster next load if full cache is missing
            self._save_summary()
            return self.summary

        # PRIORITY 2: Try loading .summary_cache.yaml (faster but may be incomplete)
        summary_path = self.dataset_path / SUMMARY_CACHE_FILENAME
        if not force_rebuild and summary_path.exists():
            try:
                with summary_path.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}

                # Crear directamente desde data
                self.summary = SummaryCache(data)

                # ALWAYS validate against filesystem for data integrity
                real_count = self._count_real_pairs_from_filesystem()
                cached_total = self.summary.img_number.get("num_pairs", 0)
                if real_count != cached_total:
                    log_warning(f"⚠️  STALE summary cache for {self.dataset_path.name}: "
                              f"cache says {cached_total}, filesystem has {real_count}. Using it anyway (no full cache).", "HANDLER")
                    # Update with real count but keep other data
                    self.summary.img_number["num_pairs"] = real_count

                log_info(f"✓ Loaded summary cache: pairs={self.summary.img_number.get('num_pairs', 0)}, "
                          f"calib={self.summary.calibration.get('marked', 0)}/{self.summary.calibration.get('found_both_chessboard', 0)}, "
                          f"manual={self.summary.img_number.get('tagged_user_to_delete', 0)}, auto={self.summary.img_number.get('tagged_auto_to_delete', 0)}", "HANDLER")
                return self.summary
            except Exception as e:
                log_error(f"✗ Failed to load summary cache: {e}", "HANDLER")

        # PRIORITY 3: No cache exists, return empty summary
        log_info(f"No cache found for {self.dataset_path.name} - returning empty summary", "HANDLER")
        self.summary = SummaryCache()
        return self.summary

    def _count_real_pairs_from_filesystem(self) -> int:
        """Count actual image pairs by scanning the filesystem (cached briefly)."""
        from backend.dataset_loader import DatasetLoader
        now = time.time()
        if self._cached_pair_count and (now - self._cached_pair_count[1]) < 5.0:
            return self._cached_pair_count[0]
        try:
            loader = DatasetLoader(str(self.dataset_path))
            if not loader.load_dataset():
                log_error(f"Failed to load dataset {self.dataset_path.name}", "HANDLER")
                return 0
            real_count = len(loader.image_bases) if loader.image_bases else 0
            self._cached_pair_count = (real_count, now)
            log_debug(f"Real filesystem count for {self.dataset_path.name}: {real_count} pairs", "HANDLER")
            return real_count
        except Exception as e:
            log_error(f"Error counting pairs for {self.dataset_path.name}: {e}", "HANDLER")
            return 0

    def _rebuild_summary_from_full_cache(self) -> SummaryCache:
        """Rebuild summary by reading .image_labels.yaml and DERIVING summary from it."""
        from backend.services.cache_service import DATASET_CACHE_FILENAME

        cache_path = self.dataset_path / DATASET_CACHE_FILENAME
        if not cache_path.exists():
            log_debug(f"No cache file at {cache_path}", "HANDLER")
            return SummaryCache()

        # Load entry from .image_labels.yaml (SOURCE OF TRUTH)
        cache_entry = load_dataset_cache_file(cache_path)
        if not isinstance(cache_entry, dict):
            log_error(f"Cache entry not a dict for {self.dataset_path.name}", "HANDLER")
            return SummaryCache()

        cached_total = cache_entry.get('total_pairs', 0)

        # Count real pairs from filesystem
        real_count = self._count_real_pairs_from_filesystem()

        # If cache says 0 but filesystem has images, cache is COMPLETELY INVALID
        if cached_total == 0 and real_count > 0:
            log_warning(f"⚠️  INVALID CACHE for {self.dataset_path.name}: "
                      f"cache says 0 but filesystem has {real_count} pairs. Returning empty summary.", "HANDLER")
            summary = SummaryCache()
            summary.img_number["num_pairs"] = real_count
            return summary

        log_debug(f"Loaded cache for {self.dataset_path.name}: "
                  f"cached_total={cached_total}, real_count={real_count}")

        # Warn if cache is stale
        if cached_total != real_count:
            log_warning(f"⚠️  CACHE STALE for {self.dataset_path.name}: "
                      f"cache says {cached_total}, filesystem has {real_count}", "HANDLER")
            # Override with real count
            cache_entry['total_pairs'] = real_count

        # ✅ DERIVE summary from entry (single source of truth)
        summary_dict = derive_summary_from_entry(cache_entry)

        # Convert dict to SummaryCache dataclass
        summary = SummaryCache(summary_dict)

        log_info(f"✅ DERIVED summary for {self.dataset_path.name}: "
                      f"total_pairs={summary.img_number.get('num_pairs', 0)}, "
                      f"tagged_manual={summary.img_number.get('tagged_user_to_delete', 0)}, tagged_auto={summary.img_number.get('tagged_auto_to_delete', 0)}, "
                      f"calib_marked={summary.calibration.get('marked', 0)}", "HANDLER")

        return summary

    def mark_dirty(self) -> None:
        """Mark cache dirty and restart debounce timer."""
        from PyQt6.QtCore import QMetaObject, Qt
        log_debug(f"mark_dirty() called for {self.dataset_path.name}, timer will fire in 2s", "HANDLER")
        self._dirty = True
        self._save_status = "pending"

        # Use QMetaObject to safely emit from any thread
        try:
            self.saveStatusChanged.emit("pending")
        except RuntimeError:
            # Signal emission failed (probably from wrong thread), ignore
            pass

        # Use QMetaObject.invokeMethod to start timer from correct thread
        QMetaObject.invokeMethod(
            self._save_timer,
            "start",
            Qt.ConnectionType.QueuedConnection
        )

        if self.workspace_manager:
            self.workspace_manager.notify_dataset_changed(self.dataset_path)

    def _flush_cache(self) -> None:
        """Actually save summary and full cache to disk. Collections are skipped (no own cache)."""
        if not self._dirty:
            log_debug(f"_flush_cache called but not dirty for {self.dataset_path.name}", "HANDLER")
            return

        # Collections don't save their own cache
        if self._is_collection:
            log_debug(f"{self.dataset_path.name} is a COLLECTION - skipping cache save (aggregates from children)", "HANDLER")
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

        # Save summary cache
        if self.summary is not None:
            log_debug(f"Saving .summary_cache.yaml: pairs={self.summary.img_number.get('num_pairs', 0)}, "
                          f"manual={self.summary.img_number.get('tagged_user_to_delete', 0)}, auto={self.summary.img_number.get('tagged_auto_to_delete', 0)}, "
                          f"calib_marked={self.summary.calibration.get('marked', 0)}", "HANDLER")
            self._save_summary()

        # Note: Full cache (.reviewer_cache.yaml) is saved by cache_writer
        # when session.snapshot_cache_payload() is called elsewhere.
        # We only save the summary cache here.

        self._dirty = False
        self._save_status = "saved"
        self.saveStatusChanged.emit("saved")

        log_info(f"✓ Cache flush complete for {self.dataset_path.name}", "HANDLER")

    def _save_summary(self) -> None:
        """Write .summary_cache.yaml to disk."""
        if self.summary is None:
            return

        summary_path = self.dataset_path / SUMMARY_CACHE_FILENAME

        # Collections only save note
        if self._is_collection:
            log_debug(f"Saving collection note only for {self.dataset_path.name}", "HANDLER")
            try:
                data = {
                    "dataset_info": {
                        "note": self.summary.dataset_info.get("note", ""),
                        "last_updated": time.time(),
                    }
                }
                with summary_path.open("w", encoding="utf-8") as f:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            except Exception as e:
                log_error(f"Failed to save collection note: {e}", "HANDLER")
            return
        try:
            # Update timestamp
            self.summary.update_timestamp()

            # Convert to dict and save
            data = asdict(self.summary)

            # Debug: log sweep_flags before saving
            sweep_flags = data.get("dataset_info", {}).get("sweep_flags", {})
            log_debug(f"[SAVE] {self.dataset_path.name} sweep_flags to save: {sweep_flags}", "HANDLER")

            with summary_path.open("w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

            sweep_dups = get_dict_path(self.summary.dataset_info, "sweep_flags.duplicates", False)
            sweep_qual = get_dict_path(self.summary.dataset_info, "sweep_flags.quality", False)
            sweep_pats = get_dict_path(self.summary.dataset_info, "sweep_flags.patterns", False)
            log_debug(f"Saved summary to {summary_path}: "
                        f"total_pairs={self.summary.img_number.get('num_pairs', 0)}, manual={self.summary.img_number.get('tagged_user_to_delete', 0)}, "
                        f"auto={self.summary.img_number.get('tagged_auto_to_delete', 0)}, sweeps=(D:{sweep_dups}, ", "HANDLER"
                    f"Q:{sweep_qual}, P:{sweep_pats})")
        except Exception as e:
            log_error(f"Failed to save summary cache: {e}", "HANDLER")

    def force_flush(self) -> None:
        """Immediate flush (on GUI close, dataset change)."""
        if self._save_timer.isActive():
            self._save_timer.stop()
        self._flush_cache()

    def load_session(self) -> Optional[DatasetSession]:
        """Lazy-load full DatasetSession (heavy operation)."""
        if self.session is not None:
            return self.session

        self.session = DatasetSession()
        if not self.session.load(self.dataset_path):
            self.session = None
            return None

        return self.session

    def update_summary_from_session(self) -> None:
        """Update summary cache from loaded session stats."""
        if self.session is None:
            log_debug(f"update_summary_from_session: No session loaded for {self.dataset_path.name}", "HANDLER")
            return

        log_debug(f"update_summary_from_session: Updating for {self.dataset_path.name}", "HANDLER")

        # Force snapshot even if not dirty - we need the sweep_flags
        # This ensures sweep_flags from session.cache_data propagate to summary
        old_dirty = self.session.cache_dirty
        self.session.cache_dirty = True  # Force snapshot
        payload = self.session.snapshot_cache_payload()
        if not old_dirty:
            # Restore dirty state if it wasn't dirty before
            self.session.cache_dirty = False

        if not payload:
            log_debug("update_summary_from_session: No payload from session", "HANDLER")
            return

        # Rebuild summary from the payload (this has ALL the data including calibration)
        from backend.services.workspace_inspector import _build_info_from_cache
        # Convert payload.dataset_entry to dict for _build_info_from_cache
        cache_dict = dict(payload.dataset_entry) if hasattr(payload, 'dataset_entry') else {}
        info = _build_info_from_cache(cache_dict, self.dataset_path, parent=None)

        if info:
            self.summary = SummaryCache.from_stats(info.stats, note=info.note)
            log_debug(f"update_summary_from_session: ✓ Updated summary: pairs={self.summary.img_number.get('num_pairs', 0)}, "
                          f"calib={self.summary.calibration.get('marked', 0)}/{self.summary.calibration.get('found_both_chessboard', 0)}, "
                          f"manual={self.summary.img_number.get('tagged_user_to_delete', 0)}, auto={self.summary.img_number.get('tagged_auto_to_delete', 0)}", "HANDLER")
        else:
            log_error("update_summary_from_session: Failed to build info from payload", "HANDLER")
