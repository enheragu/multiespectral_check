"""Encapsulate dataset lifecycle, cache persistence, and destructive actions."""
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, List,
                    Optional, Set, Tuple)

import cv2
import numpy as np
import yaml
from PyQt6.QtGui import QPixmap

from backend.dataset_loader import DatasetLoader
from backend.services.cache_service import (DATASET_CACHE_FILENAME,
                                            CachePersistPayload, CacheService,
                                            deserialize_archived_entries)
from backend.services.calibration_corners_io import (delete_corners,
                                                     load_corners,
                                                     save_corners)
from backend.services.collection import Collection
from backend.services.lru_index import LRUIndex
from backend.services.patterns.pattern_matcher import PatternMatcher
from backend.services.quality.quality_controller import QualityMetrics
from backend.services.viewer_state import ViewerState, _empty_cache_data
from backend.services.workspace_config import get_workspace_config_service
from backend.utils.calibration import undistort_pixmap
from backend.utils.duplicates import (compute_signature_from_path,
                                      get_signature, signature_distance,
                                      store_signature)
from common.log_utils import log_debug, log_info, log_perf, log_warning
from common.reasons import (REASON_BLURRY, REASON_DUPLICATE,
                            REASON_MISSING_PAIR, REASON_MOTION, REASON_PATTERN)
from common.yaml_utils import load_yaml
from config import get_config

if TYPE_CHECKING:
    from backend.services.progress_reporter import ProgressReporter

PIXMAP_CACHE_LIMIT = 24


@dataclass
class DeleteOutcome:
    moved: int
    failed: List[str]
    dataset_available: bool


class DatasetSession:
    def __init__(self) -> None:
        self.loader: Optional[DatasetLoader] = None
        self.collection: Optional[Collection] = None  # For collections
        self.dataset_path: Optional[Path] = None
        self.loaded_kind: Optional[str] = None  # 'dataset' or 'collection'
        self.state = ViewerState()
        self.cache_service = CacheService()
        self.cache_dirty = False
        self._pixmap_cache_order = LRUIndex(PIXMAP_CACHE_LIMIT)
        self._archived_entries: Dict[str, Dict[str, Any]] = {}
        self._dirty_corners: Set[str] = set()  # Track which image bases have modified corners
        self._collection_save_logged = False  # Throttle repetitive collection save logs
        self._sweep_flags: Dict[str, bool] = {
            "missing": False,
            "duplicates": False,
            "quality": False,
            "patterns": False,
        }

    # ------------------------------------------------------------------
    # Dataset lifecycle
    # ------------------------------------------------------------------
    def last_dataset(self) -> Optional[str]:
        return self.cache_service.last_dataset()

    def reset_state(self) -> None:
        self.state.reset()
        self._pixmap_cache_order.clear()
        self.loaded_kind = None

    def load(self, dir_path: Path) -> bool:
        log_debug(f"DatasetSession.load() called: {dir_path}", "SESSION")
        loader = DatasetLoader(str(dir_path))
        if not loader.load_dataset():
            log_warning("DatasetLoader.load_dataset() failed", "SESSION")
            self.loader = None
            self.dataset_path = None
            self.loaded_kind = None
            self.cache_service.set_active_dataset(None)
            self.state.reset()
            self._pixmap_cache_order.clear()
            self._archived_entries.clear()
            return False
        self.loader = loader
        self.dataset_path = dir_path
        self.loaded_kind = "dataset"
        log_info(f"Dataset loaded successfully: {dir_path.name}, pairs={len(loader.image_bases)}", "SESSION")
        self.state.reset()
        self._pixmap_cache_order.clear()
        self.cache_service.set_active_dataset(dir_path)
        self.cache_service.record_dataset_kind(dir_path, "dataset")
        self._collection_save_logged = False
        self._check_and_fix_cache_consistency()
        self._hydrate_from_cache()
        self._load_calibration_files()  # Load calibration matrices from YAML files
        self._validate_bottom_up_consistency()
        # Apply cached sweep flags to state
        self.state.cache_data["sweep_flags"]["missing"] = self._sweep_flags.get("missing", False)
        self.state.cache_data["sweep_flags"]["duplicates"] = self._sweep_flags.get("duplicates", False)
        self.state.cache_data["sweep_flags"]["quality"] = self._sweep_flags.get("quality", False)
        self.state.cache_data["sweep_flags"]["patterns"] = self._sweep_flags.get("patterns", False)
        self._filter_state_by_loader()
        self._auto_mark_missing_pairs()
        self.mark_cache_dirty()
        return True

    def load_collection(self, dir_path: Path) -> bool:
        log_debug(f"DatasetSession.load_collection() called: {dir_path}", "SESSION")

        # Create and initialize Collection
        collection = Collection(dir_path)
        if not collection.discover_children():
            log_warning("Collection.discover_children() failed - no children found", "SESSION")
            self.loader = None
            self.collection = None
            self.dataset_path = None
            self.loaded_kind = None
            self.cache_service.set_active_dataset(None)
            self.state.reset()
            self._pixmap_cache_order.clear()
            self._archived_entries.clear()
            return False

        # Aggregate data from children
        collection.aggregate_from_children()

        # Use collection as loader (has DatasetLoader-compatible interface)
        self.collection = collection
        self.loader = None  # Collection is NOT a DatasetLoader
        self.dataset_path = dir_path
        self.loaded_kind = "collection"

        log_info(f"Collection loaded successfully: {dir_path.name}, pairs={len(collection.image_bases)}", "SESSION")

        # Reset state and populate from collection's aggregated data
        self.state.reset()
        self._pixmap_cache_order.clear()
        self.cache_service.set_active_dataset(dir_path)
        self.cache_service.record_dataset_kind(dir_path, "collection")
        self._collection_save_logged = False

        # Hydrate state from collection's aggregated marks (unified format: base -> {reason, auto})
        self.state.cache_data["marks"] = collection.marks

        self.state.cache_data["signatures"] = collection.signatures
        # Update calibration marked bases, auto flag, and results
        calib_dict = self.state.cache_data.setdefault("calibration", {})
        calib_auto_flags = collection.calibration_auto
        for base in collection.calibration_marked:
            if base not in calib_dict:
                calib_dict[base] = {}
            # Presence in dict = marked (no explicit field needed)
            # Copy auto flag from collection (essential for stats panel)
            calib_dict[base]["auto"] = calib_auto_flags.get(base, False)
        # Also hydrate calibration results (detection status per channel)
        for base, results in collection.calibration_results.items():
            if base not in calib_dict:
                calib_dict[base] = {}
            calib_dict[base]["results"] = dict(results)
        # Also hydrate calibration outliers
        for base, outliers in collection.calibration_outliers.items():
            if base not in calib_dict:
                calib_dict[base] = {}
            calib_dict[base]["outlier"] = dict(outliers)
        self.state.cache_data["sweep_flags"] = collection.sweep_flags

        # Calculate missing counts
        self.state.missing_counts = collection.missing_channel_counts()

        # Rebuild derived counts
        self.state.rebuild_reason_counts()
        self.state.rebuild_calibration_summary()

        # Load calibration files from collection folder (if any)
        self._load_calibration_files()

        # Collections should never save their own cache, so don't mark dirty
        return True

    def total_pairs(self) -> int:
        if self.collection:
            return len(self.collection.image_bases)
        return len(self.loader.image_bases) if self.loader else 0

    def has_images(self) -> bool:
        if self.collection:
            return bool(self.collection.image_bases)
        return bool(self.loader and self.loader.image_bases)

    def get_base(self, index: int) -> Optional[str]:
        image_bases = self.collection.image_bases if self.collection else (self.loader.image_bases if self.loader else [])
        if not image_bases:
            return None
        if index < 0 or index >= len(image_bases):
            return None
        return image_bases[index]

    def get_all_bases(self) -> List[str]:
        """Get all image bases (works for both datasets and collections)."""
        if self.collection:
            return list(self.collection.image_bases)
        if self.loader:
            return list(self.loader.image_bases)
        return []

    def get_loader_for_base(self, base: str) -> Optional["DatasetLoader"]:
        """Get the appropriate loader for a base (works for both datasets and collections)."""
        if self.collection:
            return self.collection.get_loader_for_base(base)
        return self.loader

    def get_image_path(self, base: str, channel: str) -> Optional[Path]:
        """Get image path for a base (works for both datasets and collections)."""
        if self.collection:
            return self.collection.get_image_path(base, channel)
        if self.loader:
            return self.loader.get_image_path(base, channel)
        return None

    def get_original_image_size(self, base: str, channel: str) -> Optional[Tuple[int, int]]:
        """Get the original image size (width, height) from disk without loading full pixmap.

        This returns the size of the image file on disk, before any undistortion
        or other transformations. Useful for transforming normalized coordinates.
        """
        path = self.get_image_path(base, channel)
        if not path or not path.exists():
            return None
        # Use QPixmap to get size without full decode (Qt optimizes this)
        pm = QPixmap(str(path))
        if pm.isNull():
            return None
        return (pm.width(), pm.height())

    def calibration_filter_position(self, current_index: int) -> Tuple[int, int]:
        if not self.loader or not self.loader.image_bases or not self.state.calibration_marked:
            return 0, 0
        filtered_total = 0
        filtered_index = 0
        for idx, base in enumerate(self.loader.image_bases):
            if base not in self.state.calibration_marked:
                continue
            filtered_total += 1
            if idx == current_index:
                filtered_index = filtered_total
        return filtered_index, filtered_total

    def get_metadata_text(self, base: str, type_dir: str) -> str:
        # Get metadata from collection or loader
        if self.collection:
            metadata = self.collection.get_metadata(base, type_dir)
        elif self.loader:
            metadata = self.loader.get_metadata(base, type_dir)
        else:
            return "No metadata found"

        if not metadata:
            return "No metadata found"
        lines = [f"{key}: {value}" for key, value in metadata.items()]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Cache coordination
    # ------------------------------------------------------------------
    def mark_cache_dirty(self) -> None:
        self.cache_dirty = True

    def mark_sweep_done(self, sweep_type: str) -> None:
        """Mark a sweep as completed for this dataset. Sweep types: duplicates, missing, quality, patterns."""
        if "sweep_flags" not in self.state.cache_data:
            self.state.cache_data["sweep_flags"] = {}
        self.state.cache_data["sweep_flags"][sweep_type] = True
        self.mark_cache_dirty()
        log_debug(f"Marked sweep '{sweep_type}' as done: {self.state.cache_data['sweep_flags']}", "SESSION")

    # ------------------------------------------------------------------
    # Sweep operations (dataset owns and executes sweeps on its images)
    # ------------------------------------------------------------------
    def run_pattern_sweep(self, matcher: PatternMatcher, cancel_check: Optional[Callable[[], bool]] = None) -> int:
        """
        Execute pattern matching sweep on dataset images.
        Auto-marks cache dirty and sweep done. Returns match count.

        Args:
            matcher: PatternMatcher instance with loaded patterns
            cancel_check: Optional callable that returns True if operation should be cancelled
        """
        if not self.loader or not self.loader.image_bases:
            return 0

        matches = 0
        for idx, base in enumerate(self.loader.image_bases):
            if cancel_check and cancel_check():
                log_debug(f"Pattern sweep cancelled at {idx}/{len(self.loader.image_bases)}", "SWEEP")
                break
            if not base:
                continue
            existing = self.state.cache_data["marks"].get(base)
            if existing and existing != REASON_PATTERN:
                continue

            vis = self.loader.get_image_path(base, "visible")
            lwir = self.loader.get_image_path(base, "lwir")
            pattern_name = matcher.match_any_paths_detailed([vis, lwir])
            if not pattern_name:
                continue

            # Track which pattern matched by using pattern name in label
            reason_label = f"pattern:{pattern_name}"
            if self.state.set_mark_reason(base, reason_label, REASON_PATTERN, auto=True):
                matches += 1

        if matches:
            self.state.rebuild_reason_counts()
            self.mark_cache_dirty()
        self.mark_sweep_done('patterns')
        return matches

    def run_duplicate_sweep(
        self,
        cancel_check: Optional[Callable[[], bool]] = None,
        reporter: Optional["ProgressReporter"] = None,
    ) -> int:
        """
        Execute duplicate detection sweep on dataset images.
        Auto-marks cache dirty and sweep done. Returns duplicate count.

        Args:
            cancel_check: Optional callable that returns True if operation should be cancelled
            reporter: Optional progress reporter for GUI/terminal feedback
        """
        if not self.loader or not self.loader.image_bases:
            return 0

        total = len(self.loader.image_bases)
        dup_count = 0
        needs_save = False

        # Use reporter's cancel check if available, else use provided one
        should_cancel = (reporter.is_cancelled if reporter else None) or cancel_check

        for idx, base in enumerate(self.loader.image_bases):
            if should_cancel and should_cancel():
                log_debug(f"Duplicate sweep cancelled at {idx}/{total}", "SESSION")
                break

            lwir_sig = compute_signature_from_path(self.loader.get_image_path(base, "lwir"))
            vis_sig = compute_signature_from_path(self.loader.get_image_path(base, "visible"))
            cache_changed, marked = self.apply_signatures(idx, lwir_sig, vis_sig)

            if cache_changed or marked:
                needs_save = True
            if marked:
                dup_count += 1

            if reporter:
                reporter.advance(suffix=f"dups: {dup_count}")

        self.state.rebuild_reason_counts()
        if needs_save:
            self.mark_cache_dirty()
        self.mark_sweep_done('duplicates')
        return dup_count

    def run_quality_sweep(
        self,
        cancel_check: Optional[Callable[[], bool]] = None,
        reporter: Optional["ProgressReporter"] = None,
    ) -> Tuple[int, int]:
        """
        Execute quality/blur/motion detection sweep on dataset images.
        Auto-marks cache dirty and sweep done. Returns (blurry_count, motion_count).

        Args:
            cancel_check: Optional callable that returns True if operation should be cancelled
            reporter: Optional progress reporter for GUI/terminal feedback
        """
        if not self.loader or not self.loader.image_bases:
            return (0, 0)

        total = len(self.loader.image_bases)
        should_cancel = (reporter.is_cancelled if reporter else None) or cancel_check

        records: List[Tuple[str, QualityMetrics, QualityMetrics]] = []
        lap_values: Dict[str, List[float]] = {"lwir": [], "visible": []}
        aniso_values: Dict[str, List[float]] = {"lwir": [], "visible": []}

        # First pass: compute metrics for all images
        for idx, base in enumerate(self.loader.image_bases):
            if should_cancel and should_cancel():
                log_debug(f"Quality sweep cancelled at {idx}/{total} (pass 1)", "SWEEP")
                return (0, 0)

            lwir_metrics = self._compute_quality_metrics(base, "lwir")
            vis_metrics = self._compute_quality_metrics(base, "visible")
            records.append((base, lwir_metrics, vis_metrics))

            if lwir_metrics.laplacian_var is not None:
                lap_values["lwir"].append(lwir_metrics.laplacian_var)
            if vis_metrics.laplacian_var is not None:
                lap_values["visible"].append(vis_metrics.laplacian_var)
            if lwir_metrics.anisotropy is not None:
                aniso_values["lwir"].append(lwir_metrics.anisotropy)
            if vis_metrics.anisotropy is not None:
                aniso_values["visible"].append(vis_metrics.anisotropy)

            if reporter:
                reporter.advance(suffix="analyzing")

        # Compute thresholds
        thresholds = {
            "lap": {
                channel: self._compute_blur_threshold(vals)
                for channel, vals in lap_values.items()
            },
            "aniso": {
                channel: self._compute_aniso_threshold(vals)
                for channel, vals in aniso_values.items()
            },
        }

        # Second pass: apply thresholds and mark images
        blurry_count = 0
        motion_count = 0
        for base, lwir, vis in records:
            reason = None
            for channel, metrics in (("visible", vis), ("lwir", lwir)):
                if not metrics or metrics.laplacian_var is None:
                    continue
                lap = metrics.laplacian_var
                aniso = metrics.anisotropy
                blur_thr = thresholds["lap"].get(channel)
                motion_thr = thresholds["aniso"].get(channel)
                is_motion = motion_thr is not None and aniso is not None and aniso >= motion_thr
                is_blur = blur_thr is not None and lap <= blur_thr
                if is_motion:
                    reason = REASON_MOTION
                    break
                if is_blur and reason is None:
                    reason = REASON_BLURRY

            if not reason:
                continue
            existing = self.state.cache_data["marks"].get(base)
            if existing and existing != reason:
                continue
            if self.state.set_mark_reason(base, reason, reason, auto=True):
                if reason == REASON_MOTION:
                    motion_count += 1
                else:
                    blurry_count += 1

        self.state.rebuild_reason_counts()
        if blurry_count or motion_count:
            self.mark_cache_dirty()
        self.mark_sweep_done('quality')
        return blurry_count, motion_count

    def _compute_quality_metrics(self, base: str, channel: str) -> QualityMetrics:
        """Compute quality metrics for a single image."""
        if not self.loader:
            return QualityMetrics(None, None)

        path = self.loader.get_image_path(base, channel)
        if not path or not path.exists():
            return QualityMetrics(None, None)
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            return QualityMetrics(None, None)

        lap = cv2.Laplacian(img, cv2.CV_64F)
        lap_var = float(lap.var()) if lap.size else None

        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        mean_gx = float(np.mean(np.abs(gx))) if gx.size else 0.0
        mean_gy = float(np.mean(np.abs(gy))) if gy.size else 0.0
        lo = min(mean_gx, mean_gy)
        hi = max(mean_gx, mean_gy)
        ratio = hi / (lo + 1e-6) if hi > 0.0 else None
        return QualityMetrics(lap_var, ratio)

    @staticmethod
    def _compute_blur_threshold(values: List[float]) -> Optional[float]:
        """Compute blur threshold using Q1-1.5*IQR."""
        if not values:
            return None
        arr = np.array(sorted(values))
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        return q1 - 1.5 * iqr if iqr > 0 else None

    @staticmethod
    def _compute_aniso_threshold(values: List[float]) -> Optional[float]:
        """Compute anisotropy (motion) threshold using Q3+1.5*IQR."""
        if not values:
            return None
        arr = np.array(sorted(values))
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        return q3 + 1.5 * iqr if iqr > 0 else None

    def snapshot_cache_payload(self) -> Optional[CachePersistPayload]:
        if not self.cache_dirty:
            return None
        if not self.dataset_path:
            self.cache_dirty = False
            return None

        # Collections don't save their own cache - they distribute to children
        if self.loaded_kind == "collection":
            if not self._collection_save_logged:
                log_debug("Skipping cache save for collection - changes distributed to children", "SESSION")
                self._collection_save_logged = True
            # Corners are stored in child datasets, so flush them even for collections
            self._flush_dirty_corners()
            self.cache_dirty = False
            # For collections, distribute marks to children via Collection class
            if self.collection:
                self.collection.distribute_to_children(
                    self.state.cache_data["marks"],
                    self.state.calibration_marked,
                    self.state.cache_data.get("calibration", {}),  # Include full calibration data
                )
            return None

        # Save modified corners to individual files BEFORE taking snapshot
        self._flush_dirty_corners()

        # Simplified: snapshot_state now takes ViewerState directly
        self.cache_service.snapshot_state(
            self.state,
            self._archived_entries,
            self.total_pairs(),
        )
        payload = self.cache_service.build_persist_payload()
        self.cache_dirty = False
        return payload

    # ------------------------------------------------------------------
    # Calibration corners management (individual files)
    # ------------------------------------------------------------------
    def set_corners(self, base: str, corners: Dict[str, Optional[List[List[float]]]]) -> None:
        """Set corners for an image and mark them as dirty for saving.

        Args:
            base: Image base name
            corners: Dict with corner keys:
                - "lwir": Original LWIR corners
                - "visible": Original visible corners
                - "lwir_subpixel": Subpixel-refined LWIR corners (optional)
                - "visible_subpixel": Subpixel-refined visible corners (optional)
        """
        lwir_count = len(corners.get('lwir', []) or [])
        vis_count = len(corners.get('visible', []) or [])
        lwir_sub = len(corners.get('lwir_subpixel', []) or [])
        vis_sub = len(corners.get('visible_subpixel', []) or [])
        log_debug(
            f"set_corners({base}): LWIR={lwir_count}{f'+{lwir_sub}sub' if lwir_sub else ''}, "
            f"VIS={vis_count}{f'+{vis_sub}sub' if vis_sub else ''}",
            "SESSION"
        )
        # Store directly in cache_data["calibration"] structure
        if "calibration" not in self.state.cache_data:
            self.state.cache_data["calibration"] = {}
        if base not in self.state.cache_data["calibration"]:
            self.state.cache_data["calibration"][base] = {}
        self.state.cache_data["calibration"][base]["corners"] = corners
        self._dirty_corners.add(base)
        self.mark_cache_dirty()

    def get_corners(self, base: str) -> Optional[Dict[str, Optional[List[List[float]]]]]:
        """Get corners for an image (lazy load from disk if needed)."""
        # Check if already in memory (access cache_data["calibration"] directly)
        calib = self.state.cache_data.get("calibration", {})
        if base in calib and "corners" in calib.get(base, {}):
            log_debug(f"get_corners({base}): Found in memory", "SESSION")
            return calib[base]["corners"]

        # Lazy load from disk
        if self.dataset_path:
            log_debug(f"get_corners({base}): Loading from disk...", "SESSION")
            corners = load_corners(self.dataset_path, base)
            if corners is not None:
                # Store directly in cache_data["calibration"] structure
                if "calibration" not in self.state.cache_data:
                    self.state.cache_data["calibration"] = {}
                if base not in self.state.cache_data["calibration"]:
                    self.state.cache_data["calibration"][base] = {}
                self.state.cache_data["calibration"][base]["corners"] = corners
                return corners

        return None

    def delete_corners(self, base: str) -> None:
        """Delete corners for an image (from memory and disk)."""
        log_debug(f"delete_corners({base})", "SESSION")
        # Remove from cache_data["calibration"] structure
        calib = self.state.cache_data.get("calibration", {})
        if base in calib:
            calib[base].pop("corners", None)
        self._dirty_corners.discard(base)
        if self.dataset_path:
            delete_corners(self.dataset_path, base)

    def _flush_dirty_corners(self) -> None:
        """Save all modified corners to individual files."""
        if not self.dataset_path or not self._dirty_corners:
            return

        log_debug(f"_flush_dirty_corners(): Saving {len(self._dirty_corners)} modified corners", "SESSION")
        for base in list(self._dirty_corners):
            # Get corners from cache_data["calibration"] structure
            calib = self.state.cache_data.get("calibration", {})
            corners = calib.get(base, {}).get("corners") if base in calib else None
            if corners is not None:
                # Extract image_sizes if present
                image_sizes = corners.get("image_size")
                save_corners(self.dataset_path, base, corners, image_sizes=image_sizes)

        self._dirty_corners.clear()
        log_debug("_flush_dirty_corners(): Complete", "SESSION")

    # ------------------------------------------------------------------
    # Dangerous operations
    # ------------------------------------------------------------------
    def delete_marked_entries(self, progress_cb: Optional[Callable[[int, int], None]] = None) -> DeleteOutcome:
        start = time.perf_counter()
        if not self.loader or not self.state.cache_data["marks"]:
            return DeleteOutcome(0, [], bool(self.loader and self.loader.image_bases))
        failed: List[str] = []
        moved = 0
        targets = list(self.state.cache_data["marks"].items())
        total = len(targets)
        calib = self.state.cache_data.setdefault("calibration", {})
        for idx, (base, mark_entry) in enumerate(targets, start=1):
            # Check if this mark is auto (unified format)
            if isinstance(mark_entry, dict):
                reason = mark_entry.get("reason", "")
                is_auto = mark_entry.get("auto", False)
            else:
                reason = mark_entry  # Legacy string format
                is_auto = False
            if not self.loader.delete_entry(base, reason, auto=is_auto):
                failed.append(base)
                continue
            self._archive_entry_state(base)
            self.state.cache_data["marks"].pop(base, None)
            self._evict_pixmap_cache_entry(base)
            # Clear calibration by removing from dict (presence = marked)
            calib.pop(base, None)
            for bucket in self.state.cache_data["reproj_errors"].values():
                bucket.pop(base, None)
            self.state.cache_data["extrinsic_errors"].pop(base, None)
            self.delete_corners(base)  # Use new method to delete from disk too
            self.state.remove_calibration_entry(base)
            self.state.cache_data["overrides"].discard(base)
            moved += 1
            if progress_cb:
                progress_cb(idx, total)
        dataset_available = self.loader.load_dataset()
        self._clear_pixmap_cache()
        self.state.signatures = {}
        if dataset_available:
            self._filter_state_by_loader()
            self._auto_mark_missing_pairs()
        else:
            # Clear all calibration entries (presence = marked, so clear dict)
            calib.clear()
            self.state.cache_data["extrinsic_errors"].clear()
            self.state.cache_data["overrides"].clear()
        self.state.rebuild_reason_counts()
        self.state.rebuild_calibration_summary()
        self.mark_cache_dirty()
        log_perf(
            f"delete_marked_entries moved={moved} failed={len(failed)} total={total} in {time.perf_counter()-start:.3f}s"
        )
        return DeleteOutcome(moved, failed, dataset_available)

    def restore_from_trash(self) -> int:
        start = time.perf_counter()
        if not self.loader:
            return 0
        restored_pairs = self.loader.restore_from_trash()
        if restored_pairs == 0:
            return 0
        dataset_available = self.loader.load_dataset()
        self._clear_pixmap_cache()
        if dataset_available:
            self._filter_state_by_loader()
            self._reinstate_archived_entries()
            self._auto_mark_missing_pairs()
        self.state.rebuild_reason_counts()
        self.state.rebuild_calibration_summary()
        self.mark_cache_dirty()
        log_perf(f"restore_from_trash restored={restored_pairs} in {time.perf_counter()-start:.3f}s")
        return restored_pairs

    def reset_dataset_pristine(self) -> bool:
        start = time.perf_counter()
        if not self.loader or not self.dataset_path:
            return False
        dataset_path = Path(self.dataset_path)
        # Skip expensive restore if nothing is in trash.
        try:
            if self.count_trash_pairs() > 0:
                self.restore_from_trash()
        except Exception:
            pass
        # Remove any lingering delete markers and reasons so counts reset to zero.
        to_delete_dir = dataset_path / "to_delete"
        try:
            shutil.rmtree(to_delete_dir)
        except OSError:
            pass
        archive_dir = dataset_path / "archive"
        try:
            shutil.rmtree(archive_dir)
        except OSError:
            pass
        self.cache_service.clear_dataset_cache(dataset_path)

        config = get_config()
        cache_files_to_remove = [
            DATASET_CACHE_FILENAME,  # image_labels.yaml
            config.calibration_intrinsic_filename,  # calibration_intrinsic.yaml
            config.calibration_extrinsic_filename,  # calibration_extrinsic.yaml
            config.summary_cache_filename,  # .summary_cache.yaml
        ]

        for filename in cache_files_to_remove:
            cache_file = dataset_path / filename
            try:
                if cache_file.exists():
                    cache_file.unlink()
            except OSError:
                pass
        # Reuse existing loader to avoid double reload when nothing changed.
        if not self.loader.load_dataset():
            return False
        self.state.reset()
        self._pixmap_cache_order.clear()
        self._archived_entries.clear()
        self.cache_service.set_active_dataset(dataset_path)
        self.cache_service.record_dataset_kind(dataset_path, "dataset")
        self.cache_dirty = False
        self.state.rebuild_reason_counts()
        self.state.rebuild_calibration_summary()
        log_perf(
            f"reset_dataset_pristine path={dataset_path.name} trash_cleared={self.count_trash_pairs()==0} in {time.perf_counter()-start:.3f}s"
        )
        return True

    def count_trash_pairs(self) -> int:
        if not self.loader:
            return 0
        return self.loader.count_trash_pairs()

    def apply_signatures(
        self,
        current_index: int,
        lwir_signature: Optional[bytes],
        vis_signature: Optional[bytes],
    ) -> Tuple[bool, bool]:
        """Persist provided signatures and auto-mark duplicates if needed."""
        if not self.loader or not self.loader.image_bases:
            return False, False
        if current_index < 0 or current_index >= len(self.loader.image_bases):
            return False, False
        base = self.loader.image_bases[current_index]
        bucket = self.state.signatures.setdefault(base, {})
        cache_changed = (
            bucket.get("lwir") != lwir_signature
            or bucket.get("visible") != vis_signature
        )
        store_signature(self.state.signatures, base, "lwir", lwir_signature)
        store_signature(self.state.signatures, base, "visible", vis_signature)
        window = 3
        changed = False
        if current_index <= 0:
            return cache_changed, changed
        if not (lwir_signature and vis_signature):
            return cache_changed, changed
        start = max(0, current_index - window)
        config = get_config()
        for idx in range(current_index - 1, start - 1, -1):
            prev_base = self.loader.image_bases[idx]
            if prev_base == base:
                continue
            prev_lwir = get_signature(self.state.signatures, prev_base, "lwir")
            prev_vis = get_signature(self.state.signatures, prev_base, "visible")
            if not (prev_lwir and prev_vis):
                continue
            threshold = getattr(config, "signature_threshold", 0.005)
            if signature_distance(lwir_signature, prev_lwir) <= threshold and signature_distance(vis_signature, prev_vis) <= threshold:
                self.state.set_mark_reason(base, REASON_DUPLICATE, REASON_DUPLICATE, auto=True)
                self.state.set_mark_reason(prev_base, REASON_DUPLICATE, REASON_DUPLICATE, auto=True)
                changed = True
        return cache_changed, changed

    def _filter_state_by_loader(self) -> None:
        """Drop state entries for bases that are not present in the current loader/collection."""
        image_bases = self.collection.image_bases if self.collection else (self.loader.image_bases if self.loader else [])
        if not image_bases:
            return
        valid = set(image_bases)
        # Remove marks for missing bases (unified format)
        self.state.cache_data["marks"] = {b: entry for b, entry in self.state.cache_data["marks"].items() if b in valid}
        self.state.calibration_marked.intersection_update(valid)
        for mapping in (
            self.state.calibration_results,
            self.state.calibration_corners,
            self.state.cache_data["reproj_errors"].get("lwir", {}),
            self.state.cache_data["reproj_errors"].get("visible", {}),
            self.state.cache_data["extrinsic_errors"],
            self.state.signatures,
        ):
            if isinstance(mapping, dict):
                for base in list(mapping.keys()):
                    if base not in valid:
                        mapping.pop(base, None)
        self.state.rebuild_reason_counts()
        self.state.rebuild_calibration_summary()

    def _auto_mark_missing_pairs(self) -> None:
        # Get channel_map from collection or loader
        if self.collection:
            channel_map = self.collection.channel_map
            missing_counts = self.collection.missing_channel_counts()
        elif self.loader:
            channel_map = getattr(self.loader, 'channel_map', {})
            missing_counts = self.loader.missing_channel_counts() if hasattr(self.loader, 'missing_channel_counts') else {}
        else:
            return

        if missing_counts:
            self.state.missing_counts = missing_counts

        if not channel_map:
            return

        for base, channels in channel_map.items():
            if 'lwir' not in channels or 'visible' not in channels:
                self.state.set_mark_reason(base, REASON_MISSING_PAIR, REASON_MISSING_PAIR, auto=True)
        self.state.rebuild_reason_counts()

    def prepare_display_pair(self, base: str, view_rectified: bool) -> Tuple[Optional[QPixmap], Optional[QPixmap]]:
        """Load and cache LWIR/visible pixmaps; optionally apply rectification.

        Note: We always load fresh pixmaps (no caching) because view modes
        (rectified, aligned) can change the output. The overlay orchestrator
        handles its own caching with proper invalidation.
        """
        def _load_channel(channel: str) -> Optional[QPixmap]:
            # Get image path from collection or loader
            if self.collection:
                img_path = self.collection.get_image_path(base, channel)
            elif self.loader:
                img_path = self.loader.get_image_path(base, channel)
            else:
                return None

            if not img_path or not img_path.exists():
                return None

            pixmap: Optional[QPixmap] = QPixmap(str(img_path))
            if view_rectified:
                matrices = self.state.cache_data["_matrices"].get(channel)
                if matrices:
                    log_debug(f"Applying undistort to {channel}: camera_matrix={type(matrices.get('camera_matrix'))}", "SESSION")
                    pixmap = undistort_pixmap(
                        pixmap,
                        matrices.get("camera_matrix"),
                        matrices.get("distortion"),
                    )
            return pixmap

        lwir_pm = _load_channel("lwir")
        vis_pm = _load_channel("visible")
        return lwir_pm, vis_pm

    def load_raw_pixmaps(self, base: str) -> Tuple[Optional[QPixmap], Optional[QPixmap]]:
        """Load raw (unrectified) pixmaps for calibration analysis.

        Returns:
            Tuple of (lwir_pixmap, visible_pixmap)
        """
        return self.prepare_display_pair(base, view_rectified=False)

    def _get_cached_pixmap(self, base: str, channel: str) -> Optional[Optional[QPixmap]]:
        entry = self.state.pixmap_cache.get(base)
        if entry and channel in entry:
            self._pixmap_cache_order.touch(f"{base}:{channel}")
            return entry[channel]
        return None

    def _set_cached_pixmap(self, base: str, channel: str, pixmap: Optional[QPixmap]) -> None:
        bucket = self.state.pixmap_cache.setdefault(base, {})
        bucket[channel] = pixmap
        key = f"{base}:{channel}"
        evicted = self._pixmap_cache_order.touch(key)
        for dropped in evicted:
            if dropped and ":" in str(dropped):
                b, ch = str(dropped).split(":", 1)
                cached_bucket = self.state.pixmap_cache.get(b)
                if cached_bucket:
                    cached_bucket.pop(ch, None)
                    if not cached_bucket:
                        self.state.pixmap_cache.pop(b, None)

    def _clear_pixmap_cache(self) -> None:
        self.state.pixmap_cache.clear()
        self._pixmap_cache_order.clear()

    def _evict_pixmap_cache_entry(self, base: str) -> None:
        self.state.pixmap_cache.pop(base, None)
        for key in list(self._pixmap_cache_order):
            if str(key).startswith(f"{base}:"):
                self._pixmap_cache_order.remove(key)

    def _check_and_fix_cache_consistency(self) -> None:
        """Verify and repair cache consistency after dataset load.

        Bottom-up validation hierarchy:
        1. calibration/*.yaml (corners per image) - LOWEST LEVEL
        2. image_labels.yaml (labels per image)
        3. .cache.yaml (marks, calibration marks)
        4. .summary_cache.yaml (aggregated stats)
        5. collection cache (if exists) - HIGHEST LEVEL

        Each level must be consistent with the levels below it.
        """
        if not self.dataset_path:
            return

        # === Level 1: Scan calibration/*.yaml for corner files ===
        corners_dir = self.dataset_path / "calibration"
        images_with_corners = set()
        if corners_dir.exists() and corners_dir.is_dir():
            for corner_file in corners_dir.glob("*.yaml"):
                # Extract base from filename (IMG_0001.yaml → IMG_0001)
                base = corner_file.stem
                images_with_corners.add(base)
            log_debug(f"[CONSISTENCY] Found {len(images_with_corners)} images with corner files", "SESSION")

        # === Level 2: Load image_labels.yaml ===
        labels_file = self.dataset_path / "image_labels.yaml"
        images_with_labels = set()
        if labels_file.exists():
            try:
                labels_data = load_yaml(labels_file)
                images_with_labels = set(labels_data.keys())
                log_debug(f"[CONSISTENCY] Found {len(images_with_labels)} images in image_labels.yaml", "SESSION")
            except Exception as e:
                log_warning(f"[CONSISTENCY] Could not load image_labels.yaml: {e}", "SESSION")

        # === Level 3: Cache will be loaded by _hydrate_from_cache() ===
        # We can't check it here because cache_data isn't populated yet.
        # Instead, we'll validate AFTER _hydrate_from_cache() is called.

        # Store discovered images for later validation
        self._bottom_up_corners = images_with_corners
        self._bottom_up_labels = images_with_labels

    def _hydrate_from_cache(self) -> None:
        """Load dataset cache and populate state."""
        if not self.dataset_path:
            return

        # Load cache from dataset-specific file
        cache_entry = self.cache_service.load_dataset_entry()
        if not cache_entry or not isinstance(cache_entry, dict):
            log_debug("No cache entry found for dataset", "SESSION")
            return

        # MERGE cache_entry → state.cache_data (preserving runtime-only fields like _detection_bins)
        # Start with empty structure to ensure all runtime fields exist
        base_data = _empty_cache_data()

        # Merge loaded cache into base structure (cache values override defaults)
        # Special handling for nested dicts like sweep_flags to merge keys
        for key, value in cache_entry.items():
            if key == "sweep_flags" and isinstance(value, dict) and isinstance(base_data.get(key), dict):
                base_data[key].update(value)
            else:
                base_data[key] = value

        # Assign merged structure (marks now in unified format from cache.py normalization)
        self.state.cache_data = base_data

        # Load archived entries
        archived_raw = cache_entry.get("archived", {})
        if isinstance(archived_raw, dict):
            self._archived_entries = deserialize_archived_entries(archived_raw)

        # Rebuild derived summaries
        self.state.rebuild_reason_counts()
        self.state.rebuild_calibration_summary()

        # Count auto marks from unified format
        auto_mark_count = sum(1 for entry in self.state.cache_data["marks"].values()
                              if isinstance(entry, dict) and entry.get("auto", False))
        log_debug(f"Hydrated cache: {len(self.state.cache_data['marks'])} marks ({auto_mark_count} auto), sweep_flags={self.state.cache_data['sweep_flags']['duplicates']}", "SESSION")

    def _load_calibration_files(self) -> None:
        """Load calibration matrices and errors from YAML files.

        Priority for loading calibration:
        1. Dataset's own calibration files
        2. Workspace default calibration (if set)

        This restores calibration data (intrinsic/extrinsic matrices, reprojection errors)
        that was computed in a previous session or inherited from workspace default.
        """
        if not self.dataset_path:
            return

        config = get_config()

        # Determine which calibration files to use
        intrinsic_path = self.dataset_path / config.calibration_intrinsic_filename
        extrinsic_path = self.dataset_path / config.calibration_extrinsic_filename

        # If dataset doesn't have its own calibration, try workspace default
        if not intrinsic_path.exists() and not extrinsic_path.exists():
            ws_config = get_workspace_config_service()
            default_calib = ws_config.get_default_calibration()
            if default_calib:
                if default_calib.intrinsic_path and default_calib.intrinsic_path.exists():
                    intrinsic_path = default_calib.intrinsic_path
                    log_info(f"Using workspace default intrinsic from {default_calib.source_dataset or 'unknown'}", "SESSION")
                if default_calib.extrinsic_path and default_calib.extrinsic_path.exists():
                    extrinsic_path = default_calib.extrinsic_path
                    log_info(f"Using workspace default extrinsic from {default_calib.source_dataset or 'unknown'}", "SESSION")

        # Load intrinsic calibration
        if intrinsic_path.exists():
            try:
                data = load_yaml(intrinsic_path)
                channels = data.get("channels", {})
                if isinstance(channels, dict):
                    for channel, channel_data in channels.items():
                        if isinstance(channel_data, dict):
                            self.state.cache_data["_matrices"][channel] = channel_data
                    log_info(f"Loaded intrinsic calibration from {intrinsic_path.name}: {list(channels.keys())}", "SESSION")
            except (OSError, yaml.YAMLError) as e:
                log_warning(f"Failed to load intrinsic calibration: {e}", "SESSION")

        # Load extrinsic calibration (extrinsic_path already set above, may be from workspace default)
        if extrinsic_path.exists():
            try:
                data = load_yaml(extrinsic_path)
                # Extrinsic data structure may differ - check for common fields
                if isinstance(data, dict):
                    # Store extrinsic matrices in cache_data["_extrinsic"]
                    if "R" in data or "T" in data or "rotation" in data or "translation" in data:
                        self.state.cache_data["_extrinsic"] = data
                    log_info(f"Loaded extrinsic calibration from {extrinsic_path.name}", "SESSION")
            except (OSError, yaml.YAMLError) as e:
                log_warning(f"Failed to load extrinsic calibration: {e}", "SESSION")

        # Load calibration errors from cache file
        errors_path = self.dataset_path / config.calibration_errors_filename
        if errors_path.exists():
            try:
                errors_data = load_yaml(errors_path)
                if isinstance(errors_data, dict):
                    # Load intrinsic per_view_errors
                    for channel in ("lwir", "visible"):
                        channel_errors = errors_data.get("channels", {}).get(channel, {})
                        per_view = channel_errors.get("per_view_errors", {})
                        if isinstance(per_view, dict):
                            for base, err in per_view.items():
                                if isinstance(err, (int, float)):
                                    self.state.cache_data["reproj_errors"].setdefault(channel, {})[base] = float(err)
                    # Load stereo per_pair_errors
                    stereo_errors = errors_data.get("stereo", {})
                    per_pair = stereo_errors.get("per_pair_errors", [])
                    if isinstance(per_pair, list):
                        for entry in per_pair:
                            if isinstance(entry, dict) and isinstance(entry.get("base"), str):
                                trans_err = entry.get("translation_error")
                                if isinstance(trans_err, (int, float)):
                                    self.state.cache_data["extrinsic_errors"][entry["base"]] = float(trans_err)
                    log_info(f"Loaded calibration errors from {errors_path.name}", "SESSION")
            except (OSError, yaml.YAMLError) as e:
                log_warning(f"Failed to load calibration errors: {e}", "SESSION")

    def _validate_bottom_up_consistency(self) -> None:
        """Validate cache consistency with lower-level YAML files.

        Bottom-up validation: ensure cache reflects reality on disk.
        - If corners exist for an image, it should be in calibration dict
        - If labels exist for an image, they should match cache marks

        RESTORATION: If corners exist but not in cache, automatically add them.
        """
        if not self.dataset_path:
            return

        corners_found: Set[str] = getattr(self, '_bottom_up_corners', set())
        labels_found: Set[str] = getattr(self, '_bottom_up_labels', set())

        if not corners_found and not labels_found:
            return

        # Check calibration consistency and RESTORE missing entries
        calibration_dict = self.state.cache_data.get("calibration", {})
        if not isinstance(calibration_dict, dict):
            calibration_dict = {}
            self.state.cache_data["calibration"] = calibration_dict

        # Restore all corner files into calibration dict
        missing_in_cache = corners_found - set(calibration_dict.keys())
        if missing_in_cache:
            log_info(f"[CONSISTENCY] Restoring {len(missing_in_cache)} images with corners to calibration dict", "SESSION")
            for base in missing_in_cache:
                # Add entry to calibration dict with marked=True (since corners exist)
                self.state.set_calibration_mark(base, marked=True)
                log_debug(f"[CONSISTENCY] Restored calibration mark: {base}", "SESSION")

            # Mark cache as dirty so it gets saved
            self.mark_cache_dirty()

        # Verify labels consistency with marks
        marks = self.state.cache_data.get("marks", {})
        for base in labels_found:
            # If image has label, check if it's marked in cache
            if base not in marks:
                log_debug(f"[CONSISTENCY] Image {base} has label but no mark in cache", "SESSION")

        log_info(f"[CONSISTENCY] Bottom-up validation: {len(corners_found)} corners (restored {len(missing_in_cache)}), {len(labels_found)} labels checked", "SESSION")

    def _archive_entry_state(self, base: str) -> None:
        """Archive the state of an entry before deleting it, so it can be restored later."""
        if not base:
            return

        # Use unified image entry (cleaner, no duplication)
        self._archived_entries[base] = self.state.get_image_entry(base).copy()

    def _reinstate_archived_entries(self) -> None:
        """Restore archived entries after images are restored from trash."""
        if not self._archived_entries:
            return

        if not self.loader:
            return

        valid_bases = set(self.loader.image_bases)
        restored_count = 0
        calib = self.state.cache_data.setdefault("calibration", {})

        for base, entry in list(self._archived_entries.items()):
            if base not in valid_bases:
                continue

            # Restore mark reason
            mark_reason = entry.get("mark_reason")
            if mark_reason and isinstance(mark_reason, str):
                self.state.cache_data["marks"][base] = mark_reason

            # Ensure calibration entry exists for this base if restoring calibration_marked
            # Presence in dict = marked, so we only create entry if calibration_marked is true
            if entry.get("calibration_marked"):
                if base not in calib or not isinstance(calib[base], dict):
                    calib[base] = {"auto": False, "outlier": {"lwir": False, "visible": False, "stereo": False}, "results": {}}

            # Restore auto override
            if entry.get("auto_override"):
                self.state.cache_data["overrides"].add(base)

            # Restore outliers - from archived entry's nested format (only if entry exists)
            if base in calib:
                if "outlier" not in calib[base]:
                    calib[base]["outlier"] = {"lwir": False, "visible": False, "stereo": False}
                archived_outliers = entry.get("outliers", {})
                if isinstance(archived_outliers, dict):
                    if archived_outliers.get("lwir"):
                        calib[base]["outlier"]["lwir"] = True
                    if archived_outliers.get("visible"):
                        calib[base]["outlier"]["visible"] = True
                    if archived_outliers.get("stereo"):
                        calib[base]["outlier"]["stereo"] = True

            # Restore calibration results (only if entry exists)
            if base in calib:
                calib_results = entry.get("calibration_results")
                if isinstance(calib_results, dict) and calib_results:
                    calib[base]["results"] = calib_results

            # NOTE: reproj_errors NOT restored from archived - regenerated from calibration file

            # Restore extrinsic error
            extrinsic_err = entry.get("extrinsic_error")
            if isinstance(extrinsic_err, (int, float)):
                self.state.cache_data["extrinsic_errors"][base] = float(extrinsic_err)

            # Restore calibration corners
            calib_corners = entry.get("calibration_corners")
            if isinstance(calib_corners, dict) and calib_corners:
                calib[base]["corners"] = calib_corners

            restored_count += 1
            # Remove from archived list after restoration
            self._archived_entries.pop(base)

        if restored_count > 0:
            log_info(f"Reinstated {restored_count} archived entries", "SESSION")

    def build_outlier_rows(self, bases: Iterable[str]) -> List[Dict[str, Any]]:
        """Build outlier table rows with reprojection errors and inclusion status.

        Args:
            bases: Image base names to include in the outlier table.

        Returns:
            List of dicts with keys:
            - base: image base name
            - lwir, visible, stereo: reprojection errors (float or None)
            - include_lwir, include_visible, include_stereo: inclusion flags (bool)
        """
        reproj_errors = self.state.cache_data.get("reproj_errors", {})
        lwir_errs = reproj_errors.get("lwir", {})
        vis_errs = reproj_errors.get("visible", {})
        stereo_errs = self.state.cache_data.get("extrinsic_errors", {})

        # Get outlier flags per channel from calibration data
        outliers_lwir = self.state.calibration_outliers_intrinsic.get("lwir", set())
        outliers_vis = self.state.calibration_outliers_intrinsic.get("visible", set())
        outliers_stereo = self.state.calibration_outliers_extrinsic

        rows: List[Dict[str, Any]] = []
        for base in bases:
            calib_entry = self.state.cache_data.get("calibration", {}).get(base, {})
            if not isinstance(calib_entry, dict):
                calib_entry = {}

            # Presence in dict = marked (no explicit 'marked' field needed)
            is_marked = base in self.state.cache_data.get("calibration", {})
            is_outlier_lwir = base in outliers_lwir
            is_outlier_vis = base in outliers_vis
            is_outlier_stereo = base in outliers_stereo

            rows.append({
                "base": base,
                "lwir": lwir_errs.get(base) if isinstance(lwir_errs, dict) else None,
                "visible": vis_errs.get(base) if isinstance(vis_errs, dict) else None,
                "stereo": stereo_errs.get(base) if isinstance(stereo_errs, dict) else None,
                "include_lwir": is_marked and not is_outlier_lwir,
                "include_visible": is_marked and not is_outlier_vis,
                "include_stereo": is_marked and not is_outlier_stereo,
            })

        return rows
