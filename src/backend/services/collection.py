"""Collection: Manages a group of datasets with aggregation and distribution."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import time

from backend.dataset_loader import DatasetLoader
from backend.services.cache_service import (
    DATASET_CACHE_FILENAME,
    load_dataset_cache_file,
    save_dataset_cache_file,
)
from backend.services.stats_manager import DatasetStats
from common.log_utils import log_debug, log_info, log_warning, log_error


class Collection:
    """
    Collection encapsulates a group of child datasets.

    RESPONSIBILITIES (SOLID: Single Responsibility):
    - Discover and manage child datasets
    - Aggregate data FROM children (marks, stats, sweep flags)
    - Distribute data TO children (marks made at collection level)
    - Provide unified interface like a dataset

    OWNERSHIP PRINCIPLE:
    - Collection does NOT produce data (no marks, no detections)
    - Collection AGGREGATES data from children
    - Collection COORDINATES operations on children
    - Children OWN their data and persist it
    """

    def __init__(self, collection_path: Path) -> None:
        self.path = collection_path
        self.name = collection_path.name

        # Child management
        self._child_dirs: Dict[str, Path] = {}  # child_key -> path
        self._child_loaders: Dict[str, DatasetLoader] = {}  # child_key -> loader

        # Aggregated state (from children)
        self.image_bases: List[str] = []  # Namespaced: "child/base"
        self.channel_map: Dict[str, Set[str]] = {}  # Namespaced base -> channels
        self.stats = DatasetStats()  # Aggregated statistics

        # Cache for marks (in-memory aggregation from children)
        self._marks: Dict[str, str] = {}  # Namespaced base -> reason
        self._auto_marks: Dict[str, Set[str]] = {}  # reason -> set of namespaced bases
        self._calibration_marked: Set[str] = set()  # Namespaced bases
        self._calibration_results: Dict[str, Dict[str, bool]] = {}  # Namespaced base -> {lwir: bool, visible: bool}
        self._signatures: Dict[str, Dict[str, bytes]] = {}  # Namespaced base -> channel -> signature

        # Sweep flags (aggregated with AND logic: all children must complete)
        self._sweep_flags: Dict[str, bool] = {
            "duplicates": False,
            "quality": False,
            "patterns": False,
        }

    def discover_children(self) -> bool:
        """
        Discover child datasets under collection directory.
        Returns True if at least one child dataset was found.
        """
        start = time.perf_counter()
        self._child_dirs.clear()
        self._child_loaders.clear()
        self.image_bases.clear()
        self.channel_map.clear()

        if not self.path.exists() or not self.path.is_dir():
            return False

        # Scan for dataset directories (recursive)
        dataset_dirs: List[Path] = []
        visited: Set[Path] = set()
        stack = [self.path]

        while stack:
            current = stack.pop()
            if current in visited or current.name == "to_delete":
                continue
            visited.add(current)

            if self._is_dataset_dir(current) and current != self.path:
                dataset_dirs.append(current)

            try:
                children = [p for p in sorted(current.iterdir()) if p.is_dir() and p.name != "to_delete"]
            except OSError:
                continue
            stack.extend(children)

        if not dataset_dirs:
            log_warning(f"Collection {self.name}: No child datasets found", "COLLECTION")
            return False

        # Load each child dataset
        for child_dir in sorted(dataset_dirs):
            loader = DatasetLoader(str(child_dir))
            if not loader.load_dataset():
                log_warning(f"Collection {self.name}: Failed to load child {child_dir.name}", "COLLECTION")
                continue

            child_key = child_dir.relative_to(self.path).as_posix()
            self._child_dirs[child_key] = child_dir
            self._child_loaders[child_key] = loader

            # Namespace image bases with child key
            for base in loader.image_bases:
                namespaced = f"{child_key}/{base}"
                self.image_bases.append(namespaced)
                channels = loader.channel_map.get(base, set())
                self.channel_map[namespaced] = set(channels)

        self.image_bases.sort()
        log_info(f"Collection {self.name}: Discovered {len(self._child_loaders)} children, {len(self.image_bases)} total images", "COLLECTION")

        elapsed = time.perf_counter() - start
        log_debug(f"Collection {self.name}: Discovery completed in {elapsed:.3f}s", "COLLECTION")
        return bool(self.image_bases)

    def aggregate_from_children(self) -> None:
        """
        Aggregate data from all children: marks, stats, sweep flags.
        This is the single source of truth for collection state.
        """
        start = time.perf_counter()

        # Reset aggregated state
        self._marks.clear()
        self._auto_marks.clear()
        self._calibration_marked.clear()
        self._calibration_results.clear()
        self._signatures.clear()
        self.stats = DatasetStats()

        # Track sweep flags from all children (for AND logic)
        children_sweep_flags: Dict[str, List[bool]] = {
            'duplicates': [],
            'quality': [],
            'patterns': [],
        }

        # Aggregate from each child
        for child_key, child_dir in self._child_dirs.items():
            try:
                # Load child's cache
                child_cache_path = child_dir / DATASET_CACHE_FILENAME
                child_cache = load_dataset_cache_file(child_cache_path)

                if not isinstance(child_cache, dict):
                    log_warning(f"Collection {self.name}: Invalid cache for child {child_key}", "COLLECTION")
                    continue

                # Aggregate marks (prefix with child key)
                child_marks = child_cache.get('marks', {})
                if isinstance(child_marks, dict):
                    for base, reason in child_marks.items():
                        prefixed_base = f"{child_key}/{base}"
                        self._marks[prefixed_base] = reason

                # Aggregate auto_marks
                child_auto_marks = child_cache.get('auto_marks', {})
                if isinstance(child_auto_marks, dict):
                    for reason, bases in child_auto_marks.items():
                        # Handle both list (from YAML) and set (from normalized cache)
                        if isinstance(bases, (list, set)):
                            prefixed_bases = {f"{child_key}/{b}" for b in bases}
                            self._auto_marks.setdefault(reason, set()).update(prefixed_bases)

                # Aggregate signatures
                child_sigs = child_cache.get('signatures', {})
                if isinstance(child_sigs, dict):
                    from backend.services.cache_service import deserialize_signatures
                    deserialized_sigs = deserialize_signatures(child_sigs)
                    for base, sig_dict in deserialized_sigs.items():
                        prefixed_base = f"{child_key}/{base}"
                        # Filter out None values
                        filtered_sigs = {k: v for k, v in sig_dict.items() if v is not None}
                        if filtered_sigs:
                            self._signatures[prefixed_base] = filtered_sigs

                # Aggregate calibration_marked and calibration_results
                child_calib = child_cache.get('calibration', {})
                if isinstance(child_calib, dict):
                    for base, calib_entry in child_calib.items():
                        if isinstance(calib_entry, dict):
                            prefixed_base = f"{child_key}/{base}"
                            if calib_entry.get('marked', False):
                                self._calibration_marked.add(prefixed_base)
                            # Also aggregate results (detection status per channel)
                            results = calib_entry.get('results', {})
                            if isinstance(results, dict) and results:
                                self._calibration_results[prefixed_base] = dict(results)

                # Aggregate sweep flags
                child_sweep_flags = child_cache.get('sweep_flags', {})
                if isinstance(child_sweep_flags, dict):
                    children_sweep_flags['duplicates'].append(child_sweep_flags.get('duplicates', False))
                    children_sweep_flags['quality'].append(child_sweep_flags.get('quality', False))
                    children_sweep_flags['patterns'].append(child_sweep_flags.get('patterns', False))

                # Aggregate stats (using DatasetStats.merge)
                child_stats = self._extract_stats_from_cache(child_cache)
                self.stats.merge(child_stats)

            except Exception as e:
                log_error(f"Collection {self.name}: Failed to aggregate from child {child_key}: {e}", "COLLECTION")
                continue

        # Set collection sweep flags: ALL children must have completed (AND logic)
        if children_sweep_flags['duplicates']:
            self._sweep_flags["duplicates"] = all(children_sweep_flags['duplicates'])
        if children_sweep_flags['quality']:
            self._sweep_flags["quality"] = all(children_sweep_flags['quality'])
        if children_sweep_flags['patterns']:
            self._sweep_flags["patterns"] = all(children_sweep_flags['patterns'])

        elapsed = time.perf_counter() - start
        log_info(
            f"Collection {self.name}: Aggregated from {len(self._child_dirs)} children: "
            f"{len(self._marks)} marks, {sum(len(v) for v in self._auto_marks.values())} auto_marks, "
            f"sweep_flags=(D:{self._sweep_flags['duplicates']}, Q:{self._sweep_flags['quality']}, P:{self._sweep_flags['patterns']}) "
            f"in {elapsed:.3f}s",
            "COLLECTION"
        )

    def distribute_to_children(self, marks: Dict[str, str], auto_marks: Dict[str, Set[str]],
                               calibration_marked: Set[str],
                               calibration_data: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        Distribute marks made at collection level back to child datasets.

        When marks are made in collection view, they need to be saved
        to individual child dataset caches.

        Args:
            marks: Dict of base -> reason
            auto_marks: Dict of reason -> set of bases
            calibration_marked: Set of bases marked for calibration
            calibration_data: Optional dict of base -> {marked, auto, results, ...}
        """
        start = time.perf_counter()
        calibration_data = calibration_data or {}

        # Group by child
        child_marks: Dict[str, Dict[str, str]] = {}  # child_key -> {base: reason}
        child_auto_marks: Dict[str, Dict[str, Set[str]]] = {}  # child_key -> {reason: {bases}}
        child_calibration: Dict[str, Dict[str, Dict[str, Any]]] = {}  # child_key -> {base: calib_entry}

        # Process regular marks
        for prefixed_base, reason in marks.items():
            if "/" not in prefixed_base:
                continue
            child_key, local_base = prefixed_base.split("/", 1)
            child_marks.setdefault(child_key, {})[local_base] = reason

        # Process auto_marks
        for reason, prefixed_bases in auto_marks.items():
            for prefixed_base in prefixed_bases:
                if "/" not in prefixed_base:
                    continue
                child_key, local_base = prefixed_base.split("/", 1)
                child_auto_marks.setdefault(child_key, {}).setdefault(reason, set()).add(local_base)

        # Process calibration data (includes marked, auto, results)
        # Iterate over ALL entries in calibration_data, not just calibration_marked
        # This ensures results are persisted even if marked status changes
        all_calib_bases = set(calibration_marked) | set(calibration_data.keys())
        for prefixed_base in all_calib_bases:
            if "/" not in prefixed_base:
                continue
            child_key, local_base = prefixed_base.split("/", 1)
            # Get full calibration entry if available, or create minimal one
            calib_entry = calibration_data.get(prefixed_base, {})
            # Set marked based on whether it's in calibration_marked
            calib_entry["marked"] = prefixed_base in calibration_marked
            child_calibration.setdefault(child_key, {})[local_base] = dict(calib_entry)

        # Save to each child's cache
        updated_children = 0
        for child_key in set(child_marks.keys()) | set(child_auto_marks.keys()) | set(child_calibration.keys()):
            child_dir = self._child_dirs.get(child_key)
            if not child_dir or not child_dir.exists():
                log_warning(f"Collection {self.name}: Child directory not found: {child_key}", "COLLECTION")
                continue

            try:
                # Load existing cache
                child_cache_path = child_dir / DATASET_CACHE_FILENAME
                child_cache = load_dataset_cache_file(child_cache_path)

                # Merge marks
                existing_marks = child_cache.get("marks", {})
                if not isinstance(existing_marks, dict):
                    existing_marks = {}
                marks_to_add = child_marks.get(child_key, {})
                if marks_to_add:
                    existing_marks.update(marks_to_add)
                    child_cache["marks"] = existing_marks

                # Merge auto_marks
                existing_auto = child_cache.get("auto_marks", {})
                if not isinstance(existing_auto, dict):
                    existing_auto = {}
                else:
                    # Deserialize lists to sets
                    existing_auto = {r: set(b) if isinstance(b, list) else b for r, b in existing_auto.items()}

                auto_to_add = child_auto_marks.get(child_key, {})
                if auto_to_add:
                    for reason, bases in auto_to_add.items():
                        existing_auto.setdefault(reason, set()).update(bases)
                    # Serialize back to lists
                    child_cache["auto_marks"] = {r: list(b) for r, b in existing_auto.items()}

                # Merge calibration (now includes full entry with results)
                calibration_to_add = child_calibration.get(child_key, {})
                if calibration_to_add:
                    existing_calib = child_cache.get("calibration", {})
                    if not isinstance(existing_calib, dict):
                        existing_calib = {}

                    for local_base, calib_entry in calibration_to_add.items():
                        if local_base not in existing_calib:
                            existing_calib[local_base] = {}
                        # Merge entry preserving existing data
                        existing_entry = existing_calib[local_base]
                        if not isinstance(existing_entry, dict):
                            existing_entry = {}
                        # Update with new values (marked, auto, results, outliers)
                        existing_entry["marked"] = calib_entry.get("marked", True)
                        if "auto" in calib_entry:
                            existing_entry["auto"] = calib_entry["auto"]
                        if "results" in calib_entry and calib_entry["results"]:
                            existing_entry["results"] = calib_entry["results"]
                        if "outlier_lwir" in calib_entry:
                            existing_entry["outlier_lwir"] = calib_entry["outlier_lwir"]
                        if "outlier_visible" in calib_entry:
                            existing_entry["outlier_visible"] = calib_entry["outlier_visible"]
                        if "outlier_stereo" in calib_entry:
                            existing_entry["outlier_stereo"] = calib_entry["outlier_stereo"]
                        existing_calib[local_base] = existing_entry

                    child_cache["calibration"] = existing_calib

                # Update reason_counts
                all_marks = set(existing_marks.keys())
                reason_counts: Dict[str, int] = {}
                for base in all_marks:
                    reason = existing_marks.get(base, "")
                    if reason:
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
                child_cache["reason_counts"] = reason_counts

                # Save child cache
                save_dataset_cache_file(child_cache_path, child_cache)

                # Notify handler registry
                from backend.services.handler_registry import get_handler_registry
                registry = get_handler_registry()
                handler = registry.get(child_dir)
                if handler:
                    handler.mark_dirty()
                    handler.force_flush()

                updated_children += 1
                mark_count = len(marks_to_add)
                auto_count = sum(len(bases) for bases in auto_to_add.values())
                calib_count = len(calibration_to_add)
                if mark_count or auto_count or calib_count:
                    log_info(f"Collection {self.name}: Distributed to {child_key}: {mark_count} marks, {auto_count} auto_marks, {calib_count} calibration", "COLLECTION")

            except Exception as e:
                log_error(f"Collection {self.name}: Failed to distribute to child {child_key}: {e}", "COLLECTION")
                continue

        elapsed = time.perf_counter() - start
        log_info(f"Collection {self.name}: Distribution complete, updated {updated_children} children in {elapsed:.3f}s", "COLLECTION")

    def get_image_path(self, base: str, channel: str) -> Optional[Path]:
        """Get image path for namespaced base."""
        child_key, child_base = self._split_base(base)
        loader = self._child_loaders.get(child_key)
        if loader:
            return loader.get_image_path(child_base, channel)
        return None

    def get_loader_for_base(self, base: str) -> Optional["DatasetLoader"]:
        """Get the loader for a namespaced base."""
        child_key, _ = self._split_base(base)
        return self._child_loaders.get(child_key)

    def get_metadata(self, base: str, channel: str) -> Optional[Dict]:
        """Get metadata for namespaced base."""
        child_key, child_base = self._split_base(base)
        loader = self._child_loaders.get(child_key)
        if loader:
            return loader.get_metadata(child_base, channel)
        return None

    def delete_entry(self, base: str, reason: str, auto: bool = False) -> bool:
        """Delete entry from appropriate child dataset."""
        child_key, child_base = self._split_base(base)
        loader = self._child_loaders.get(child_key)
        if not loader:
            return False
        ok = loader.delete_entry(child_base, reason, auto=auto)
        if ok:
            # Reload children to reflect changes
            self.discover_children()
        return ok

    def restore_from_trash(self) -> int:
        """Restore all trashed images from all children."""
        restored = 0
        for child_key, loader in self._child_loaders.items():
            child_restored = loader.restore_from_trash()
            restored += child_restored
            if child_restored > 0:
                log_info(f"Collection {self.name}: Restored {child_restored} images from child {child_key}", "COLLECTION")

        if restored:
            self.discover_children()

        return restored

    def count_trash_pairs(self) -> int:
        """Count total trashed images across all children."""
        return sum(loader.count_trash_pairs() for loader in self._child_loaders.values())

    def missing_channel_counts(self) -> Dict[str, int]:
        """Count missing channels across all images."""
        counts = {"lwir": 0, "visible": 0}
        for channels in self.channel_map.values():
            if "lwir" not in channels:
                counts["lwir"] += 1
            if "visible" not in channels:
                counts["visible"] += 1
        return counts

    # Properties for read-only access to aggregated data
    @property
    def marks(self) -> Dict[str, str]:
        """Aggregated marks (namespaced)."""
        return dict(self._marks)

    @property
    def auto_marks(self) -> Dict[str, Set[str]]:
        """Aggregated auto_marks (namespaced)."""
        return {reason: set(bases) for reason, bases in self._auto_marks.items()}

    @property
    def calibration_marked(self) -> Set[str]:
        """Aggregated calibration marks (namespaced)."""
        return set(self._calibration_marked)

    @property
    def calibration_results(self) -> Dict[str, Dict[str, bool]]:
        """Aggregated calibration detection results (namespaced)."""
        return dict(self._calibration_results)

    @property
    def signatures(self) -> Dict[str, Dict[str, bytes]]:
        """Aggregated signatures (namespaced)."""
        return dict(self._signatures)

    @property
    def sweep_flags(self) -> Dict[str, bool]:
        """Aggregated sweep flags (AND logic)."""
        return dict(self._sweep_flags)

    # Helper methods
    @staticmethod
    def _is_dataset_dir(path: Path) -> bool:
        """Check if directory contains lwir/ and visible/ subdirs."""
        return (path / "lwir").is_dir() and (path / "visible").is_dir()

    @staticmethod
    def _split_base(base: str) -> Tuple[str, str]:
        """Split namespaced base into child_key and local base."""
        if "/" in base:
            child_key, child_base = base.rsplit("/", 1)
            return child_key, child_base
        return "", base

    def _extract_stats_from_cache(self, cache: Dict) -> DatasetStats:
        """Extract DatasetStats from cache dict."""
        stats = DatasetStats()

        # Basic counts
        stats.total_pairs = cache.get("total_pairs", 0)
        stats.removed_total = cache.get("removed_total", 0)
        stats.tagged_manual = cache.get("tagged_manual", 0)
        stats.tagged_auto = cache.get("tagged_auto", 0)

        # Reason dictionaries
        stats.removed_by_reason = cache.get("removed_by_reason", {})
        stats.removed_user_by_reason = cache.get("removed_user_by_reason", {})
        stats.removed_auto_by_reason = cache.get("removed_auto_by_reason", {})
        stats.tagged_by_reason = cache.get("tagged_by_reason", {})
        stats.tagged_auto_by_reason = cache.get("tagged_auto_by_reason", {})

        # Calibration counts
        stats.calibration_marked = cache.get("calibration_marked", 0)
        stats.calibration_both = cache.get("calibration_both", 0)
        stats.calibration_partial = cache.get("calibration_partial", 0)
        stats.calibration_missing = cache.get("calibration_missing", 0)
        stats.outlier_lwir = cache.get("outlier_lwir", 0)
        stats.outlier_visible = cache.get("outlier_visible", 0)
        stats.outlier_stereo = cache.get("outlier_stereo", 0)

        # Sweep flags
        sweep_flags = cache.get("sweep_flags", {})
        if isinstance(sweep_flags, dict):
            stats.sweep_duplicates_done = sweep_flags.get("duplicates", False)
            stats.sweep_quality_done = sweep_flags.get("quality", False)
            stats.sweep_patterns_done = sweep_flags.get("patterns", False)

        return stats

    @staticmethod
    def is_collection_dir(path: Path) -> bool:
        """
        Check if path is a collection directory (contains child datasets).
        Static method for use before instantiation.
        """
        if not path.is_dir():
            return False

        stack = [path]
        visited: Set[Path] = set()

        while stack:
            current = stack.pop()
            if current in visited or current.name == "to_delete":
                continue
            visited.add(current)

            if current != path and Collection._is_dataset_dir(current):
                return True

            try:
                children = [p for p in current.iterdir() if p.is_dir()]
            except OSError:
                continue
            stack.extend(children)

        return False
