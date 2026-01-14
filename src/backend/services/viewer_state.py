"""Mutable state container for per-dataset viewer data."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, List

from PyQt6.QtGui import QPixmap

from backend.utils.duplicates import SignatureCache
from common.dict_helpers import get_dict_path, set_dict_path
from common.log_utils import log_debug, log_info, log_warning, log_error


def _empty_cache_data() -> Dict[str, Any]:
    """Create empty cache data structure matching YAML schema."""
    return {
        "marks": {},
        "reason_counts": {},
        "auto_counts": {},
        "auto_marks": {},
        "calibration": {},
        "reproj_errors": {"lwir": {}, "visible": {}},
        "extrinsic_errors": {},
        "overrides": set(),  # Set[str] in memory, List[str] in YAML
        "archived": {},
        "sweep_flags": {
            "duplicates": False,
            "missing": False,
            "quality": False,
            "patterns": False,
        },
        # Runtime-only fields (not persisted to YAML)
        "_matrices": {"lwir": None, "visible": None},
        "_extrinsic": None,
        "_detection_bins": {},
        "_detection_counts": {"both": 0, "partial": 0, "missing": 0},
    }


@dataclass
class ViewerState:
    """State container with dict as single source of truth.

    ARCHITECTURE CHANGE (Phase 1):
    - cache_data: Dict is the ONLY storage (matches YAML structure)
    - Properties provide typed access for compatibility
    - No more hydrate/serialize conversion (direct dict ↔ YAML)

    Adding new fields: Just add to cache_data, no code changes needed!
    """
    # SINGLE SOURCE OF TRUTH: Dict matching YAML structure
    cache_data: Dict[str, Any] = field(default_factory=_empty_cache_data)

    # Cache only (not persisted)
    pixmap_cache: Dict[str, Dict[str, Optional[QPixmap]]] = field(default_factory=dict)
    signatures: SignatureCache = field(default_factory=dict)

    # Runtime-only data (derived, not persisted)
    missing_counts: Dict[str, int] = field(default_factory=dict)

    # Per-image unified data (future migration target)
    image_data: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # PROPERTIES: Only for complex conversions (Set↔List, derived data)
    # Simple dict access: Use cache_data["key"] directly!
    # Path access: Use get_dict_path(cache_data, "path.to.key")

    # Properties ELIMINADOS - acceso directo:
    # - calibration_detection_bins → cache_data["_detection_bins"]
    # - calibration_detection_counts → cache_data["_detection_counts"]
    # - auto_marks → cache_data["auto_marks"]
    # - auto_override → cache_data["overrides"]
    # - calibration_reproj_errors → cache_data["reproj_errors"]
    # - extrinsic_pair_errors → cache_data["extrinsic_errors"]
    # - calibration_matrices → cache_data["_matrices"]

    # ===================================================================
    # CALIBRATION DATA HELPERS
    # Direct access to cache_data["calibration"] with clear semantics
    # ===================================================================

    def is_calibration_marked(self, base: str) -> bool:
        """Check if base is marked for calibration."""
        calib_entry = self.cache_data.get("calibration", {}).get(base)
        return isinstance(calib_entry, dict) and calib_entry.get("marked", False)

    def get_calibration_results(self, base: str) -> Dict[str, Optional[bool]]:
        """Get calibration detection results for a base."""
        calib_entry = self.cache_data.get("calibration", {}).get(base, {})
        return calib_entry.get("results", {}) if isinstance(calib_entry, dict) else {}

    def is_calibration_outlier(self, base: str, channel: str) -> bool:
        """Check if base is outlier for channel (lwir/visible/stereo)."""
        calib_entry = self.cache_data.get("calibration", {}).get(base)
        if not isinstance(calib_entry, dict):
            return False
        key = f"outlier_{channel}" if channel != "stereo" else "outlier_stereo"
        return bool(calib_entry.get(key, False))

    @property
    def reason_counts(self) -> Dict[str, int]:
        """Reason counts for marked images. Direct access to cache_data."""
        counts = self.cache_data["reason_counts"]
        return counts if isinstance(counts, dict) else {}

    @property
    def calibration_marked(self) -> Set[str]:
        """Bases that are marked for calibration."""
        calib = self.cache_data.get("calibration", {})
        return {base for base, data in calib.items() if isinstance(data, dict) and data.get("marked")}

    @property
    def calibration_results(self) -> Dict[str, Dict[str, Optional[bool]]]:
        """Calibration detection results per base."""
        calib = self.cache_data.get("calibration", {})
        return {
            base: data.get("results", {})
            for base, data in calib.items()
            if isinstance(data, dict) and "results" in data
        }

    @property
    def calibration_corners(self) -> Dict[str, Dict[str, Optional[List[List[float]]]]]:
        """Calibration corners per base (loaded on demand from files)."""
        calib = self.cache_data.get("calibration", {})
        return {
            base: data.get("corners", {})
            for base, data in calib.items()
            if isinstance(data, dict) and "corners" in data
        }

    @property
    def calibration_outliers_intrinsic(self) -> Dict[str, Set[str]]:
        """Intrinsic outliers per channel."""
        calib = self.cache_data.get("calibration", {})
        outliers: Dict[str, Set[str]] = {"lwir": set(), "visible": set()}
        for base, data in calib.items():
            if not isinstance(data, dict):
                continue
            if data.get("outlier_lwir"):
                outliers["lwir"].add(base)
            if data.get("outlier_visible"):
                outliers["visible"].add(base)
        return outliers

    @calibration_outliers_intrinsic.setter
    def calibration_outliers_intrinsic(self, value: Dict[str, Set[str]]) -> None:
        """Set intrinsic outliers by updating calibration dict."""
        # First clear all outlier flags
        calib = self.cache_data.get("calibration", {})
        for data in calib.values():
            if isinstance(data, dict):
                data["outlier_lwir"] = False
                data["outlier_visible"] = False

        # Then set new outliers
        lwir_outliers = value.get("lwir", set())
        vis_outliers = value.get("visible", set())

        for base in lwir_outliers:
            if base not in calib:
                calib[base] = {"marked": False, "outlier_lwir": False, "outlier_visible": False, "outlier_stereo": False, "results": {}}
            calib[base]["outlier_lwir"] = True

        for base in vis_outliers:
            if base not in calib:
                calib[base] = {"marked": False, "outlier_lwir": False, "outlier_visible": False, "outlier_stereo": False, "results": {}}
            calib[base]["outlier_visible"] = True

    @property
    def calibration_outliers_extrinsic(self) -> Set[str]:
        """Extrinsic outliers."""
        calib = self.cache_data.get("calibration", {})
        return {base for base, data in calib.items() if isinstance(data, dict) and data.get("outlier_stereo")}

    # All removed - use cache_data["key"] directly instead
    # Sweep flags removed - use cache_data["sweep_flags"][key] directly

    def reset(self) -> None:
        """Reset state to empty (reinitialize cache_data)."""
        self.cache_data = _empty_cache_data()
        self.pixmap_cache.clear()
        self.signatures.clear()
        self.image_data.clear()

    # Helper methods for calibration manipulation
    def set_calibration_mark(self, base: str, marked: bool = True,
                           outlier_lwir: bool = False, outlier_visible: bool = False,
                           outlier_stereo: bool = False, *, auto: bool = False) -> None:
        """Set or clear calibration mark for a base.

        Args:
            base: Image base name
            marked: Whether to mark or unmark
            outlier_lwir: Mark as lwir outlier
            outlier_visible: Mark as visible outlier
            outlier_stereo: Mark as stereo outlier
            auto: True if marked by auto-detection sweep, False if manual
        """
        if marked:
            existing = self.cache_data["calibration"].get(base, {})
            self.cache_data["calibration"][base] = {
                "marked": True,
                "auto": auto,  # Track if auto-detected or manual
                "outlier_lwir": outlier_lwir,
                "outlier_visible": outlier_visible,
                "outlier_stereo": outlier_stereo,
                "results": existing.get("results", {}),  # Preserve existing results
            }
        else:
            self.cache_data["calibration"].pop(base, None)

    def set_calibration_results(self, base: str, results: Dict[str, Optional[bool]]) -> None:
        """Set calibration detection results for a base.

        Updates the results dict within the calibration entry, creating entry if needed.
        """
        calib = self.cache_data["calibration"]
        if base not in calib:
            # Create minimal entry if doesn't exist
            calib[base] = {"marked": False, "results": {}}
        calib[base]["results"] = dict(results)

    def clear_calibration_results(self, base: str) -> None:
        """Clear calibration results for a base (keeps mark if present)."""
        calib = self.cache_data["calibration"]
        if base in calib:
            calib[base]["results"] = {}

    def is_calibration_auto(self, base: str) -> bool:
        """Check if calibration mark was auto-detected."""
        calib_entry = self.cache_data.get("calibration", {}).get(base)
        return isinstance(calib_entry, dict) and calib_entry.get("auto", False)

    def add_auto_mark(self, reason: str, base: str) -> None:
        """Add an auto mark for a base under a reason."""
        if reason not in self.cache_data["auto_marks"]:
            self.cache_data["auto_marks"][reason] = set()
        self.cache_data["auto_marks"][reason].add(base)

    def remove_auto_mark(self, reason: str, base: str) -> None:
        """Remove an auto mark for a base."""
        if reason in self.cache_data["auto_marks"]:
            self.cache_data["auto_marks"][reason].discard(base)
            if not self.cache_data["auto_marks"][reason]:
                del self.cache_data["auto_marks"][reason]

    def rebuild_reason_counts(self) -> None:
        self.cache_data["reason_counts"].clear()
        # Prune auto marks to existing entries
        marks = self.cache_data["marks"]
        auto_marks = self.cache_data["auto_marks"]
        for reason in list(auto_marks.keys()):
            auto_marks[reason] = {b for b in auto_marks[reason] if marks.get(b) == reason}
            if not auto_marks[reason]:
                auto_marks.pop(reason, None)
        for reason in marks.values():
            self._adjust_reason_count(reason, 1)

    def mark_sweep_done(self, sweep_type: str) -> None:
        """Mark a sweep as completed. Sweep types: duplicates, missing, quality, patterns."""
        if "sweep_flags" not in self.cache_data:
            self.cache_data["sweep_flags"] = {}
        self.cache_data["sweep_flags"][sweep_type] = True

    def is_sweep_done(self, sweep_type: str) -> bool:
        """Check if a sweep has been completed. Sweep types: duplicates, missing, quality, patterns."""
        sweep_flags = self.cache_data.get("sweep_flags", {})
        return bool(sweep_flags.get(sweep_type, False))

    def rebuild_calibration_summary(self) -> None:
        self.cache_data["_detection_bins"].clear()
        self.cache_data["_detection_counts"] = {"both": 0, "partial": 0, "missing": 0}
        for base in self.calibration_marked:
            self._apply_calibration_summary_delta(base)

    def breakdown_marks(self) -> Dict[str, int]:
        """Compute manual/auto breakdown using current counts."""
        reason_counts = self.cache_data["reason_counts"]
        total_marked = sum(reason_counts.values())
        duplicate_marked = reason_counts.get("duplicate", 0)
        missing_pair_marked = reason_counts.get("missing_pair", 0)
        manual_delete = reason_counts.get("user_marked", 0)
        blurry_marked = reason_counts.get("blurry", 0)
        motion_marked = reason_counts.get("motion", 0)
        auto_marks = self.cache_data["auto_marks"]
        auto_blurry = len(auto_marks.get("blurry", set()))
        auto_motion = len(auto_marks.get("motion", set()))
        manual_blurry = max(0, blurry_marked - auto_blurry)
        manual_motion = max(0, motion_marked - auto_motion)
        manual_marked = max(0, total_marked - duplicate_marked - missing_pair_marked - auto_blurry - auto_motion)
        sync_marked = reason_counts.get("sync_error", 0)
        pattern_marked = sum(count for reason, count in reason_counts.items() if isinstance(reason, str) and reason.startswith("pattern"))
        auto_pattern = sum(len(bucket) for reason, bucket in auto_marks.items() if isinstance(reason, str) and reason.startswith("pattern"))
        manual_patterns = max(0, pattern_marked - auto_pattern)
        return {
            "manual_total": manual_marked,
            "manual_delete": manual_delete,
            "manual_blurry": manual_blurry,
            "auto_blurry": auto_blurry,
            "manual_motion": manual_motion,
            "auto_motion": auto_motion,
            "sync": sync_marked,
            "detected_blurry": auto_blurry,
            "detected_motion": auto_motion,
            "detected_duplicates": duplicate_marked,
            "detected_missing": missing_pair_marked,
            "detected_patterns": auto_pattern,
            "manual_patterns": manual_patterns,
        }

    def _adjust_reason_count(self, reason: Optional[str], delta: int) -> None:
        if not reason:
            return
        reason_counts = self.cache_data["reason_counts"]
        next_value = reason_counts.get(reason, 0) + delta
        if next_value <= 0:
            reason_counts.pop(reason, None)
        else:
            reason_counts[reason] = next_value

    def refresh_calibration_entry(self, base: str) -> None:
        if base not in self.calibration_marked:
            self.remove_calibration_entry(base)
            return
        self._apply_calibration_summary_delta(base)

    def remove_calibration_entry(self, base: str) -> None:
        bins = self.cache_data["_detection_bins"]
        counts = self.cache_data["_detection_counts"]
        detected = bins.pop(base, None)
        if detected:
            counts[detected] = max(0, counts.get(detected, 0) - 1)

    def _apply_calibration_summary_delta(self, base: str) -> None:
        bins = self.cache_data["_detection_bins"]
        counts = self.cache_data["_detection_counts"]
        classification = self._classify_calibration_detection(base)
        previous = bins.get(base)
        if previous == classification:
            pass
        else:
            if previous:
                counts[previous] = max(
                    0,
                    counts.get(previous, 0) - 1,
                )
            bins[base] = classification
            counts[classification] = counts.get(classification, 0) + 1

    def _classify_calibration_detection(self, base: str) -> str:
        results = self.calibration_results.get(base, {})
        positives = sum(1 for cam in ("lwir", "visible") if results.get(cam) is True)
        if positives >= 2:
            return "both"
        if positives == 1:
            return "partial"
        return "missing"

    def set_mark_reason(self, base: str, reason: Optional[str], manual_reason: str, *, auto: bool = False) -> bool:
        """Assign or clear a delete reason while managing auto overrides."""
        import os
        debug = os.environ.get("DEBUG_MARKS", "").lower() in {"1", "true", "on"}

        # If auto-marking, check if user has overridden (manually unmarked) this image
        if auto and reason is not None and base in self.cache_data["overrides"]:
            if debug:
                log_debug(f"Skipping AUTO mark for {base} -> {reason} (user override)", "MARK")
            return False

        # Log manual marks explicitly
        if not auto and reason is not None:
            log_info(f"MANUAL mark set: {base} -> {reason} (manual_reason={manual_reason})", "MARK")
        elif auto and reason is not None and debug:
            log_debug(f"AUTO mark set: {base} -> {reason}", "MARK")

        if reason is None:
            marks = self.cache_data["marks"]
            previous = marks.pop(base, None)
            if previous is None:
                return False
            self._adjust_reason_count(previous, -1)
            auto_marks = self.cache_data["auto_marks"]
            if previous in auto_marks:
                auto_marks[previous].discard(base)
                if not auto_marks[previous]:
                    auto_marks.pop(previous, None)
            if previous != manual_reason:
                self.cache_data["overrides"].add(base)
            if debug:
                log_debug(f"Removed mark from {base} (was: {previous})", "MARK")
            return True
        marks = self.cache_data["marks"]
        existing = marks.get(base)
        if existing == reason:
            return False
        if existing:
            self._adjust_reason_count(existing, -1)
            auto_marks = self.cache_data["auto_marks"]
            if existing in auto_marks:
                auto_marks[existing].discard(base)
                if not auto_marks[existing]:
                    auto_marks.pop(existing, None)
        self.cache_data["overrides"].discard(base)
        marks[base] = reason
        self._adjust_reason_count(reason, 1)
        if auto:
            auto_marks = self.cache_data["auto_marks"]
            bucket = auto_marks.setdefault(reason, set())
            bucket.add(base)
            if debug:
                log_debug(f"Added to auto_marks[{reason}]: {base}", "MARK")
        else:
            auto_marks = self.cache_data["auto_marks"]
            if reason in auto_marks:
                auto_marks[reason].discard(base)
                if not auto_marks[reason]:
                    auto_marks.pop(reason, None)
                log_info(f"Removed from auto_marks[{reason}]: {base} (manual mark)", "MARK")
        if debug:
            log_debug(f"Set mark on {base}: {reason} (auto={auto}, existing={existing})", "MARK")
        return True

    # ========================================================================
    # Unified image data helpers (NEW - recommended for new code)
    # ========================================================================

    def get_image_entry(self, base: str) -> Dict[str, Any]:
        """Get unified entry for an image (READ-ONLY reference).

        ⚠️ WARNING: This dict is for READING/ARCHIVING only. To modify state,
        use update_image_entry() or modify cache_data directly.

        Returns dictionary with all data for this image:
        - mark_reason, auto_override
        - calibration_marked, calibration_results
        - outliers (lwir, visible, stereo)
        - reproj_errors, extrinsic_error
        - calibration_corners
        """
        if base not in self.image_data:
            outliers_intrinsic = self.calibration_outliers_intrinsic or {}
            self.image_data[base] = {
                "mark_reason": self.cache_data["marks"].get(base),
                "auto_override": base in self.cache_data["overrides"],
                "calibration_marked": base in self.calibration_marked,
                "outliers": {
                    "lwir": base in outliers_intrinsic.get("lwir", set()),
                    "visible": base in outliers_intrinsic.get("visible", set()),
                    "stereo": base in self.calibration_outliers_extrinsic,
                },
                "calibration_results": self.calibration_results.get(base, {}),
                "reproj_errors": {
                    "lwir": get_dict_path(self.cache_data, f"reproj_errors.lwir.{base}"),
                    "visible": get_dict_path(self.cache_data, f"reproj_errors.visible.{base}"),
                },
                "extrinsic_error": self.cache_data["extrinsic_errors"].get(base),
                "calibration_corners": self.calibration_corners.get(base),
            }

        return self.image_data[base]

    def update_image_entry(self, base: str, **updates: Any) -> None:
        """Update specific fields in image entry (creates if doesn't exist)."""
        entry = self.get_image_entry(base)
        for key, value in updates.items():
            set_dict_path(entry, key, value)

    def clear_image_entry(self, base: str) -> bool:
        """Remove all data for an image.

        Follows data ownership: modifies cache_data directly, not through properties.
        Properties are read-only views, cache_data is the single source of truth.
        """
        self.image_data.pop(base, None)

        # Clear marks
        self.cache_data["marks"].pop(base, None)

        # Clear auto_marks - need to check all reasons
        auto_marks = self.cache_data.get("auto_marks", {})
        for reason, bases in auto_marks.items():
            if isinstance(bases, (list, set)) and base in bases:
                if isinstance(bases, list):
                    bases.remove(base)
                else:
                    bases.discard(base)

        # Clear calibration data - modify the dict directly
        calib = self.cache_data.get("calibration", {})
        if base in calib:
            del calib[base]

        return True

    def toggle_manual_mark(self, base: str, manual_reason: str) -> bool:
        """Toggle the manual deletion flag used by the Delete key."""
        if base in self.cache_data["marks"]:
            return self.set_mark_reason(base, None, manual_reason)
        return self.set_mark_reason(base, manual_reason, manual_reason)

    def clear_markings(self) -> None:
        """Remove all delete marks and auto overrides."""
        self.cache_data["marks"].clear()
        self.cache_data["reason_counts"].clear()
        self.cache_data["overrides"].clear()
        self.cache_data["auto_marks"].clear()
