"""Mutable state container for per-dataset viewer data."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from PyQt6.QtGui import QPixmap

from backend.utils.duplicates import SignatureCache
from common.dict_helpers import get_dict_path, set_dict_path
from common.log_utils import log_debug, log_error, log_info, log_warning


def _empty_cache_data() -> Dict[str, Any]:
    """Create empty cache data structure matching YAML schema.

    Marks format (unified): marks[base] = {reason: str, auto: bool}
    """
    return {
        "marks": {},  # {base: {reason: str, auto: bool}}
        "reason_counts": {},
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
    # - auto_override → cache_data["overrides"]
    # - calibration_reproj_errors → cache_data["reproj_errors"]
    # - extrinsic_pair_errors → cache_data["extrinsic_errors"]
    # - calibration_matrices → cache_data["_matrices"]
    #
    # UNIFIED MARKS FORMAT (new):
    # - marks[base] = {reason: str, auto: bool}
    # - Use get_mark_reason(), is_mark_auto(), get_auto_marks_for_reason()

    # ===================================================================
    # MARKS DATA HELPERS
    # New unified format: marks[base] = {reason: str, auto: bool}
    # ===================================================================

    def get_mark_reason(self, base: str) -> Optional[str]:
        """Get mark reason for a base, or None if not marked."""
        entry = self.cache_data["marks"].get(base)
        if isinstance(entry, dict):
            return entry.get("reason")
        return None

    def is_mark_auto(self, base: str) -> bool:
        """Check if mark was set automatically."""
        entry = self.cache_data["marks"].get(base)
        if isinstance(entry, dict):
            return bool(entry.get("auto", False))
        return False

    def get_auto_marks_for_reason(self, reason: str) -> Set[str]:
        """Get all bases with auto marks for a given reason."""
        return {
            base for base, entry in self.cache_data["marks"].items()
            if isinstance(entry, dict) and entry.get("reason") == reason and entry.get("auto", False)
        }

    def count_auto_marks_for_reason(self, reason: str) -> int:
        """Count auto marks for a reason (efficient, no set creation)."""
        return sum(
            1 for entry in self.cache_data["marks"].values()
            if isinstance(entry, dict) and entry.get("reason") == reason and entry.get("auto", False)
        )

    # ===================================================================
    # CALIBRATION DATA HELPERS
    # Direct access to cache_data["calibration"] with clear semantics
    # ===================================================================

    def is_calibration_marked(self, base: str) -> bool:
        """Check if base is marked for calibration (presence in dict = marked)."""
        calib_entry = self.cache_data.get("calibration", {}).get(base)
        return isinstance(calib_entry, dict)

    def get_calibration_results(self, base: str) -> Dict[str, Optional[bool]]:
        """Get calibration detection results for a base."""
        calib_entry = self.cache_data.get("calibration", {}).get(base, {})
        return calib_entry.get("results", {}) if isinstance(calib_entry, dict) else {}

    def is_calibration_outlier(self, base: str, channel: str) -> bool:
        """Check if base is outlier for channel (lwir/visible/stereo)."""
        calib_entry = self.cache_data.get("calibration", {}).get(base)
        if not isinstance(calib_entry, dict):
            return False
        outlier_dict = calib_entry.get("outlier", {})
        if isinstance(outlier_dict, dict):
            return bool(outlier_dict.get(channel, False))
        return False

    @property
    def reason_counts(self) -> Dict[str, int]:
        """Reason counts for marked images. Direct access to cache_data."""
        counts = self.cache_data["reason_counts"]
        return counts if isinstance(counts, dict) else {}

    @property
    def calibration_marked(self) -> Set[str]:
        """Bases that are marked for calibration (presence in dict = marked)."""
        calib = self.cache_data.get("calibration", {})
        return {base for base, data in calib.items() if isinstance(data, dict)}

    @property
    def calibration_count_auto(self) -> int:
        """Count of auto-detected calibration images."""
        calib = self.cache_data.get("calibration", {})
        return sum(
            1 for base, data in calib.items()
            if isinstance(data, dict) and data.get("auto", False)
        )

    @property
    def calibration_count_manual(self) -> int:
        """Count of manually-marked calibration images."""
        calib = self.cache_data.get("calibration", {})
        return sum(
            1 for base, data in calib.items()
            if isinstance(data, dict) and not data.get("auto", False)
        )

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
    def calibration_usable_extrinsic_count(self) -> int:
        """Count of images usable for extrinsic calibration (both detected, not stereo outliers)."""
        both_detected = self.cache_data["_detection_counts"].get("both", 0)
        stereo_outliers = len(self.calibration_outliers_extrinsic)
        return max(0, both_detected - stereo_outliers)

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
            outlier_dict = data.get("outlier", {})
            if isinstance(outlier_dict, dict):
                if outlier_dict.get("lwir"):
                    outliers["lwir"].add(base)
                if outlier_dict.get("visible"):
                    outliers["visible"].add(base)
        return outliers

    @calibration_outliers_intrinsic.setter
    def calibration_outliers_intrinsic(self, value: Dict[str, Set[str]]) -> None:
        """Set intrinsic outliers by updating calibration dict."""
        calib = self.cache_data.get("calibration", {})
        # First clear all outlier flags
        for data in calib.values():
            if isinstance(data, dict):
                if "outlier" not in data:
                    data["outlier"] = {"lwir": False, "visible": False, "stereo": False}
                if isinstance(data["outlier"], dict):
                    data["outlier"]["lwir"] = False
                    data["outlier"]["visible"] = False

        # Then set new outliers
        lwir_outliers = value.get("lwir", set())
        vis_outliers = value.get("visible", set())

        for base in lwir_outliers:
            if base not in calib:
                calib[base] = {"auto": False, "outlier": {"lwir": False, "visible": False, "stereo": False}, "results": {}}
            if "outlier" not in calib[base]:
                calib[base]["outlier"] = {"lwir": False, "visible": False, "stereo": False}
            calib[base]["outlier"]["lwir"] = True

        for base in vis_outliers:
            if base not in calib:
                calib[base] = {"auto": False, "outlier": {"lwir": False, "visible": False, "stereo": False}, "results": {}}
            if "outlier" not in calib[base]:
                calib[base]["outlier"] = {"lwir": False, "visible": False, "stereo": False}
            calib[base]["outlier"]["visible"] = True

    @property
    def calibration_outliers_extrinsic(self) -> Set[str]:
        """Extrinsic (stereo) outliers."""
        calib = self.cache_data.get("calibration", {})
        result: Set[str] = set()
        for base, data in calib.items():
            if not isinstance(data, dict):
                continue
            outlier_dict = data.get("outlier", {})
            if isinstance(outlier_dict, dict) and outlier_dict.get("stereo"):
                result.add(base)
        return result

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
                           outlier_stereo: bool = False, *, auto: Optional[bool] = None) -> None:
        """Set or clear calibration mark for a base.

        Args:
            base: Image base name
            marked: Whether to mark or unmark
            outlier_lwir: Mark as lwir outlier
            outlier_visible: Mark as visible outlier
            outlier_stereo: Mark as stereo outlier
            auto: True if marked by auto-detection sweep, False if manual.
                  None preserves existing value (default for manual toggle).
        """
        if marked:
            existing = self.cache_data["calibration"].get(base, {})
            # Preserve existing auto flag unless explicitly overridden
            existing_auto = existing.get("auto", False) if isinstance(existing, dict) else False
            resolved_auto = auto if auto is not None else existing_auto
            # Presence in dict = marked (no explicit 'marked' field needed)
            self.cache_data["calibration"][base] = {
                "auto": resolved_auto,
                "outlier": {
                    "lwir": outlier_lwir,
                    "visible": outlier_visible,
                    "stereo": outlier_stereo,
                },
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
            # Create minimal entry if doesn't exist (presence = marked)
            calib[base] = {"auto": False, "outlier": {"lwir": False, "visible": False, "stereo": False}, "results": {}}
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

    def rebuild_reason_counts(self) -> None:
        """Rebuild reason counts from marks (new unified format)."""
        self.cache_data["reason_counts"].clear()
        for base, entry in self.cache_data["marks"].items():
            if isinstance(entry, dict):
                reason = entry.get("reason")
                if reason:
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
        """Compute manual/auto breakdown using new unified marks format."""
        marks = self.cache_data["marks"]
        reason_counts = self.cache_data["reason_counts"]

        # Count by reason and auto status
        auto_by_reason: Dict[str, int] = {}
        for entry in marks.values():
            if isinstance(entry, dict):
                reason = entry.get("reason", "")
                if entry.get("auto", False) and reason:
                    auto_by_reason[reason] = auto_by_reason.get(reason, 0) + 1

        total_marked = sum(reason_counts.values())
        duplicate_marked = reason_counts.get("duplicate", 0)
        missing_pair_marked = reason_counts.get("missing_pair", 0)
        manual_delete = reason_counts.get("user_marked", 0)
        blurry_marked = reason_counts.get("blurry", 0)
        motion_marked = reason_counts.get("motion", 0)

        auto_blurry = auto_by_reason.get("blurry", 0)
        auto_motion = auto_by_reason.get("motion", 0)
        manual_blurry = max(0, blurry_marked - auto_blurry)
        manual_motion = max(0, motion_marked - auto_motion)
        manual_marked = max(0, total_marked - duplicate_marked - missing_pair_marked - auto_blurry - auto_motion)
        sync_marked = reason_counts.get("sync_error", 0)

        pattern_marked = sum(count for reason, count in reason_counts.items() if isinstance(reason, str) and reason.startswith("pattern"))
        auto_pattern = sum(count for reason, count in auto_by_reason.items() if isinstance(reason, str) and reason.startswith("pattern"))
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
        """Assign or clear a delete reason.

        New unified format: marks[base] = {reason: str, auto: bool}
        """
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

        marks = self.cache_data["marks"]

        if reason is None:
            # Removing mark
            previous_entry = marks.pop(base, None)
            if previous_entry is None:
                return False
            previous_reason = previous_entry.get("reason") if isinstance(previous_entry, dict) else previous_entry
            self._adjust_reason_count(previous_reason, -1)
            if previous_reason != manual_reason:
                self.cache_data["overrides"].add(base)
            if debug:
                log_debug(f"Removed mark from {base} (was: {previous_reason})", "MARK")
            return True

        # Setting mark
        existing_entry = marks.get(base)
        existing_reason = existing_entry.get("reason") if isinstance(existing_entry, dict) else None

        if existing_reason == reason:
            # Same reason, maybe update auto flag
            if isinstance(existing_entry, dict) and existing_entry.get("auto") == auto:
                return False  # No change
            # Update auto flag
            marks[base] = {"reason": reason, "auto": auto}
            if debug:
                log_debug(f"Updated auto flag for {base}: {reason} (auto={auto})", "MARK")
            return True

        if existing_reason:
            self._adjust_reason_count(existing_reason, -1)

        self.cache_data["overrides"].discard(base)
        marks[base] = {"reason": reason, "auto": auto}
        self._adjust_reason_count(reason, 1)

        if debug:
            log_debug(f"Set mark on {base}: {reason} (auto={auto}, existing={existing_reason})", "MARK")
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

        # Clear marks (unified format: {reason, auto})
        self.cache_data["marks"].pop(base, None)

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
