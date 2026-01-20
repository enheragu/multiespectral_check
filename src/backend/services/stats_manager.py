"""Statistics manager for dataset/collection stats.

Centralizes all statistics tracking and formatting logic.
Uses dict-based internal storage - NO properties, NO legacy formats.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from common.dict_helpers import get_dict_path, merge_stats_dicts
from common.reasons import reason_text


def empty_stats_dict() -> Dict[str, Any]:
    """Return empty stats dict with proper structure.

    Structure (simplified - no redundant totals):
        img:
            total: 0
            removed: 0
        tagged:
            user: {}  # reason -> count (no wrapper, no total)
            auto: {}  # reason -> count (includes pattern:xxx)
        removed:
            user: {}  # reason -> count
            auto: {}  # reason -> count
        calibration:
            user: {both: 0, partial: 0, none: 0}  # no total, sum to get it
            auto: {both: 0, partial: 0, none: 0}
            outlier: {lwir: 0, visible: 0, stereo: 0}
        sweep:
            duplicates: False
            quality: False
            patterns: False
    """
    return {
        "img": {
            "total": 0,
            "removed": 0,
        },
        "tagged": {
            "user": {},  # reason -> count directly
            "auto": {},  # reason -> count directly
        },
        "removed": {
            "user": {},  # reason -> count directly
            "auto": {},  # reason -> count directly
        },
        "calibration": {
            "user": {
                "both": 0,
                "partial": 0,
                "none": 0,
            },
            "auto": {
                "both": 0,
                "partial": 0,
                "none": 0,
            },
            "outlier": {
                "lwir": 0,
                "visible": 0,
                "stereo": 0,
            },
        },
        "sweep": {
            "duplicates": False,
            "quality": False,
            "patterns": False,
        },
    }


class DatasetStats:
    """Statistics for a single dataset or collection.

    Uses dict-based internal storage. Access data via:
    - self.data["path"]["to"]["value"] for direct access
    - get_dict_path(self.data, "path.to.value") for safe access

    NO properties - use dict access directly.
    """

    def __init__(self, data: Dict[str, Any] | None = None) -> None:
        """Initialize from dict or empty."""
        if data is None:
            self._data = empty_stats_dict()
        else:
            self._data = deepcopy(data)

    @property
    def data(self) -> Dict[str, Any]:
        """Get internal data dict."""
        return self._data

    def to_dict(self) -> Dict[str, Any]:
        """Return a copy of internal dict for serialization."""
        return deepcopy(self._data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetStats":
        """Create from dict."""
        return cls(data)

    def merge(self, other: "DatasetStats") -> None:
        """Merge another stats object into this one (for aggregating collections)."""
        merge_stats_dicts(self._data, other._data)

    # ============================================================================
    # GETTERS: Convenience accessors for common paths (read-only)
    # These are NOT legacy - they access the new format directly
    # ============================================================================

    def get(self, path: str, default: Any = 0) -> Any:
        """Get value at path with default."""
        return get_dict_path(self._data, path, default)

    # Image counts
    @property
    def total_pairs(self) -> int:
        return self.get("img.total", 0) or 0

    @property
    def removed_total(self) -> int:
        return self.get("img.removed", 0) or 0

    # Tagged counts (calculated from reasons dict)
    @property
    def tagged_manual(self) -> int:
        user_reasons = self.get("tagged.user", {}) or {}
        return sum(user_reasons.values()) if isinstance(user_reasons, dict) else 0

    @property
    def tagged_auto(self) -> int:
        auto_reasons = self.get("tagged.auto", {}) or {}
        return sum(auto_reasons.values()) if isinstance(auto_reasons, dict) else 0

    # Calibration counts (calculated from breakdown)
    @property
    def calibration_marked(self) -> int:
        """Total calibration marked = user + auto (each = both + partial + none)."""
        user = self.get("calibration.user", {}) or {}
        auto = self.get("calibration.auto", {}) or {}
        user_total = sum(user.get(k, 0) for k in ("both", "partial", "none"))
        auto_total = sum(auto.get(k, 0) for k in ("both", "partial", "none"))
        return user_total + auto_total

    @property
    def calibration_both(self) -> int:
        user = self.get("calibration.user.both", 0) or 0
        auto = self.get("calibration.auto.both", 0) or 0
        return user + auto

    # Sweep flags
    @property
    def sweep_duplicates_done(self) -> bool:
        return self.get("sweep.duplicates", False) or False

    @property
    def sweep_quality_done(self) -> bool:
        return self.get("sweep.quality", False) or False

    @property
    def sweep_patterns_done(self) -> bool:
        return self.get("sweep.patterns", False) or False

    # Pattern matches (extracted from tagged.auto with pattern: prefix)
    @property
    def pattern_matches(self) -> Dict[str, int]:
        """Pattern matches extracted from tagged.auto (pattern:xxx keys)."""
        auto_reasons = self.get("tagged.auto", {}) or {}
        if not isinstance(auto_reasons, dict):
            return {}
        return {
            k.replace("pattern:", ""): v
            for k, v in auto_reasons.items()
            if k.startswith("pattern:")
        }

    # ============================================================================
    # FORMATTING: Use get_dict_path directly - NO intermediate properties
    # ============================================================================

    def format_removed_count(self, *, compact: bool = True) -> str:
        """Format removed/deleted count."""
        removed = get_dict_path(self._data, "img.removed", 0, int) or 0
        if removed == 0:
            return "" if compact else "0"
        return str(removed)

    def format_removed_reasons(self, *, compact: bool = True) -> str:
        """Format breakdown of deleted items by reason."""
        removed = get_dict_path(self._data, "img.removed", 0, int) or 0
        if removed == 0:
            return "" if compact else "None"

        # Combine user + auto removed reasons (now directly at removed.user/auto)
        user_reasons = get_dict_path(self._data, "removed.user", {}, dict) or {}
        auto_reasons = get_dict_path(self._data, "removed.auto", {}, dict) or {}
        combined: Dict[str, int] = {}
        for reason, count in user_reasons.items():
            combined[reason] = combined.get(reason, 0) + count
        for reason, count in auto_reasons.items():
            combined[reason] = combined.get(reason, 0) + count

        if not combined:
            return str(removed)

        parts = [
            f"{reason_text(k)}: {v}"
            for k, v in combined.items()
            if not compact or v > 0
        ]
        return ", ".join(parts) if parts else (str(removed) if compact else "None")

    def format_tagged_summary(self, *, compact: bool = True, multiline: bool = True) -> str:
        """Format tagged (marked for deletion) summary with Auto/Manual breakdown."""
        # Totals calculated from reason dicts (no stored total)
        manual_reasons = get_dict_path(self._data, "tagged.user", {}, dict) or {}
        auto_reasons = get_dict_path(self._data, "tagged.auto", {}, dict) or {}
        manual_total = sum(manual_reasons.values()) if isinstance(manual_reasons, dict) else 0
        auto_total = sum(auto_reasons.values()) if isinstance(auto_reasons, dict) else 0
        total = manual_total + auto_total

        if total == 0:
            return "" if compact else "None"

        lines = []

        if manual_total > 0 or not compact:
            reason_breakdown = self._format_reason_breakdown(manual_reasons, compact=compact)
            lines.append(f"Manual: {manual_total}{reason_breakdown}")

        if auto_total > 0 or not compact:
            reason_breakdown = self._format_reason_breakdown(auto_reasons, compact=compact)
            lines.append(f"Auto: {auto_total}{reason_breakdown}")

        if not lines:
            return str(total)

        separator = "\n" if multiline else " | "
        return separator.join(lines)

    def _format_reason_breakdown(self, reasons: Dict[str, int], *, compact: bool) -> str:
        """Format reason dict as ' (reason1: count1, reason2: count2)' or empty string."""
        if not reasons:
            return ""

        parts = [
            f"{reason_text(k)}: {v}"
            for k, v in reasons.items()
            if not compact or v > 0
        ]
        return f" ({', '.join(parts)})" if parts else ""

    def format_calibration(self, *, compact: bool = True) -> str:
        """Format calibration statistics as two lines (manual/auto)."""
        # Totals calculated from breakdown (no stored total)
        user_data = get_dict_path(self._data, "calibration.user", {}, dict) or {}
        auto_data = get_dict_path(self._data, "calibration.auto", {}, dict) or {}

        manual_both = user_data.get("both", 0) or 0
        manual_partial = user_data.get("partial", 0) or 0
        manual_none = user_data.get("none", 0) or 0
        manual_total = manual_both + manual_partial + manual_none

        auto_both = auto_data.get("both", 0) or 0
        auto_partial = auto_data.get("partial", 0) or 0
        auto_none = auto_data.get("none", 0) or 0
        auto_total = auto_both + auto_partial + auto_none

        calib_total = manual_total + auto_total
        if calib_total == 0:
            return "" if compact else "Manual: 0\nAuto: 0"

        lines = []

        # Manual line
        manual_parts = []
        if manual_both > 0 or not compact:
            manual_parts.append(f"both {manual_both}")
        if manual_partial > 0 or not compact:
            manual_parts.append(f"partial {manual_partial}")
        if manual_none > 0 or not compact:
            manual_parts.append(f"none {manual_none}")
        manual_breakdown = f" ({', '.join(manual_parts)})" if manual_parts else ""
        lines.append(f"Manual: {manual_total}{manual_breakdown}")

        # Auto line
        auto_parts = []
        if auto_both > 0 or not compact:
            auto_parts.append(f"both {auto_both}")
        if auto_partial > 0 or not compact:
            auto_parts.append(f"partial {auto_partial}")
        if auto_none > 0 or not compact:
            auto_parts.append(f"none {auto_none}")
        auto_breakdown = f" ({', '.join(auto_parts)})" if auto_parts else ""
        lines.append(f"Auto: {auto_total}{auto_breakdown}")

        return "\n".join(lines)

    def format_outliers(self, *, compact: bool = True) -> str:
        """Format outlier statistics."""
        lwir = get_dict_path(self._data, "calibration.outlier.lwir", 0, int) or 0
        visible = get_dict_path(self._data, "calibration.outlier.visible", 0, int) or 0
        stereo = get_dict_path(self._data, "calibration.outlier.stereo", 0, int) or 0

        parts = []
        if lwir > 0 or not compact:
            parts.append(f"L {lwir}")
        if visible > 0 or not compact:
            parts.append(f"V {visible}")
        if stereo > 0 or not compact:
            parts.append(f"S {stereo}")

        if not parts:
            return "" if compact else "None"
        return "; ".join(parts)

    def format_summary(self, *, compact: bool = True) -> str:
        """Format complete summary of all statistics."""
        total_pairs = get_dict_path(self._data, "img.total", 0, int) or 0
        removed = get_dict_path(self._data, "img.removed", 0, int) or 0

        lines = [f"Total pairs: {total_pairs}"]

        if removed > 0 or not compact:
            lines.append(f"Deleted: {self.format_removed_count(compact=False)}")
            reasons = self.format_removed_reasons(compact=compact)
            if reasons and reasons != str(removed):
                lines.append(f"  {reasons}")

        tagged = self.format_tagged_summary(compact=compact, multiline=True)
        if tagged:
            lines.append(f"Tagged for deletion:\n  {tagged.replace(chr(10), chr(10) + '  ')}")

        calib = self.format_calibration(compact=compact)
        if calib:
            lines.append(f"Calibration: {calib}")

        outliers = self.format_outliers(compact=compact)
        if outliers:
            lines.append(f"Outliers: {outliers}")

        return "\n".join(lines)


__all__ = ["DatasetStats", "empty_stats_dict"]
