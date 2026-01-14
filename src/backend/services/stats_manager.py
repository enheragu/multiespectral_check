"""Statistics manager for dataset/collection stats.

Centralizes all statistics tracking and formatting logic, providing both
compact and full format output for consistency across UI components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from common.reasons import reason_text


@dataclass
class DatasetStats:
    """Statistics for a single dataset or collection.

    Tracks pairs, deleted items, tagged items (manual/auto), calibration, and outliers.
    Provides compact and full format strings for UI display.
    """

    # Basic counts
    total_pairs: int = 0
    removed_total: int = 0
    tagged_manual: int = 0
    tagged_auto: int = 0

    # Breakdown by reason
    removed_by_reason: Dict[str, int] = field(default_factory=dict)  # Total removed (user+auto combined)
    removed_user_by_reason: Dict[str, int] = field(default_factory=dict)  # User-initiated removals
    removed_auto_by_reason: Dict[str, int] = field(default_factory=dict)  # Auto-detected removals
    tagged_by_reason: Dict[str, int] = field(default_factory=dict)
    tagged_auto_by_reason: Dict[str, int] = field(default_factory=dict)
    pattern_matches: Dict[str, int] = field(default_factory=dict)  # pattern_name -> count

    # Calibration
    calibration_marked: int = 0
    calibration_both: int = 0
    calibration_partial: int = 0
    calibration_missing: int = 0

    # Outliers
    outlier_lwir: int = 0
    outlier_visible: int = 0
    outlier_stereo: int = 0

    # Sweep completion flags (user-initiated operations)
    sweep_duplicates_done: bool = False
    sweep_quality_done: bool = False
    sweep_patterns_done: bool = False

    def __post_init__(self) -> None:
        """Ensure all dict fields are initialized."""
        if self.removed_by_reason is None:
            self.removed_by_reason = {}
        if self.tagged_by_reason is None:
            self.tagged_by_reason = {}
        if self.tagged_auto_by_reason is None:
            self.tagged_auto_by_reason = {}

    @classmethod
    def from_dict(cls, data: dict) -> DatasetStats:
        """Create from cache dict."""
        return cls(
            total_pairs=data.get("total_pairs", 0),
            removed_total=data.get("removed_total", 0),
            tagged_manual=data.get("tagged_manual", 0),
            tagged_auto=data.get("tagged_auto", 0),
            removed_by_reason=data.get("removed_by_reason", {}),
            tagged_by_reason=data.get("tagged_by_reason", {}),
            tagged_auto_by_reason=data.get("tagged_auto_by_reason", {}),
            calibration_marked=data.get("calibration_marked", 0),
            calibration_both=data.get("calibration_both", 0),
            calibration_partial=data.get("calibration_partial", 0),
            calibration_missing=data.get("calibration_missing", 0),
            outlier_lwir=data.get("outlier_lwir", 0),
            outlier_visible=data.get("outlier_visible", 0),
            outlier_stereo=data.get("outlier_stereo", 0),
            sweep_duplicates_done=data.get("sweep_duplicates_done", False),
            sweep_quality_done=data.get("sweep_quality_done", False),
            sweep_patterns_done=data.get("sweep_patterns_done", False),
        )

    def to_dict(self) -> dict:
        """Serialize to cache dict."""
        return {
            "total_pairs": self.total_pairs,
            "removed_total": self.removed_total,
            "tagged_manual": self.tagged_manual,
            "tagged_auto": self.tagged_auto,
            "removed_by_reason": self.removed_by_reason,
            "tagged_by_reason": self.tagged_by_reason,
            "tagged_auto_by_reason": self.tagged_auto_by_reason,
            "calibration_marked": self.calibration_marked,
            "calibration_both": self.calibration_both,
            "calibration_partial": self.calibration_partial,
            "calibration_missing": self.calibration_missing,
            "outlier_lwir": self.outlier_lwir,
            "outlier_visible": self.outlier_visible,
            "outlier_stereo": self.outlier_stereo,
        }

    def merge(self, other: DatasetStats) -> None:
        """Merge another stats object into this one (for aggregating collections)."""
        self.total_pairs += other.total_pairs
        self.removed_total += other.removed_total
        self.tagged_manual += other.tagged_manual
        self.tagged_auto += other.tagged_auto

        for reason, count in other.removed_by_reason.items():
            self.removed_by_reason[reason] = self.removed_by_reason.get(reason, 0) + count
        for reason, count in other.tagged_by_reason.items():
            self.tagged_by_reason[reason] = self.tagged_by_reason.get(reason, 0) + count
        for reason, count in other.tagged_auto_by_reason.items():
            self.tagged_auto_by_reason[reason] = self.tagged_auto_by_reason.get(reason, 0) + count
        for pattern, count in other.pattern_matches.items():
            self.pattern_matches[pattern] = self.pattern_matches.get(pattern, 0) + count

        self.calibration_marked += other.calibration_marked
        self.calibration_both += other.calibration_both
        self.calibration_partial += other.calibration_partial
        self.calibration_missing += other.calibration_missing

        self.outlier_lwir += other.outlier_lwir
        self.outlier_visible += other.outlier_visible
        self.outlier_stereo += other.outlier_stereo

        # Sweep flags: OR logic (if any child has done the sweep, collection has it done)
        self.sweep_duplicates_done = self.sweep_duplicates_done or other.sweep_duplicates_done
        self.sweep_quality_done = self.sweep_quality_done or other.sweep_quality_done
        self.sweep_patterns_done = self.sweep_patterns_done or other.sweep_patterns_done

    # ============================================================================
    # FORMATTING: Removed (deleted) images
    # ============================================================================

    def format_removed_count(self, *, compact: bool = True) -> str:
        """Format removed/deleted count.

        Args:
            compact: If True, return empty string when zero; if False, always show "0".

        Returns:
            Formatted string for removed count column.
        """
        if self.removed_total == 0:
            return "" if compact else "0"
        return str(self.removed_total)

    def format_removed_reasons(self, *, compact: bool = True) -> str:
        """Format breakdown of deleted items by reason.

        Args:
            compact: If True, omit reasons with zero count; if False, show all.

        Returns:
            Formatted string with reason breakdown (e.g., "Blurry: 5, Motion: 3").
        """
        if self.removed_total == 0:
            return "" if compact else "None"

        reasons = self.removed_by_reason or {}
        if not reasons:
            return str(self.removed_total)

        parts = [
            f"{reason_text(k)}: {v}"
            for k, v in reasons.items()
            if not compact or v > 0
        ]
        return ", ".join(parts) if parts else (str(self.removed_total) if compact else "None")

    # ============================================================================
    # FORMATTING: Tagged (marked for deletion) images
    # ============================================================================

    def format_tagged_summary(self, *, compact: bool = True, multiline: bool = True) -> str:
        """Format tagged (marked for deletion) summary with Auto/Manual breakdown.

        Args:
            compact: If True, omit zero counts; if False, show all.
            multiline: If True, separate Manual/Auto on different lines; if False, single line.

        Returns:
            Formatted string with Manual/Auto breakdown and reason details.
        """
        manual_total = self.tagged_manual
        auto_total = self.tagged_auto
        total = manual_total + auto_total

        if total == 0:
            return "" if compact else "None"

        lines = []

        if manual_total > 0 or not compact:
            reason_breakdown = self._format_reason_breakdown(
                self.tagged_by_reason, compact=compact
            )
            lines.append(f"Manual: {manual_total}{reason_breakdown}")

        if auto_total > 0 or not compact:
            reason_breakdown = self._format_reason_breakdown(
                self.tagged_auto_by_reason, compact=compact
            )
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

    # ============================================================================
    # FORMATTING: Calibration
    # ============================================================================

    def format_calibration(self, *, compact: bool = True) -> str:
        """Format calibration statistics.

        Args:
            compact: If True, omit zero counts; if False, show all.

        Returns:
            Formatted string with calibration breakdown (e.g., "5 (both 3; partial 2)").
        """
        total = self.calibration_marked
        if total == 0:
            return "" if compact else "0"

        parts = []
        if self.calibration_both > 0 or not compact:
            parts.append(f"both {self.calibration_both}")
        if self.calibration_partial > 0 or not compact:
            parts.append(f"partial {self.calibration_partial}")
        if self.calibration_missing > 0 or not compact:
            parts.append(f"missing {self.calibration_missing}")

        breakdown = f" ({'; '.join(parts)})" if parts else ""
        return f"Tagged: {total}{breakdown}"

    # ============================================================================
    # FORMATTING: Outliers
    # ============================================================================

    def format_outliers(self, *, compact: bool = True) -> str:
        """Format outlier statistics.

        Args:
            compact: If True, omit zero counts; if False, show all.

        Returns:
            Formatted string with outlier breakdown (e.g., "L 2; V 1; S 0").
        """
        parts = []
        if self.outlier_lwir > 0 or not compact:
            parts.append(f"L {self.outlier_lwir}")
        if self.outlier_visible > 0 or not compact:
            parts.append(f"V {self.outlier_visible}")
        if self.outlier_stereo > 0 or not compact:
            parts.append(f"S {self.outlier_stereo}")

        if not parts:
            return "" if compact else "None"
        return "; ".join(parts)

    # ============================================================================
    # FORMATTING: Comprehensive summary
    # ============================================================================

    def format_summary(self, *, compact: bool = True) -> str:
        """Format complete summary of all statistics.

        Args:
            compact: If True, omit zero counts; if False, show everything.

        Returns:
            Multi-line formatted summary suitable for tooltips or detailed views.
        """
        lines = [
            f"Total pairs: {self.total_pairs}",
        ]

        if self.removed_total > 0 or not compact:
            lines.append(f"Deleted: {self.format_removed_count(compact=False)}")
            reasons = self.format_removed_reasons(compact=compact)
            if reasons and reasons != str(self.removed_total):
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
