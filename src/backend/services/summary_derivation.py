"""Summary derivation from dataset cache entry (image_labels.yaml).

This module provides functions to derive .summary_cache.yaml content from the
authoritative .image_labels.yaml file, ensuring data consistency and eliminating
duplication.

Principle: Summary is DERIVED, not independent. It's a cache for fast workspace
scanning, not a source of truth.
"""
import time
from typing import Any, Dict

from backend.services.stats_manager import empty_stats_dict
from backend.utils.cache import DatasetCache
from common.yaml_utils import get_timestamp_fields


def derive_summary_from_entry(entry: DatasetCache) -> Dict[str, Any]:
    """Generate complete summary cache from image_labels entry.

    This is the ONLY way summary should be created. It derives all data from
    the authoritative image_labels.yaml, ensuring consistency.

    Args:
        entry: Dataset cache entry from image_labels.yaml (source of truth)

    Returns:
        Complete summary dict ready for .summary_cache.yaml

    The returned dict has this structure (simplified - no redundant totals):
        dataset_info:
            note: str
            last_updated: float
            last_updated_str: str
        stats:
            img: {total, removed}
            tagged: {user: {reason: count}, auto: {reason: count}}
            removed: {user: {reason: count}, auto: {reason: count}}
            calibration: {user: {both, partial, none}, auto: {...}, outlier: {...}}
            sweep: {duplicates, quality, patterns}
    """
    marks = entry.get("marks", {})
    archived = entry.get("archived", {})
    calibration = entry.get("calibration", {})
    sweep_flags = entry.get("sweep_flags", {})
    total_pairs = entry.get("total_pairs", 0)
    note = entry.get("note", "")

    # Build stats dict using the canonical structure
    stats = empty_stats_dict()

    # =========================================================================
    # Image counts
    # =========================================================================
    stats["img"]["total"] = int(total_pairs)
    stats["img"]["removed"] = len(archived)

    # =========================================================================
    # Tagged marks (user vs auto)
    # =========================================================================
    user_reasons: Dict[str, int] = {}
    auto_reasons: Dict[str, int] = {}

    for base, mark_entry in marks.items():
        if isinstance(mark_entry, dict):
            reason = mark_entry.get("reason", "")
            is_auto = mark_entry.get("auto", False)
        else:
            # Legacy format fallback
            reason = str(mark_entry) if mark_entry else ""
            is_auto = False

        if not reason:
            continue

        if is_auto:
            auto_reasons[reason] = auto_reasons.get(reason, 0) + 1
        else:
            user_reasons[reason] = user_reasons.get(reason, 0) + 1

    # Store reasons directly (no wrapper, no total - calculated at read time)
    stats["tagged"]["user"] = user_reasons
    stats["tagged"]["auto"] = auto_reasons

    # =========================================================================
    # Removed (archived) - separate by user vs auto if available
    # =========================================================================
    removed_user: Dict[str, int] = {}
    removed_auto: Dict[str, int] = {}

    for base, arch_entry in archived.items():
        if isinstance(arch_entry, dict):
            reason = arch_entry.get("mark_reason", "unknown")
            is_auto = arch_entry.get("auto", False)
            if is_auto:
                removed_auto[reason] = removed_auto.get(reason, 0) + 1
            else:
                removed_user[reason] = removed_user.get(reason, 0) + 1

    # Store reasons directly (no wrapper)
    stats["removed"]["user"] = removed_user
    stats["removed"]["auto"] = removed_auto

    # =========================================================================
    # Calibration (user vs auto, with detection breakdown)
    # No totals stored - only breakdown (both/partial/none), totals calculated at read time
    # =========================================================================
    calib_user_both = 0
    calib_user_partial = 0
    calib_user_none = 0

    calib_auto_both = 0
    calib_auto_partial = 0
    calib_auto_none = 0

    outlier_lwir = 0
    outlier_visible = 0
    outlier_stereo = 0

    for base, calib_entry in calibration.items():
        if not isinstance(calib_entry, dict):
            continue

        is_auto = calib_entry.get("auto", False)

        # Detection type from results
        results = calib_entry.get("results", {})
        detection_type = "none"
        if isinstance(results, dict):
            lwir_ok = results.get("lwir") is True
            vis_ok = results.get("visible") is True
            if lwir_ok and vis_ok:
                detection_type = "both"
            elif lwir_ok or vis_ok:
                detection_type = "partial"

        # Count by auto/user (only breakdown, no totals)
        if is_auto:
            if detection_type == "both":
                calib_auto_both += 1
            elif detection_type == "partial":
                calib_auto_partial += 1
            else:
                calib_auto_none += 1
        else:
            if detection_type == "both":
                calib_user_both += 1
            elif detection_type == "partial":
                calib_user_partial += 1
            else:
                calib_user_none += 1

        # Outliers (nested format)
        outlier_dict = calib_entry.get("outlier", {})
        if isinstance(outlier_dict, dict):
            if outlier_dict.get("lwir"):
                outlier_lwir += 1
            if outlier_dict.get("visible"):
                outlier_visible += 1
            if outlier_dict.get("stereo"):
                outlier_stereo += 1

    # No totals stored - calculated at read time from breakdown
    stats["calibration"]["user"]["both"] = calib_user_both
    stats["calibration"]["user"]["partial"] = calib_user_partial
    stats["calibration"]["user"]["none"] = calib_user_none
    stats["calibration"]["auto"]["both"] = calib_auto_both
    stats["calibration"]["auto"]["partial"] = calib_auto_partial
    stats["calibration"]["auto"]["none"] = calib_auto_none
    stats["calibration"]["outlier"]["lwir"] = outlier_lwir
    stats["calibration"]["outlier"]["visible"] = outlier_visible
    stats["calibration"]["outlier"]["stereo"] = outlier_stereo

    # =========================================================================
    # Sweep flags
    # =========================================================================
    stats["sweep"]["duplicates"] = bool(sweep_flags.get("duplicates", False))
    stats["sweep"]["quality"] = bool(sweep_flags.get("quality", False))
    stats["sweep"]["patterns"] = bool(sweep_flags.get("patterns", False))

    # =========================================================================
    # Build final summary structure
    # =========================================================================
    timestamps = get_timestamp_fields()
    summary = {
        "dataset_info": {
            "note": note,
            **timestamps,
        },
        "stats": stats,
    }

    return summary


def update_summary_sweep_flags(summary: Dict[str, Any], sweep_flags: Dict[str, bool]) -> None:
    """Update sweep flags in summary dict (helper for incremental updates).

    Args:
        summary: Summary dict to update (modified in place)
        sweep_flags: Sweep flags from image_labels.yaml
    """
    if "stats" not in summary:
        summary["stats"] = empty_stats_dict()

    summary["stats"]["sweep"]["duplicates"] = bool(sweep_flags.get("duplicates", False))
    summary["stats"]["sweep"]["quality"] = bool(sweep_flags.get("quality", False))
    summary["stats"]["sweep"]["patterns"] = bool(sweep_flags.get("patterns", False))


def update_summary_note(summary: Dict[str, Any], note: str) -> None:
    """Update note in summary dict (helper for incremental updates).

    Args:
        summary: Summary dict to update (modified in place)
        note: Note text from image_labels.yaml
    """
    if "dataset_info" not in summary:
        summary["dataset_info"] = {}
    summary["dataset_info"]["note"] = note
    summary["dataset_info"]["last_updated"] = time.time()


__all__ = [
    "derive_summary_from_entry",
    "update_summary_sweep_flags",
    "update_summary_note",
]
