"""Summary derivation from dataset cache entry (image_labels.yaml).

This module provides functions to derive .summary_cache.yaml content from the
authoritative .image_labels.yaml file, ensuring data consistency and eliminating
duplication.

Principle: Summary is DERIVED, not independent. It's a cache for fast workspace
scanning, not a source of truth.
"""
from typing import Dict, Any, Set
import time

from common.dict_helpers import get_dict_path, set_dict_path
from backend.utils.cache import DatasetCache


def derive_summary_from_entry(entry: DatasetCache) -> Dict[str, Any]:
    """Generate complete summary cache from image_labels entry.

    This is the ONLY way summary should be created. It derives all data from
    the authoritative image_labels.yaml, ensuring consistency.

    Args:
        entry: Dataset cache entry from image_labels.yaml (source of truth)

    Returns:
        Complete summary dict ready for .summary_cache.yaml

    Example:
        >>> entry = load_dataset_cache_file(dataset_path / ".image_labels.yaml")
        >>> summary = derive_summary_from_entry(entry)
        >>> save_summary_cache(dataset_path / ".summary_cache.yaml", summary)
    """
    marks = entry.get("marks", {})
    auto_marks = entry.get("auto_marks", {})
    archived = entry.get("archived", {})
    calibration = entry.get("calibration", {})
    sweep_flags = entry.get("sweep_flags", {})
    total_pairs = entry.get("total_pairs", 0)
    note = entry.get("note", "")

    # Separate user vs auto marks
    user_marks: Dict[str, Set[str]] = {}
    auto_marks_sets: Dict[str, Set[str]] = {}

    # Deserialize auto_marks if needed (might be List[str] from YAML)
    if isinstance(auto_marks, dict):
        for reason, bases in auto_marks.items():
            if isinstance(bases, list):
                auto_marks_sets[reason] = set(bases)
            elif isinstance(bases, set):
                auto_marks_sets[reason] = bases

    # Build auto_marks flat set for quick lookup
    all_auto_bases = set()
    for bases in auto_marks_sets.values():
        all_auto_bases.update(bases)

    # Separate marks into user vs auto
    for base, reason in marks.items():
        if base in all_auto_bases:
            continue  # Skip auto marks
        if reason not in user_marks:
            user_marks[reason] = set()
        user_marks[reason].add(base)

    # Count marks by type
    user_mark_count = sum(len(bases) for bases in user_marks.values())
    auto_mark_count = sum(len(bases) for bases in auto_marks_sets.values())
    archived_count = len(archived)

    # Count removed (archived images)
    removed_count = archived_count

    # Build reason counts
    removed_reasons: Dict[str, int] = {}
    removed_user_reasons: Dict[str, int] = {}
    removed_auto_reasons: Dict[str, int] = {}

    for base, arch_entry in archived.items():
        if isinstance(arch_entry, dict):
            reason = arch_entry.get("mark_reason")
            if reason:
                removed_reasons[reason] = removed_reasons.get(reason, 0) + 1

    tagged_user_to_delete_reasons: Dict[str, int] = {
        reason: len(bases) for reason, bases in user_marks.items()
    }

    tagged_auto_to_delete_reasons: Dict[str, int] = {
        reason: len(bases) for reason, bases in auto_marks_sets.items()
    }

    # Calibration stats (derive from calibration dict)
    calib_marked = 0
    found_both = 0
    found_only_lwir = 0
    found_only_visible = 0
    found_none = 0
    outlier_lwir = 0
    outlier_visible = 0
    outlier_stereo = 0

    for base, calib_entry in calibration.items():
        if not isinstance(calib_entry, dict):
            continue

        if calib_entry.get("marked"):
            calib_marked += 1

        results = calib_entry.get("results", {})
        if isinstance(results, dict):
            lwir_detected = results.get("lwir")
            vis_detected = results.get("visible")

            if lwir_detected and vis_detected:
                found_both += 1
            elif lwir_detected:
                found_only_lwir += 1
            elif vis_detected:
                found_only_visible += 1
            elif lwir_detected is False and vis_detected is False:
                found_none += 1

        if calib_entry.get("outlier_lwir"):
            outlier_lwir += 1
        if calib_entry.get("outlier_visible"):
            outlier_visible += 1
        if calib_entry.get("outlier_stereo"):
            outlier_stereo += 1

    # Pattern matches (future feature, empty for now)
    pattern_matches: Dict[str, int] = {}

    # Build summary structure
    summary = {
        "dataset_info": {
            "note": note,
            "last_updated": time.time(),
            "sweep_flags": {
                "missing": bool(sweep_flags.get("missing", False)),
                "duplicates": bool(sweep_flags.get("duplicates", False)),
                "patterns": bool(sweep_flags.get("patterns", False)),
                "quality": bool(sweep_flags.get("quality", False)),
            },
        },
        "img_number": {
            "num_pairs": int(total_pairs),
            "removed_pairs": removed_count,
            "tagged_user_to_delete": user_mark_count,
            "tagged_auto_to_delete": auto_mark_count,
        },
        "removed_reasons": removed_reasons,
        "removed_user_reasons": removed_user_reasons,
        "removed_auto_reasons": removed_auto_reasons,
        "tagged_user_to_delete_reasons": tagged_user_to_delete_reasons,
        "tagged_auto_to_delete_reasons": tagged_auto_to_delete_reasons,
        "calibration": {
            "marked": calib_marked,
            "found_both_chessboard": found_both,
            "found_only_lwir_chessboard": found_only_lwir,
            "found_only_visible_chessboard": found_only_visible,
            "found_none_chessboard": found_none,
            "outlier_discarded_lwir": outlier_lwir,
            "outlier_discarded_visible": outlier_visible,
            "outlier_discarded_stereo": outlier_stereo,
        },
        "pattern_matches": pattern_matches,
    }

    return summary


def update_summary_sweep_flags(summary: Dict[str, Any], sweep_flags: Dict[str, bool]) -> None:
    """Update sweep flags in summary dict (helper for incremental updates).

    Args:
        summary: Summary dict to update (modified in place)
        sweep_flags: Sweep flags from image_labels.yaml
    """
    set_dict_path(summary, "dataset_info.sweep_flags", {
        "missing": bool(sweep_flags.get("missing", False)),
        "duplicates": bool(sweep_flags.get("duplicates", False)),
        "patterns": bool(sweep_flags.get("patterns", False)),
        "quality": bool(sweep_flags.get("quality", False)),
    })


def update_summary_note(summary: Dict[str, Any], note: str) -> None:
    """Update note in summary dict (helper for incremental updates).

    Args:
        summary: Summary dict to update (modified in place)
        note: Note text from image_labels.yaml
    """
    set_dict_path(summary, "dataset_info.note", note)
    set_dict_path(summary, "dataset_info.last_updated", time.time())


__all__ = [
    "derive_summary_from_entry",
    "update_summary_sweep_flags",
    "update_summary_note",
]
