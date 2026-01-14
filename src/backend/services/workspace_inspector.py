"""Helpers to inspect workspace datasets and persist per-dataset notes."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set, TYPE_CHECKING
import time
from concurrent.futures import ThreadPoolExecutor

import yaml


if TYPE_CHECKING:
    from backend.services.thread_pool_manager import ThreadPoolManager

try:
    from backend.services.thread_pool_manager import get_thread_pool_manager
except ImportError:
    get_thread_pool_manager = None  # type: ignore

from backend.services.cache_service import (
    DATASET_CACHE_FILENAME,
    empty_dataset_entry,
    load_dataset_cache_file,
    save_dataset_cache_file,
    deserialize_calibration,
)
from backend.services.summary_derivation import derive_summary_from_entry
from common.dict_helpers import get_dict_path
from backend.utils.cache import DatasetCache
from common.reasons import AUTO_REASONS
from backend.services.stats_manager import DatasetStats
from common.log_utils import log_debug, log_perf, log_debug



WORKSPACE_INDEX_FILENAME = ".workspace_index.json"

# Bump this version whenever workspace aggregation logic changes
# to force cache regeneration
WORKSPACE_LOGIC_VERSION = 3


def _normalize_int_dict(data: object) -> Dict[str, int]:
    """Convert dict values to int, handling None/empty gracefully."""
    if not isinstance(data, dict):
        return {}
    return {k: int(v) for k, v in data.items()}


def _merge_count_dicts(target: Dict[str, int], source: Dict[str, int]) -> None:
    """Merge source counts into target dict in-place."""
    for key, val in source.items():
        target[key] = target.get(key, 0) + val


@dataclass
class WorkspaceDatasetInfo:
    """Information about a dataset or collection in the workspace."""
    name: str
    path: Path
    note: str = ""
    is_collection: bool = False
    children: List["WorkspaceDatasetInfo"] = field(default_factory=list)
    parent: Optional[str] = None
    stats: DatasetStats = field(default_factory=DatasetStats)


def scan_workspace(workspace_dir: Path) -> List[WorkspaceDatasetInfo]:
    if not workspace_dir.exists() or not workspace_dir.is_dir():
        return []
    cached = _workspace_cache.get(workspace_dir)
    sig = _workspace_signature(workspace_dir)
    index = _load_workspace_index(workspace_dir)
    if cached and cached.get("sig") == sig:
        log_perf(f"scan_workspace cache hit: {workspace_dir}", "perf:workspace")
        entries_obj = cached.get("entries", [])
        if isinstance(entries_obj, list):
            return entries_obj
        return []
    start = time.perf_counter()
    entries: List[WorkspaceDatasetInfo] = []
    dataset_infos: Dict[str, WorkspaceDatasetInfo] = {}
    new_index: Dict[str, Dict[str, object]] = {}
    futures: List[Tuple] = []

    dataset_meta: List[Tuple[Path, str, Optional[str]]] = []
    for entry in sorted(p for p in workspace_dir.iterdir() if p.is_dir()):
        # First check if this directory has dataset children (is a collection)
        child_dirs = [p for p in entry.iterdir() if p.is_dir() and _is_dataset_dir(p)]
        if child_dirs:
            # This is a collection - add all children with parent hint
            for child in sorted(child_dirs):
                dataset_meta.append((child, str(child.relative_to(workspace_dir)), entry.name))
                log_debug(f"Added child {child.name} with parent={entry.name}", "WORKSPACE_MGR")
        elif _is_dataset_dir(entry):
            # Standalone dataset (no children)
            dataset_meta.append((entry, str(entry.relative_to(workspace_dir)), None))
            log_debug(f"Added standalone {entry.name} with parent=None", "WORKSPACE_MGR")

    prev_datasets = index.get("datasets", {}) if isinstance(index, dict) else {}

    # Use ThreadPoolManager for coordinated resource usage if available
    if get_thread_pool_manager is not None:
        pool_manager = get_thread_pool_manager()
        max_workers = pool_manager.limits.workspace_scan
    else:
        max_workers = _scan_workers()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for path, rel, parent in dataset_meta:
            sig_rel = _quick_signature(path)
            cached_summary = prev_datasets.get(rel) if isinstance(prev_datasets, dict) else None
            cached_sig = cached_summary.get("sig") if isinstance(cached_summary, dict) else None
            summary = cached_summary.get("summary") if isinstance(cached_summary, dict) else None
            if cached_sig and abs(float(cached_sig) - sig_rel) < 1e-6 and isinstance(summary, dict):
                info = _summary_to_info(summary, workspace_dir, parent)
                if info:
                    dataset_infos[rel] = info
                    new_index[rel] = {"sig": sig_rel, "summary": _info_to_summary(info, workspace_dir)}
                    continue
            futures.append((executor.submit(collect_dataset_info, path, parent), rel, sig_rel))

        for future, rel, sig_rel in futures:
            info = future.result()
            if not info:
                continue
            dataset_infos[rel] = info
            new_index[rel] = {"sig": sig_rel, "summary": _info_to_summary(info, workspace_dir)}

    # Build collections from dataset infos
    collections: Dict[str, List[WorkspaceDatasetInfo]] = {}
    standalone_datasets: List[WorkspaceDatasetInfo] = []

    log_debug(f"Building collections dict from {len(dataset_infos)} dataset_infos", "WORKSPACE_MGR")

    for info in dataset_infos.values():
        if info.parent:
            collections.setdefault(info.parent, []).append(info)
            log_debug(f"Added {info.name} to collection {info.parent}", "WORKSPACE_MGR")
        elif not info.is_collection:
            # Dataset without parent = standalone
            standalone_datasets.append(info)
            log_debug(f"Added {info.name} as standalone", "WORKSPACE_MGR")

    # Add entries in order: collections+children, then standalone datasets
    log_debug(f"Processing {len(collections)} collections", "WORKSPACE_MGR")
    for coll_name, coll_children in collections.items():
        log_debug(f"Collection {coll_name} has {len(coll_children)} children", "WORKSPACE_MGR")

    for entry in sorted(p for p in workspace_dir.iterdir() if p.is_dir()):
        # Check if this directory has children (is a collection)
        children = collections.get(entry.name, [])
        if children:
            # This is a collection - aggregate and add with children
            log_debug(f"Creating collection entry for {entry.name} with {len(children)} children", "WORKSPACE_MGR")
            collection_info = _aggregate_collection(entry, children)
            entries.append(collection_info)
            entries.extend(children)
            # Save collection summary to index so it can be restored
            collection_rel = str(entry.relative_to(workspace_dir))
            new_index[collection_rel] = {
                "sig": _quick_signature(entry),
                "summary": _info_to_summary(collection_info, workspace_dir)
            }
            continue
        # If not a collection and not a dataset dir, skip
        if not _is_dataset_dir(entry):
            continue

    # Add standalone datasets at the end
    entries.extend(sorted(standalone_datasets, key=lambda x: x.name))

    _workspace_cache[workspace_dir] = {"sig": sig, "entries": entries}
    _save_workspace_index(workspace_dir, sig, new_index)
    log_perf(f"scan_workspace built {len(entries)} entries in {time.perf_counter() - start:.3f}s","perf:workspace")
    return entries


def collect_dataset_info(dataset_dir: Path, parent: Optional[str] = None) -> WorkspaceDatasetInfo | None:
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return None
    if not _is_dataset_dir(dataset_dir):
        return None
    start = time.perf_counter()
    cache_path = dataset_dir / DATASET_CACHE_FILENAME
    cache_entry: DatasetCache = load_dataset_cache_file(cache_path)
    cache_mtime = cache_path.stat().st_mtime if cache_path.exists() else 0.0

    quick_sig = _quick_signature(dataset_dir)
    cached_sig = cache_entry.get("workspace_sig") if isinstance(cache_entry, dict) else None
    if cached_sig and abs(float(cached_sig) - quick_sig) < 1e-6:
        info = _build_info_from_cache(cache_entry, dataset_dir, parent)
        if info:
            log_perf(f"dataset {dataset_dir.name}: cache_shortcut=True in {time.perf_counter() - start:.3f}s", "perf:workspace")
            return info
    cache_valid = cache_mtime >= _dataset_signature(dataset_dir)
    cache_dirty = False

    total_pairs, pairs_dirty = _load_total_pairs(dataset_dir, cache_entry, cache_valid)
    cache_dirty = cache_dirty or pairs_dirty

    # ALWAYS compute reason_counts from marks dict (single source of truth)
    # reason_counts in cache may be corrupted from earlier bugs
    marks = cache_entry.get("marks") if isinstance(cache_entry, dict) else {}
    tagged_reason_counts: Dict[str, int] = {}

    if isinstance(marks, dict):
        # Count marks by reason from marks dict
        for base, reason in marks.items():
            if isinstance(reason, str) and isinstance(base, str):
                tagged_reason_counts[reason] = tagged_reason_counts.get(reason, 0) + 1
        log_debug(f"Dataset: {dataset_dir.name}: Computed reason_counts from marks: {tagged_reason_counts}", "WorkspaceInspector")

    # Deserialize auto_marks (dict of reason -> list of bases)
    tagged_auto_marks_raw = cache_entry.get("auto_marks") if isinstance(cache_entry, dict) else {}
    tagged_auto_marks: Dict[str, Set[str]] = {}
    if isinstance(tagged_auto_marks_raw, dict):
        for reason, bases in tagged_auto_marks_raw.items():
            if isinstance(reason, str) and isinstance(bases, list):
                tagged_auto_marks[reason] = {b for b in bases if isinstance(b, str)}

    if tagged_auto_marks:
        log_debug(f"Dataset: {dataset_dir.name}: auto_marks loaded: {', '.join(f'{k}={len(v)}' for k, v in tagged_auto_marks.items())}", "WorkspaceInspector")

    # If we have reason_counts but no auto_marks, reconstruct them
    if tagged_reason_counts and not tagged_auto_marks:
        marks = cache_entry.get("marks") if isinstance(cache_entry, dict) else {}

        if isinstance(marks, dict):
            for base, reason in marks.items():
                if isinstance(base, str) and isinstance(reason, str) and reason in AUTO_REASONS:
                    tagged_auto_marks.setdefault(reason, set()).add(base)
            if tagged_auto_marks:
                log_debug(f"Dataset: {dataset_dir.name}: reconstructed auto_marks from marks: {list(tagged_auto_marks.keys())}", "WorkspaceInspector")
            # Mark cache dirty so reconstructed auto_counts get saved
            if tagged_auto_marks:
                cache_dirty = True

    # Calculate counts from auto_marks
    tagged_auto_counts = {reason: len(bases) for reason, bases in tagged_auto_marks.items()}

    log_debug(f"Dataset: {dataset_dir.name}: reason_counts={tagged_reason_counts}, auto_counts={tagged_auto_counts}", "WorkspaceInspector")

    tagged_manual_by_reason: Dict[str, int] = {}
    for reason, count in tagged_reason_counts.items():
        auto_val = tagged_auto_counts.get(reason, 0)
        tagged_manual_by_reason[reason] = max(0, count - auto_val)
    tagged_manual_total = sum(tagged_manual_by_reason.values())
    tagged_auto_total = sum(tagged_auto_counts.values())

    log_debug(f"Dataset: {dataset_dir.name}: manual_total={tagged_manual_total}, auto_total={tagged_auto_total}, manual_by_reason={tagged_manual_by_reason}", "WorkspaceInspector")

    removed_user_by_reason, removed_auto_by_reason = _load_delete_reasons(dataset_dir)
    removed_by_reason: Dict[str, int] = {}
    for reason, count in removed_user_by_reason.items():
        removed_by_reason[reason] = removed_by_reason.get(reason, 0) + count
    for reason, count in removed_auto_by_reason.items():
        removed_by_reason[reason] = removed_by_reason.get(reason, 0) + count
    removed_total = _count_deleted_pairs(dataset_dir)

    note_value = cache_entry.get("note") if isinstance(cache_entry, dict) else ""
    note = note_value if isinstance(note_value, str) else ""
    calib = _calibration_summary(cache_entry)

    # Extract sweep_flags from cache
    sweep_flags = cache_entry.get("sweep_flags", {}) if isinstance(cache_entry, dict) else {}
    sweep_dups = sweep_flags.get("duplicates", False) if isinstance(sweep_flags, dict) else False
    sweep_qual = sweep_flags.get("quality", False) if isinstance(sweep_flags, dict) else False
    sweep_pats = sweep_flags.get("patterns", False) if isinstance(sweep_flags, dict) else False

    cache_entry["workspace_sig"] = quick_sig
    cache_entry["reason_counts"] = tagged_reason_counts
    cache_entry["auto_counts"] = tagged_auto_counts
    if cache_dirty:
        save_dataset_cache_file(cache_path, cache_entry)
        log_debug(f"Dataset: {dataset_dir.name}: Saved cache with auto_counts={tagged_auto_counts}", "WorkspaceInspector")

    stats = DatasetStats(
        total_pairs=total_pairs,
        removed_total=removed_total,
        tagged_manual=tagged_manual_total,
        tagged_auto=tagged_auto_total,
        removed_by_reason=removed_by_reason if removed_total > 0 else {},
        removed_user_by_reason=removed_user_by_reason if removed_total > 0 else {},
        removed_auto_by_reason=removed_auto_by_reason if removed_total > 0 else {},
        tagged_by_reason=tagged_manual_by_reason,
        tagged_auto_by_reason=tagged_auto_counts,
        calibration_marked=calib["marked"],
        calibration_both=calib["both"],
        calibration_partial=calib["partial"],
        calibration_missing=calib["missing"],
        outlier_lwir=calib["out_lwir"],
        outlier_visible=calib["out_vis"],
        outlier_stereo=calib["out_stereo"],
        sweep_duplicates_done=sweep_dups,
        sweep_quality_done=sweep_qual,
        sweep_patterns_done=sweep_pats,
    )

    info = WorkspaceDatasetInfo(
        name=dataset_dir.name,
        path=dataset_dir,
        note=note,
        parent=parent,
        stats=stats,
    )
    log_perf(
        f"dataset {dataset_dir.name}: pairs={total_pairs} removed={removed_total} "
        f"calib={calib['marked']} cache_valid={cache_valid} in {time.perf_counter() - start:.3f}s",
        "perf:workspace"
    )
    return info


def save_dataset_note(dataset_dir: Path, note: str) -> None:
    cache_path = dataset_dir / DATASET_CACHE_FILENAME
    entry: DatasetCache = load_dataset_cache_file(cache_path)
    if not isinstance(entry, dict):
        entry = empty_dataset_entry()
    entry["note"] = note.strip()
    save_dataset_cache_file(cache_path, entry)


def _load_dataset_note(dataset_dir: Path) -> str:
    cache_path = dataset_dir / DATASET_CACHE_FILENAME
    entry: DatasetCache = load_dataset_cache_file(cache_path)
    note = entry.get("note") if isinstance(entry, dict) else ""
    return note if isinstance(note, str) else ""


def _load_reason_counts(
    dataset_dir: Path,
    cache_entry: DatasetCache,
    cache_valid: bool,
) -> tuple[Dict[str, int], Dict[str, int], bool]:
    cache_dirty = False
    cached_reasons = cache_entry.get("reason_counts") if isinstance(cache_entry, dict) else {}
    cached_auto = cache_entry.get("auto_counts") if isinstance(cache_entry, dict) else {}
    normalized_auto = _normalize_int_dict(cached_auto)
    reason_files = _reason_files(dataset_dir)
    has_reason_files = bool(reason_files)
    if cache_valid and (isinstance(cached_reasons, dict) or isinstance(cached_auto, dict)):
        normalized_reasons = _normalize_int_dict(cached_reasons)
        if not normalized_reasons and normalized_auto:
            normalized_reasons = dict(normalized_auto)
        if normalized_reasons or not has_reason_files:
            return normalized_reasons, normalized_auto, False

    reason_counts = _load_delete_reasons(dataset_dir, reason_files)
    cache_entry["reason_counts"] = reason_counts
    cache_entry["auto_counts"] = normalized_auto
    cache_dirty = True
    normalized_reasons = _normalize_int_dict(reason_counts)
    if not normalized_reasons and normalized_auto:
        normalized_reasons = dict(normalized_auto)
    return normalized_reasons, normalized_auto, cache_dirty


def _load_total_pairs(dataset_dir: Path, cache_entry: DatasetCache, cache_valid: bool) -> tuple[int, bool]:
    cached_total = cache_entry.get("total_pairs") if isinstance(cache_entry, dict) else None
    if cache_valid and isinstance(cached_total, (int, float)):
        return int(cached_total), False
    total_pairs = _quick_count_pairs(dataset_dir)
    cache_entry["total_pairs"] = total_pairs
    return total_pairs, True


def _calibration_summary(entry: DatasetCache) -> Dict[str, int]:
    raw = entry.get("calibration") if isinstance(entry, dict) else {}
    calibration = deserialize_calibration(raw)

    # Extract marked, results, and outliers from the new dict format
    marked = {base for base, data in calibration.items() if data.get("marked")}
    results = {base: data.get("results", {}) for base, data in calibration.items()}
    out_lwir = {base for base, data in calibration.items() if data.get("outlier_lwir")}
    out_visible = {base for base, data in calibration.items() if data.get("outlier_visible")}
    out_stereo = {base for base, data in calibration.items() if data.get("outlier_stereo")}

    counts = {"marked": len(marked), "both": 0, "partial": 0, "missing": 0}
    bases = set(marked) | set(results.keys())

    # If results are empty but we have marked images, infer detection status from saved corners on disk
    inferred: Dict[str, Dict[str, bool]] = {}
    if marked and (not isinstance(results, dict) or not results):
        try:
            # Late import to avoid circular deps; calibration corners live in per-dataset YAMLs.
            from backend.services.calibration_corners_io import load_corners_for_dataset

            dataset_path = entry.get("dataset_path") if isinstance(entry, dict) else None
            # Some callers don't inject dataset_path; infer from active workspace inspector caller if present.
            if dataset_path is None:
                dataset_path = entry.get("_dataset_path") if isinstance(entry, dict) else None
            # load_corners_for_dataset expects a Path
            if dataset_path is not None:
                corners_map = load_corners_for_dataset(Path(dataset_path))
                # corners_map: base -> {lwir: [...], visible: [...]}
                for base, bucket in (corners_map or {}).items():
                    if not isinstance(base, str) or not isinstance(bucket, dict):
                        continue
                    inferred[base] = {
                        "lwir": bool(bucket.get("lwir")),
                        "visible": bool(bucket.get("visible")),
                    }
        except Exception:
            inferred = {}

    for base in bases:
        res = results.get(base, {}) if isinstance(results, dict) else {}
        if isinstance(res, dict) and (res.get("lwir") is True or res.get("visible") is True):
            positives = sum(1 for cam in ("lwir", "visible") if res.get(cam) is True)
        else:
            inferred_res = inferred.get(base, {})
            positives = sum(1 for cam in ("lwir", "visible") if inferred_res.get(cam) is True)
        if positives >= 2:
            counts["both"] += 1
        elif positives == 1:
            counts["partial"] += 1
        else:
            counts["missing"] += 1
    return {
        **counts,
        "out_lwir": len(out_lwir),
        "out_vis": len(out_visible),
        "out_stereo": len(out_stereo),
    }


def _load_delete_reasons(dataset_dir: Path, reason_files: Optional[List[Path]] = None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Load delete reasons separated by source (user vs auto).

    Returns:
        Tuple of (user_reasons, auto_reasons) dicts with counts per reason
    """
    reasons_dir = dataset_dir / "to_delete" / "reasons"
    if reason_files is None:
        reason_files = _reason_files(dataset_dir)
    if not reasons_dir.exists() or not reasons_dir.is_dir() or not reason_files:
        return {}, {}
    user_counts: Dict[str, int] = {}
    auto_counts: Dict[str, int] = {}
    for yaml_path in reason_files:
        try:
            with open(yaml_path, "r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle)
        except (OSError, yaml.YAMLError):  # noqa: PERF203
            continue
        reason_value = payload.get("reason") if isinstance(payload, dict) else None
        if not isinstance(reason_value, str):
            reason_value = "unknown"
        # Read source field (defaults to "user" for old YAMLs without source)
        source = payload.get("source", "user") if isinstance(payload, dict) else "user"
        if source == "auto":
            auto_counts[reason_value] = auto_counts.get(reason_value, 0) + 1
        else:
            user_counts[reason_value] = user_counts.get(reason_value, 0) + 1
    return user_counts, auto_counts


def _count_deleted_pairs(dataset_dir: Path) -> int:
    to_delete_dir = dataset_dir / "to_delete"
    lwir_dir = to_delete_dir / "lwir"
    vis_dir = to_delete_dir / "visible"
    counts = []
    for folder in (lwir_dir, vis_dir):
        if not folder.exists() or not folder.is_dir():
            counts.append(0)
            continue
        counts.append(len([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]))
    if not counts:
        return 0
    # Use the minimum to approximate paired removals; fall back to the max if one side is empty.
    if counts[0] == 0 or counts[1] == 0:
        return max(counts)
    return min(counts)


def _is_dataset_dir(path: Path) -> bool:
    return (path / "lwir").is_dir() and (path / "visible").is_dir()


def _workspace_signature(workspace_dir: Path) -> float:
    try:
        # Include logic version so cache invalidates when processing logic changes
        mtimes = [workspace_dir.stat().st_mtime, float(WORKSPACE_LOGIC_VERSION)]
        for entry in workspace_dir.iterdir():
            if not entry.is_dir():
                continue
            try:
                mtimes.append(entry.stat().st_mtime)
            except OSError:
                continue
            if _is_dataset_dir(entry):
                mtimes.append(_dataset_signature(entry))
                continue
            for child in entry.iterdir():
                if child.is_dir() and _is_dataset_dir(child):
                    mtimes.append(_dataset_signature(child))
        return max(mtimes)
    except OSError:
        return 0.0


def _dataset_signature(dataset_dir: Path) -> float:
    try:
        mtimes = [dataset_dir.stat().st_mtime]
        for folder in (
            dataset_dir / "lwir",
            dataset_dir / "visible",
            dataset_dir / "to_delete",
            dataset_dir / "to_delete" / "lwir",
            dataset_dir / "to_delete" / "visible",
            dataset_dir / "to_delete" / "reasons",
        ):
            if folder.exists():
                try:
                    mtimes.append(folder.stat().st_mtime)
                except OSError:
                    pass
        for yaml_path in _reason_files(dataset_dir):
            try:
                mtimes.append(yaml_path.stat().st_mtime)
            except OSError:
                continue
        return max(mtimes)
    except OSError:
        return 0.0


_workspace_cache: Dict[Path, Dict[str, object]] = {}


def invalidate_workspace_cache(workspace_dir: Path) -> None:
    """Invalidate workspace cache so it rescans on next access.

    Call this after modifying dataset statistics (marks, auto_marks, etc.)
    to ensure workspace panel shows updated data.
    """

    if workspace_dir in _workspace_cache:
        log_debug(f"Removing workspace from memory cache: {workspace_dir.name}", "CacheInvalidate")
        del _workspace_cache[workspace_dir]
    # Also delete workspace index file to force full rescan
    index_path = workspace_dir / WORKSPACE_INDEX_FILENAME
    if index_path.exists():
        log_debug(f"Deleting workspace index: {index_path.name}", "CacheInvalidate")
        try:
            index_path.unlink()
        except OSError:
            pass


def _quick_signature(dataset_dir: Path) -> float:
    try:
        sigs = [dataset_dir.stat().st_mtime]
        cache_path = dataset_dir / DATASET_CACHE_FILENAME
        if cache_path.exists():
            try:
                sigs.append(cache_path.stat().st_mtime)
            except OSError:
                pass
        for sub in (dataset_dir / "lwir", dataset_dir / "visible"):
            try:
                sigs.append(sub.stat().st_mtime)
            except OSError:
                continue
        return max(sigs)
    except OSError:
        return 0.0


def _quick_count_pairs(dataset_dir: Path) -> int:
    try:
        lwir_dir = dataset_dir / "lwir"
        vis_dir = dataset_dir / "visible"
        if not lwir_dir.exists() or not vis_dir.exists():
            return 0

        def _bases(root: Path, type_dir: str) -> set:
            bases = set()
            for cur, dirs, files in os.walk(root):
                dirs[:] = [d for d in dirs if d not in {"to_delete", "reasons"}]
                for name in files:
                    suffix = Path(name).suffix.lower()
                    if suffix not in {".jpg", ".jpeg", ".png"}:
                        continue
                    rel_parent = Path(cur).relative_to(root)
                    stem = Path(name).stem
                    if stem.startswith(f"{type_dir}_"):
                        stem = stem[len(type_dir) + 1 :]
                    base = str(rel_parent / stem) if rel_parent != Path(".") else stem
                    bases.add(base)
            return bases

        lwir_bases = _bases(lwir_dir, "lwir")
        vis_bases = _bases(vis_dir, "visible")
        return len(lwir_bases | vis_bases)
    except OSError:
        return 0


def _build_info_from_cache(cache_entry: DatasetCache, dataset_dir: Path, parent: Optional[str]) -> Optional[WorkspaceDatasetInfo]:
    """Build WorkspaceDatasetInfo from cache entry using derive_summary_from_entry().

    This ensures consistency: summary is DERIVED from entry, not manually computed.
    """
    if not isinstance(cache_entry, dict):
        return None

    # ✅ Use derive_summary_from_entry() for consistency (single source of truth)
    summary_dict = derive_summary_from_entry(cache_entry)

    # Extract values using get_dict_path (safe nested access)
    total_pairs = get_dict_path(summary_dict, "img_number.num_pairs", 0, expected_type=int) or 0
    tagged_manual = get_dict_path(summary_dict, "img_number.tagged_user_to_delete", 0, expected_type=int) or 0
    tagged_auto = get_dict_path(summary_dict, "img_number.tagged_auto_to_delete", 0, expected_type=int) or 0

    # Reason breakdowns
    tagged_by_reason: Dict[str, int] = get_dict_path(summary_dict, "tagged_user_to_delete_reasons", {}, expected_type=dict) or {}
    tagged_auto_by_reason: Dict[str, int] = get_dict_path(summary_dict, "tagged_auto_to_delete_reasons", {}, expected_type=dict) or {}

    # Calibration stats
    calibration: Dict[str, Any] = get_dict_path(summary_dict, "calibration", {}, expected_type=dict) or {}
    calib_marked = get_dict_path(calibration, "marked", 0, expected_type=int) or 0
    calib_both = get_dict_path(calibration, "found_both_chessboard", 0, expected_type=int) or 0
    calib_only_lwir = get_dict_path(calibration, "found_only_lwir_chessboard", 0, expected_type=int) or 0
    calib_only_vis = get_dict_path(calibration, "found_only_visible_chessboard", 0, expected_type=int) or 0
    calib_none = get_dict_path(calibration, "found_none_chessboard", 0, expected_type=int) or 0
    # Combine for total partial calibration
    calib_partial = calib_only_lwir + calib_only_vis

    # Outliers
    out_lwir = get_dict_path(calibration, "outlier_discarded_lwir", 0, expected_type=int) or 0
    out_vis = get_dict_path(calibration, "outlier_discarded_visible", 0, expected_type=int) or 0
    out_stereo = get_dict_path(calibration, "outlier_discarded_stereo", 0, expected_type=int) or 0

    # Note
    note = get_dict_path(summary_dict, "dataset_info.note", "", expected_type=str) or ""

    # Sweep flags from cache_entry (not from derived summary)
    sweep_flags = cache_entry.get("sweep_flags", {}) if isinstance(cache_entry, dict) else {}
    sweep_dups = sweep_flags.get("duplicates", False) if isinstance(sweep_flags, dict) else False
    sweep_qual = sweep_flags.get("quality", False) if isinstance(sweep_flags, dict) else False
    sweep_pats = sweep_flags.get("patterns", False) if isinstance(sweep_flags, dict) else False

    # Load deleted data from filesystem (not in cache)
    removed_user_by_reason, removed_auto_by_reason = _load_delete_reasons(dataset_dir)
    removed_by_reason: Dict[str, int] = {}
    for reason, count in removed_user_by_reason.items():
        removed_by_reason[reason] = removed_by_reason.get(reason, 0) + count
    for reason, count in removed_auto_by_reason.items():
        removed_by_reason[reason] = removed_by_reason.get(reason, 0) + count
    removed_total = _count_deleted_pairs(dataset_dir)
    if removed_total == 0:
        removed_by_reason = {}
        removed_user_by_reason = {}
        removed_auto_by_reason = {}

    stats = DatasetStats(
        total_pairs=total_pairs,
        removed_total=removed_total,
        tagged_manual=tagged_manual,
        tagged_auto=tagged_auto,
        removed_by_reason=removed_by_reason,
        removed_user_by_reason=removed_user_by_reason,
        removed_auto_by_reason=removed_auto_by_reason,
        tagged_by_reason=tagged_by_reason,
        tagged_auto_by_reason=tagged_auto_by_reason,
        calibration_marked=calib_marked,
        calibration_both=calib_both,
        calibration_partial=calib_partial,
        calibration_missing=calib_none,
        outlier_lwir=out_lwir,
        outlier_visible=out_vis,
        outlier_stereo=out_stereo,
        sweep_duplicates_done=sweep_dups,
        sweep_quality_done=sweep_qual,
        sweep_patterns_done=sweep_pats,
    )

    return WorkspaceDatasetInfo(
        name=dataset_dir.name,
        path=dataset_dir,
        note=note,
        parent=parent,
        stats=stats,
    )


def _reason_files(dataset_dir: Path) -> List[Path]:
    reasons_dir = dataset_dir / "to_delete" / "reasons"
    if not reasons_dir.exists() or not reasons_dir.is_dir():
        return []
    return [p for p in reasons_dir.glob("*.yaml") if p.is_file()]


def _aggregate_collection(collection_dir: Path, children: List[WorkspaceDatasetInfo]) -> WorkspaceDatasetInfo:
    """Aggregate collection info from children using Collection class.

    Creates a Collection instance, aggregates data, and builds WorkspaceDatasetInfo.
    """
    from backend.services.collection import Collection

    child_names = [c.name for c in children]
    log_debug(f"Collection {collection_dir.name}: aggregating from {len(children)} children: {child_names}", "INSPECTOR")

    # Create Collection and aggregate from children
    collection = Collection(collection_dir)
    collection.discover_children()
    collection.aggregate_from_children()

    # Extract aggregated data from collection
    tagged_manual_total = len([b for b, r in collection.marks.items() if not r.startswith("auto:")])
    tagged_auto_total = len([b for b, r in collection.marks.items() if r.startswith("auto:")])

    # Count by reason
    tagged_by_reason: Dict[str, int] = {}
    tagged_auto_by_reason: Dict[str, int] = {}
    for base, reason in collection.marks.items():
        if reason.startswith("auto:"):
            actual_reason = reason[5:]  # Strip "auto:" prefix
            tagged_auto_by_reason[actual_reason] = tagged_auto_by_reason.get(actual_reason, 0) + 1
        else:
            tagged_by_reason[reason] = tagged_by_reason.get(reason, 0) + 1

    log_debug(f"Collection {collection_dir.name}: aggregated - manual={tagged_manual_total}, auto={tagged_auto_total}", "INSPECTOR")

    # Aggregate these from children
    total_pairs = sum(child.stats.total_pairs for child in children)
    removed_total = sum(child.stats.removed_total for child in children)
    removed_by_reason: Dict[str, int] = {}
    for child in children:
        _merge_count_dicts(removed_by_reason, child.stats.removed_by_reason)

    # Create collection stats by merging children (includes sweep flags OR logic)
    from backend.services.stats_manager import DatasetStats
    collection_stats = DatasetStats()
    for child in children:
        collection_stats.merge(child.stats)

    return WorkspaceDatasetInfo(
        name=collection_dir.name,
        path=collection_dir,
        note="",
        is_collection=True,
        children=children,
        parent=None,
        stats=collection_stats,
    )


def _info_to_summary(info: WorkspaceDatasetInfo, workspace_dir: Path) -> Dict[str, object]:
    return {
        "name": info.name,
        "rel": str(info.path.relative_to(workspace_dir)),
        "total_pairs": info.stats.total_pairs,
        "removed_total": info.stats.removed_total,
        "tagged_manual": info.stats.tagged_manual,
        "tagged_auto": info.stats.tagged_auto,
        "removed_by_reason": dict(info.stats.removed_by_reason),
        "tagged_by_reason": dict(info.stats.tagged_by_reason),
        "tagged_auto_by_reason": dict(info.stats.tagged_auto_by_reason),
        "note": info.note,
        "parent": info.parent,
        "is_collection": info.is_collection,
        "calibration": {
            "marked": info.stats.calibration_marked,
            "both": info.stats.calibration_both,
            "partial": info.stats.calibration_partial,
            "missing": info.stats.calibration_missing,
            "out_lwir": info.stats.outlier_lwir,
            "out_vis": info.stats.outlier_visible,
            "out_stereo": info.stats.outlier_stereo,
        },
        "sweep_flags": {
            "duplicates": info.stats.sweep_duplicates_done,
            "quality": info.stats.sweep_quality_done,
            "patterns": info.stats.sweep_patterns_done,
        },
    }


def _summary_to_info(summary: Dict[str, object], workspace_dir: Path, parent_hint: Optional[str]) -> Optional[WorkspaceDatasetInfo]:
    """Convert summary dict to WorkspaceDatasetInfo using get_dict_path() for safe access."""
    if not isinstance(summary, dict):
        return None

    # Basic fields with safe access
    rel = get_dict_path(summary, "rel", expected_type=str)
    name = get_dict_path(summary, "name", expected_type=str)
    if not rel or not name:
        return None

    path = workspace_dir / rel
    parent = parent_hint if parent_hint is not None else get_dict_path(summary, "parent", expected_type=str)
    is_coll = get_dict_path(summary, "is_collection", False, expected_type=bool)

    # Image counts
    total_pairs = get_dict_path(summary, "total_pairs", 0, expected_type=int)
    removed_total = get_dict_path(summary, "removed_total", 0, expected_type=int)
    tagged_manual = get_dict_path(summary, "tagged_manual", 0, expected_type=int)
    tagged_auto = get_dict_path(summary, "tagged_auto", 0, expected_type=int)

    # Reason breakdowns
    removed_by_reason = _normalize_int_dict(get_dict_path(summary, "removed_by_reason", {}))
    tagged_by_reason = _normalize_int_dict(get_dict_path(summary, "tagged_by_reason", {}))
    tagged_auto_by_reason = _normalize_int_dict(get_dict_path(summary, "tagged_auto_by_reason", {}))

    # Calibration stats using get_dict_path
    calib_marked = get_dict_path(summary, "calibration.marked", 0, expected_type=int)
    calib_both = get_dict_path(summary, "calibration.both", 0, expected_type=int)
    calib_partial = get_dict_path(summary, "calibration.partial", 0, expected_type=int)
    calib_missing = get_dict_path(summary, "calibration.missing", 0, expected_type=int)
    out_lwir = get_dict_path(summary, "calibration.out_lwir", 0, expected_type=int)
    out_vis = get_dict_path(summary, "calibration.out_vis", 0, expected_type=int)
    out_stereo = get_dict_path(summary, "calibration.out_stereo", 0, expected_type=int)

    # Sweep flags using get_dict_path
    sweep_duplicates = get_dict_path(summary, "sweep_flags.duplicates", False, expected_type=bool)
    sweep_quality = get_dict_path(summary, "sweep_flags.quality", False, expected_type=bool)
    sweep_patterns = get_dict_path(summary, "sweep_flags.patterns", False, expected_type=bool)

    # Note
    note = get_dict_path(summary, "note", "", expected_type=str)

    stats = DatasetStats(
        total_pairs=total_pairs,
        removed_total=removed_total,
        tagged_manual=tagged_manual,
        tagged_auto=tagged_auto,
        removed_by_reason=removed_by_reason,
        tagged_by_reason=tagged_by_reason,
        tagged_auto_by_reason=tagged_auto_by_reason,
        calibration_marked=calib_marked,
        calibration_both=calib_both,
        calibration_partial=calib_partial,
        calibration_missing=calib_missing,
        outlier_lwir=out_lwir,
        outlier_visible=out_vis,
        outlier_stereo=out_stereo,
        sweep_duplicates_done=sweep_duplicates,
        sweep_quality_done=sweep_quality,
        sweep_patterns_done=sweep_patterns,
    )

    info = WorkspaceDatasetInfo(
        name=name,
        path=path,
        note=note,
        is_collection=is_coll,
        parent=parent,
        stats=stats,
    )

    # Refresh deleted info from disk to avoid stale cached values
    removed_total = _count_deleted_pairs(info.path)
    if removed_total > 0:
        removed_user_by_reason, removed_auto_by_reason = _load_delete_reasons(info.path)
        # Merge both user and auto into removed_by_reason already computed above
        for reason, count in removed_user_by_reason.items():
            removed_by_reason[reason] = removed_by_reason.get(reason, 0) + count
        for reason, count in removed_auto_by_reason.items():
            removed_by_reason[reason] = removed_by_reason.get(reason, 0) + count
    else:
        removed_by_reason = {}
        removed_user_by_reason = {}
        removed_auto_by_reason = {}

    # Create new stats with updated removed data from disk
    updated_stats = DatasetStats(
        total_pairs=info.stats.total_pairs,
        removed_total=removed_total,
        tagged_manual=info.stats.tagged_manual,
        tagged_auto=info.stats.tagged_auto,
        removed_by_reason=removed_by_reason,
        removed_user_by_reason=removed_user_by_reason,
        removed_auto_by_reason=removed_auto_by_reason,
        tagged_by_reason=info.stats.tagged_by_reason,
        tagged_auto_by_reason=info.stats.tagged_auto_by_reason,
        calibration_marked=info.stats.calibration_marked,
        calibration_both=info.stats.calibration_both,
        calibration_partial=info.stats.calibration_partial,
        calibration_missing=info.stats.calibration_missing,
        outlier_lwir=info.stats.outlier_lwir,
        outlier_visible=info.stats.outlier_visible,
        outlier_stereo=info.stats.outlier_stereo,
        sweep_duplicates_done=info.stats.sweep_duplicates_done,
        sweep_quality_done=info.stats.sweep_quality_done,
        sweep_patterns_done=info.stats.sweep_patterns_done,
    )

    # Always reconstruct with fresh deleted data from disk
    return WorkspaceDatasetInfo(
        name=info.name,
        path=info.path,
        note=info.note,
        is_collection=info.is_collection,
        children=info.children,
        parent=info.parent,
        stats=updated_stats,
    )


def _load_workspace_index(workspace_dir: Path) -> Dict[str, object]:
    index_path = workspace_dir / WORKSPACE_INDEX_FILENAME
    try:
        with open(index_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, dict) else {}
    except OSError:
        return {}
    except json.JSONDecodeError:
        return {}


def _save_workspace_index(workspace_dir: Path, workspace_sig: float, datasets: Dict[str, Dict[str, object]]) -> None:
    index_path = workspace_dir / WORKSPACE_INDEX_FILENAME
    payload = {"sig": workspace_sig, "datasets": datasets}
    try:
        with open(index_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except OSError:
        return


def _scan_workers() -> int:
    cpu = os.cpu_count() or 2
    return max(2, min(8, cpu))


