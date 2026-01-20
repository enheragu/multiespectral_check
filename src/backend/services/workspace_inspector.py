"""Helpers to inspect workspace datasets and persist per-dataset notes."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from backend.services.cache_service import (DATASET_CACHE_FILENAME,
                                            empty_dataset_entry,
                                            load_dataset_cache_file,
                                            save_dataset_cache_file)
from backend.services.stats_manager import DatasetStats
from backend.services.summary_derivation import derive_summary_from_entry
from backend.utils.cache import DatasetCache
from common.dict_helpers import get_dict_path, normalize_int_dict
from common.log_utils import log_debug, log_perf
from common.yaml_utils import load_yaml

if TYPE_CHECKING:
    from backend.services.thread_pool_manager import ThreadPoolManager

try:
    from backend.services.thread_pool_manager import get_thread_pool_manager
except ImportError:
    get_thread_pool_manager = None  # type: ignore

WORKSPACE_INDEX_FILENAME = ".workspace_index.json"

# Bump this version whenever workspace aggregation logic changes
# to force cache regeneration
WORKSPACE_LOGIC_VERSION = 3


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
    """Collect dataset info from filesystem/cache.

    Uses derive_summary_from_entry() for consistency - single source of truth.
    """
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return None
    if not _is_dataset_dir(dataset_dir):
        return None
    start = time.perf_counter()
    cache_path = dataset_dir / DATASET_CACHE_FILENAME
    cache_entry: DatasetCache = load_dataset_cache_file(cache_path)
    cache_mtime = cache_path.stat().st_mtime if cache_path.exists() else 0.0

    # Check if we can use cached shortcut
    quick_sig = _quick_signature(dataset_dir)
    cached_sig = cache_entry.get("workspace_sig") if isinstance(cache_entry, dict) else None
    if cached_sig and abs(float(cached_sig) - quick_sig) < 1e-6:
        info = _build_info_from_cache(cache_entry, dataset_dir, parent)
        if info:
            log_perf(f"dataset {dataset_dir.name}: cache_shortcut=True in {time.perf_counter() - start:.3f}s", "perf:workspace")
            return info

    # Cache miss - ensure total_pairs is updated
    cache_valid = cache_mtime >= _dataset_signature(dataset_dir)
    total_pairs, pairs_dirty = _load_total_pairs(dataset_dir, cache_entry, cache_valid)
    if pairs_dirty:
        cache_entry["total_pairs"] = total_pairs

    # Update workspace signature
    cache_entry["workspace_sig"] = quick_sig
    if pairs_dirty:
        save_dataset_cache_file(cache_path, cache_entry)

    # Derive summary and build info (single source of truth)
    summary_dict = derive_summary_from_entry(cache_entry)
    stats_data = get_dict_path(summary_dict, "stats", {}, expected_type=dict) or {}
    stats = DatasetStats(stats_data)
    note = get_dict_path(summary_dict, "dataset_info.note", "", expected_type=str) or ""

    info = WorkspaceDatasetInfo(
        name=dataset_dir.name,
        path=dataset_dir,
        note=note,
        parent=parent,
        stats=stats,
    )
    log_perf(
        f"dataset {dataset_dir.name}: pairs={stats.total_pairs} removed={stats.removed_total} "
        f"calib={stats.calibration_marked} cache_valid={cache_valid} in {time.perf_counter() - start:.3f}s",
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
    normalized_auto = normalize_int_dict(cached_auto)
    reason_files = _reason_files(dataset_dir)
    has_reason_files = bool(reason_files)
    if cache_valid and (isinstance(cached_reasons, dict) or isinstance(cached_auto, dict)):
        normalized_reasons = normalize_int_dict(cached_reasons)
        if not normalized_reasons and normalized_auto:
            normalized_reasons = dict(normalized_auto)
        if normalized_reasons or not has_reason_files:
            return normalized_reasons, normalized_auto, False

    reason_counts = _load_delete_reasons(dataset_dir, reason_files)
    cache_entry["reason_counts"] = reason_counts
    cache_entry["auto_counts"] = normalized_auto
    cache_dirty = True
    normalized_reasons = normalize_int_dict(reason_counts)
    if not normalized_reasons and normalized_auto:
        normalized_reasons = dict(normalized_auto)
    return normalized_reasons, normalized_auto, cache_dirty


def _load_total_pairs(dataset_dir: Path, cache_entry: DatasetCache, cache_valid: bool) -> tuple[int, bool]:
    """Count total image pairs from filesystem.

    NOTE: total_pairs is NOT persisted to YAML (derived from filesystem).
    Always counts from filesystem for accuracy.
    """
    total_pairs = _quick_count_pairs(dataset_dir)
    cache_entry["total_pairs"] = total_pairs  # Update in-memory cache
    return total_pairs, False  # Never dirty (not persisted)


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
        payload = load_yaml(yaml_path)
        if payload=={} or payload is None:
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

    Call this after modifying dataset statistics (marks, calibration, etc.)
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

    # Extract stats dict directly - much simpler!
    stats_data = get_dict_path(summary_dict, "stats", {}, expected_type=dict) or {}
    stats = DatasetStats(stats_data)

    # Note from dataset_info
    note = get_dict_path(summary_dict, "dataset_info.note", "", expected_type=str) or ""

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
    """Aggregate collection info from children by merging stats.

    Uses DatasetStats.merge() for proper aggregation (sums ints, ORs bools).
    """

    child_names = [c.name for c in children]
    log_debug(f"Collection {collection_dir.name}: aggregating from {len(children)} children: {child_names}", "INSPECTOR")

    # Create collection stats by merging all children
    collection_stats = DatasetStats()
    for child in children:
        collection_stats.merge(child.stats)

    log_debug(f"Collection {collection_dir.name}: aggregated - "
              f"pairs={collection_stats.total_pairs}, "
              f"manual={collection_stats.tagged_manual}, "
              f"auto={collection_stats.tagged_auto}", "INSPECTOR")

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
    """Convert WorkspaceDatasetInfo to summary dict for index storage."""
    return {
        "name": info.name,
        "rel": str(info.path.relative_to(workspace_dir)),
        "note": info.note,
        "parent": info.parent,
        "is_collection": info.is_collection,
        "stats": info.stats.to_dict(),  # Direct dict serialization!
    }


def _summary_to_info(summary: Dict[str, object], workspace_dir: Path, parent_hint: Optional[str]) -> Optional[WorkspaceDatasetInfo]:
    """Convert summary dict to WorkspaceDatasetInfo."""
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
    note = get_dict_path(summary, "note", "", expected_type=str)

    # Stats from dict - much simpler!
    stats_data = get_dict_path(summary, "stats", {}, expected_type=dict) or {}
    stats = DatasetStats(stats_data)

    return WorkspaceDatasetInfo(
        name=name,
        path=path,
        note=note,
        is_collection=is_coll,
        parent=parent,
        stats=stats,
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


