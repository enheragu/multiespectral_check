"""Persistent YAML cache helpers for viewer state."""
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from common.yaml_utils import get_timestamp_fields, load_yaml, save_yaml

from common.log_utils import log_warning
from config import get_config

CACHE_DIR = Path.home() / ".config" / "multiespectral_viewer"
CACHE_FILE = CACHE_DIR / "cache.yaml"

CacheData = Dict[str, Any]
DatasetCache = Dict[str, Any]


def _default_cache() -> CacheData:
    config = get_config()
    return {
        "version": config.cache_version,
        "last_dataset": None,
        "datasets": {},
        "preferences": {},
    }


def _normalize_dataset_entry(raw: Any) -> DatasetCache:
    """Normalize raw YAML data to in-memory format.

    Expected marks format (unified):
        marks: {base: {reason: str, auto: bool}}

    overrides: YAML stores as List, memory uses Set
    """
    entry: DatasetCache = {
        "marks": {},
        "reason_counts": {},
        "calibration": {},
        "extrinsic_errors": {},
        "overrides": set(),  # Set[str]
        "note": "",
        "archived": {},
        "total_pairs": 0,
        "sweep_flags": {},
    }
    if not isinstance(raw, dict):
        return entry

    # Load marks (unified format: base -> {reason, auto})
    raw_marks = raw.get("marks", {})
    if isinstance(raw_marks, dict):
        for base, value in raw_marks.items():
            base_str = str(base)
            if isinstance(value, dict) and "reason" in value:
                # Expected format: {reason: str, auto: bool}
                entry["marks"][base_str] = {
                    "reason": str(value.get("reason", "")),
                    "auto": bool(value.get("auto", False)),
                }
            # Skip invalid entries (should not happen with migrated data)

    entry["reason_counts"] = raw.get("reason_counts", {}) if isinstance(raw.get("reason_counts"), dict) else {}

    entry["calibration"] = raw.get("calibration", {}) if isinstance(raw.get("calibration"), dict) else {}
    entry["extrinsic_errors"] = raw.get("extrinsic_errors", {}) if isinstance(raw.get("extrinsic_errors"), dict) else {}

    # overrides: List→Set conversion at load boundary
    raw_overrides = raw.get("overrides", [])
    entry["overrides"] = set(raw_overrides) if isinstance(raw_overrides, (list, set)) else set()

    note = raw.get("note")
    entry["note"] = note if isinstance(note, str) else ""

    archived = raw.get("archived")
    entry["archived"] = archived if isinstance(archived, dict) else {}

    total_pairs = raw.get("total_pairs")
    entry["total_pairs"] = int(total_pairs) if isinstance(total_pairs, (int, float)) else 0

    sweep_flags = raw.get("sweep_flags")
    entry["sweep_flags"] = sweep_flags if isinstance(sweep_flags, dict) else {}

    return entry


def serialize_dataset_entry(entry: DatasetCache) -> DatasetCache:
    """Prepare dataset entry for YAML persistence.

    Marks format: {base: {reason: str, auto: bool}}
    Set→List conversion at save boundary (YAML doesn't handle sets well).
    Adds last_updated timestamp for traceability.

    NOTE: Some fields are NOT persisted because they're derived:
    - reason_counts: rebuilt from marks at load time
    - total_pairs: counted from filesystem at load time
    """
    PERSIST_FIELDS = {
        "marks",  # Source of truth for tagged images
        "calibration", "extrinsic_errors",  # Calibration data (reproj_errors regenerated from per_view_errors)
        "archived", "sweep_flags", "note", "overrides",  # Other state
    }

    result: DatasetCache = {k: deepcopy(v) for k, v in entry.items() if k in PERSIST_FIELDS}
    result.setdefault("marks", {})
    result.setdefault("sweep_flags", {})

    # Set→List conversions at save boundary
    overrides = result.get("overrides", set())
    result["overrides"] = sorted(overrides) if isinstance(overrides, (set, list)) else []

    # Add timestamp for traceability
    result.update(get_timestamp_fields())

    return result


def deserialize_dataset_entry(raw: Any) -> DatasetCache:
    return _normalize_dataset_entry(raw)


def _normalize_cache(raw: Any) -> CacheData:
    cache = _default_cache()
    if not isinstance(raw, dict):
        return cache
    last_dataset = raw.get("last_dataset")
    if isinstance(last_dataset, str):
        cache["last_dataset"] = last_dataset
    datasets = raw.get("datasets")
    if isinstance(datasets, dict):
        normalized: Dict[str, DatasetCache] = {}
        for path, entry in datasets.items():
            if isinstance(path, str):
                normalized[path] = _normalize_dataset_entry(entry)
        cache["datasets"] = normalized
    prefs = raw.get("preferences")
    if isinstance(prefs, dict):
        cache["preferences"] = prefs
    else:
        cache["preferences"] = {}
    return cache


def load_cache() -> CacheData:
    if not CACHE_FILE.exists():
        return _default_cache()

    config = get_config()

    # Check cache size and warn if too large
    try:
        cache_size_mb = CACHE_FILE.stat().st_size / (1024 * 1024)
        if cache_size_mb > config.cache_large_size_mb:
            perf_enabled = os.environ.get("PERF_LOG", "").lower() not in {"", "0", "false", "off"}
            if perf_enabled:
                log_warning(f"Cache file is {cache_size_mb:.1f}MB, loading may be slow", "CACHE")
    except OSError:
        pass

    raw = load_yaml(CACHE_FILE)
    if raw=={} or raw is None:
        return _default_cache()

    cache = _normalize_cache(raw)
    cache["version"] = config.cache_version

    # Clean signatures from all datasets - they belong in local files only
    datasets = cache.get("datasets", {})
    if isinstance(datasets, dict):
        for dataset_entry in datasets.values():
            if isinstance(dataset_entry, dict) and "signatures" in dataset_entry:
                dataset_entry["signatures"] = {}

    # Aggressive cleanup if cache is huge
    try:
        if CACHE_FILE.stat().st_size > config.cache_large_size_mb * 1024 * 1024:
            trim_cache(cache, max_entries=3)  # Keep only 3 most recent
    except OSError:
        pass

    return cache


def save_cache(cache: CacheData) -> None:
    config = get_config()
    cache["version"] = config.cache_version
    # Add timestamp for traceability
    cache.update(get_timestamp_fields())
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    try:
        save_yaml(CACHE_FILE, cache, sort_keys=True)
    except OSError:
        return


def ensure_dataset_entry(cache: CacheData, dataset_path: str) -> DatasetCache:
    datasets = cache.setdefault("datasets", {})
    assert isinstance(datasets, dict)
    entry = datasets.get(dataset_path)
    if entry is None:
        entry = _normalize_dataset_entry({})
        datasets[dataset_path] = entry
    else:
        entry = _normalize_dataset_entry(entry)
        datasets[dataset_path] = entry
    return entry


def touch_dataset(cache: CacheData, dataset_path: str) -> None:
    datasets = cache.get("datasets")
    if not isinstance(datasets, dict) or dataset_path not in datasets:
        return
    entry = datasets.pop(dataset_path)
    datasets[dataset_path] = entry


def trim_cache(cache: CacheData, max_entries: Optional[int] = None) -> None:
    if max_entries is None:
        max_entries = get_config().cache_max_datasets
    datasets = cache.get("datasets")
    if not isinstance(datasets, dict):
        cache["datasets"] = {}
        return
    while len(datasets) > max_entries:
        last_dataset = cache.get("last_dataset")
        for key in list(datasets.keys()):
            if key == last_dataset and len(datasets) > 1:
                continue
            datasets.pop(key)
            break
        else:
            # Could not trim without removing the active dataset
            break

