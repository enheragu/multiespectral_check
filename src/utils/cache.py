"""Persistent YAML cache helpers for viewer state."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

CACHE_DIR = Path.home() / ".config" / "multiespectral_viewer"
CACHE_FILE = CACHE_DIR / "cache.yaml"
CACHE_VERSION = 1
MAX_DATASETS = 5

CacheData = Dict[str, Any]
DatasetCache = Dict[str, Any]


def _default_cache() -> CacheData:
    return {
        "version": CACHE_VERSION,
        "last_dataset": None,
        "datasets": {},
        "preferences": {},
    }


def _normalize_dataset_entry(raw: Any) -> DatasetCache:
    entry: DatasetCache = {
        "marks": {},
        "signatures": {},
        "calibration": {},
        "matrices": {},
        "extrinsic": {},
        "reproj_errors": {},
        "extrinsic_errors": {},
        "overrides": [],
    }
    if isinstance(raw, dict):
        entry["marks"] = raw.get("marks", {}) if isinstance(raw.get("marks"), dict) else {}
        entry["signatures"] = raw.get("signatures", {}) if isinstance(raw.get("signatures"), dict) else {}
        entry["calibration"] = raw.get("calibration", {}) if isinstance(raw.get("calibration"), dict) else {}
        entry["matrices"] = raw.get("matrices", {}) if isinstance(raw.get("matrices"), dict) else {}
        extrinsic = raw.get("extrinsic")
        if isinstance(extrinsic, dict):
            entry["extrinsic"] = extrinsic
        entry["reproj_errors"] = raw.get("reproj_errors", {}) if isinstance(raw.get("reproj_errors"), dict) else {}
        entry["extrinsic_errors"] = raw.get("extrinsic_errors", {}) if isinstance(raw.get("extrinsic_errors"), dict) else {}
        overrides = raw.get("overrides")
        if isinstance(overrides, list):
            entry["overrides"] = [item for item in overrides if isinstance(item, str)]
    return entry


def serialize_dataset_entry(entry: DatasetCache) -> DatasetCache:
    return {
        "marks": entry.get("marks", {}),
        "signatures": entry.get("signatures", {}),
        "calibration": entry.get("calibration", {}),
        "matrices": entry.get("matrices", {}),
        "extrinsic": entry.get("extrinsic", {}),
        "reproj_errors": entry.get("reproj_errors", {}),
        "extrinsic_errors": entry.get("extrinsic_errors", {}),
        "overrides": [item for item in entry.get("overrides", []) if isinstance(item, str)],
    }


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
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)  # type: ignore[arg-type]
    except (OSError, yaml.YAMLError):  # noqa: PERF203
        return _default_cache()
    cache = _normalize_cache(raw)
    cache["version"] = CACHE_VERSION
    return cache


def save_cache(cache: CacheData) -> None:
    cache["version"] = CACHE_VERSION
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as handle:
            yaml.safe_dump(cache, handle, allow_unicode=False, sort_keys=True)
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


def trim_cache(cache: CacheData, max_entries: int = MAX_DATASETS) -> None:
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

