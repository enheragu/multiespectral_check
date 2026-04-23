"""Label summary derivation — scan label YAML files on disk and produce counts.

Mirrors the pattern of ``summary_derivation.py`` for the main dataset cache.
The produced dict is written to ``.labels_summary_cache.yaml`` at dataset root
and can be aggregated across datasets for workspace-level reports.

Principle: the cache is **derived**, not authoritative.  Individual per-image
label YAML files under ``labels/{channel}/`` are the source of truth.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.log_utils import log_debug, log_info, log_warning
from common.yaml_utils import get_timestamp_fields, load_yaml, save_yaml

# ---------------------------------------------------------------------------
# Cache filename (hidden file at dataset root)
# ---------------------------------------------------------------------------
LABELS_SUMMARY_CACHE_FILENAME = ".labels_summary_cache.yaml"

# ---------------------------------------------------------------------------
# Types returned by derivation
# ---------------------------------------------------------------------------
LabelsSummaryDict = Dict[str, Any]

_CHANNELS = ("visible", "lwir")

# Chart histogram/grid parameters
_GRID_CELLS = 50       # N×N cells for position and size heatmaps
_MAX_BBOX_SAMPLES = 4000  # reservoir sample cap for bbox overlay plot
_SKIP_ATTRS = frozenset({"model", "model_version", "raw_label", "confidence", "source"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def derive_labels_summary_from_disk(
    dataset_path: Path,
    config_path: Optional[Path] = None,
) -> LabelsSummaryDict:
    """Scan ``labels/`` on disk and return a summary dict.

    The function reads every ``labels/{channel}/*.yaml`` file and aggregates
    counts **and** chart data (size histograms + position heatmap grid).
    It does **not** touch the in-memory ``LabelStorage`` cache so it can
    safely be called from background threads or when no service is active.

    Bbox values are already normalised [0,1] centre-xywh in the YAML files,
    so no image dimensions are needed.

    Args:
        dataset_path: Root of the dataset directory.
        config_path: Optional explicit path to the label config YAML.
            When given it is preferred over any automatic resolution.
    """
    labels_dir = dataset_path / "labels"
    summary = _empty_labels_summary()

    if not labels_dir.is_dir():
        _stamp(summary)
        return summary

    # Load class name map from workspace config (or explicit config_path)
    name_map = _load_class_name_map(labels_dir, config_path=config_path)

    for channel in _CHANNELS:
        channel_dir = labels_dir / channel
        if not channel_dir.is_dir():
            continue
        for yaml_path in sorted(channel_dir.rglob("*.yaml")):
            try:
                data = load_yaml(yaml_path)
            except Exception as exc:  # noqa: BLE001
                log_warning(f"Skipping unreadable label file {yaml_path}: {exc}", "LABEL_SUMMARY")
                continue
            if not isinstance(data, dict):
                continue
            annotations = data.get("annotations", [])
            if not annotations:
                continue

            summary["images_labeled"][channel] += 1

            for ann in annotations:
                if not isinstance(ann, dict):
                    continue
                source = ann.get("source", "manual")
                class_id = str(ann.get("class_id", "?"))
                attrs = ann.get("attributes", {})
                bbox = ann.get("bbox")  # [cx, cy, w, h] normalised

                summary["total_annotations"] += 1
                summary["by_channel"][channel] = summary["by_channel"].get(channel, 0) + 1
                summary["by_source"][source] = summary["by_source"].get(source, 0) + 1

                # Per-class bucket
                cls_bucket = summary["by_class"].setdefault(class_id, _empty_class_bucket())
                cls_bucket["name"] = name_map.get(class_id)
                cls_bucket["total"] += 1
                cls_bucket["by_source"][source] = cls_bucket["by_source"].get(source, 0) + 1
                cls_bucket["by_channel"][channel] = cls_bucket["by_channel"].get(channel, 0) + 1

                # Attribute value distribution (skip internal attrs)
                if isinstance(attrs, dict):
                    for attr_name, attr_val in attrs.items():
                        if attr_name in _SKIP_ATTRS or attr_val in (None, "", "unknown"):
                            continue
                        attr_bucket = cls_bucket["attributes"].setdefault(attr_name, {})
                        val_str = str(attr_val)
                        attr_bucket[val_str] = attr_bucket.get(val_str, 0) + 1

                # Chart data — 2-D grids + bbox samples
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    cx, cy, w, h = (float(v) for v in bbox)
                    idx = summary["total_annotations"]  # already incremented
                    _accumulate_charts(summary["charts"], cx, cy, w, h, idx)
                    cls_idx = cls_bucket["total"]        # already incremented
                    _accumulate_charts(cls_bucket["charts"], cx, cy, w, h, cls_idx)

    _stamp(summary)
    return summary


def load_labels_summary_cache(dataset_path: Path) -> Optional[LabelsSummaryDict]:
    """Load cached summary from disk (fast path).  Returns *None* on miss."""
    cache_path = dataset_path / LABELS_SUMMARY_CACHE_FILENAME
    if not cache_path.exists():
        return None
    try:
        data = load_yaml(cache_path)
        if isinstance(data, dict) and "total_annotations" in data:
            return data
    except Exception as exc:  # noqa: BLE001
        log_warning(f"Failed to load labels summary cache: {exc}", "LABEL_SUMMARY")
    return None


def save_labels_summary_cache(dataset_path: Path, summary: LabelsSummaryDict) -> None:
    """Persist summary to disk."""
    cache_path = dataset_path / LABELS_SUMMARY_CACHE_FILENAME
    try:
        save_yaml(cache_path, summary)
        log_debug(f"Saved labels summary cache for {dataset_path.name}", "LABEL_SUMMARY")
    except Exception as exc:  # noqa: BLE001
        log_warning(f"Failed to save labels summary cache: {exc}", "LABEL_SUMMARY")


def merge_labels_summaries(summaries: List[LabelsSummaryDict]) -> LabelsSummaryDict:
    """Merge multiple per-dataset summaries into one aggregate.

    Used for workspace-level reports — sums all numeric fields, merges
    per-class buckets, and sums chart histograms/grids element-wise.
    """
    merged = _empty_labels_summary()
    for s in summaries:
        merged["total_annotations"] += s.get("total_annotations", 0)
        for ch in _CHANNELS:
            merged["images_labeled"][ch] += s.get("images_labeled", {}).get(ch, 0)
            merged["by_channel"][ch] += s.get("by_channel", {}).get(ch, 0)
        for src in ("auto", "reviewed", "manual"):
            merged["by_source"][src] += s.get("by_source", {}).get(src, 0)
        # Global charts
        _merge_charts(merged["charts"], s.get("charts", {}))
        for class_id, cls_bucket in s.get("by_class", {}).items():
            dest = merged["by_class"].setdefault(class_id, _empty_class_bucket())
            # Prefer non-None name
            if cls_bucket.get("name") and not dest.get("name"):
                dest["name"] = cls_bucket["name"]
            dest["total"] += cls_bucket.get("total", 0)
            for src in ("auto", "reviewed", "manual"):
                dest["by_source"][src] += cls_bucket.get("by_source", {}).get(src, 0)
            for ch in _CHANNELS:
                dest["by_channel"][ch] += cls_bucket.get("by_channel", {}).get(ch, 0)
            for attr_name, val_counts in cls_bucket.get("attributes", {}).items():
                attr_dest = dest["attributes"].setdefault(attr_name, {})
                for val, cnt in val_counts.items():
                    attr_dest[val] = attr_dest.get(val, 0) + cnt
            # Per-class charts
            _merge_charts(dest["charts"], cls_bucket.get("charts", {}))
    _stamp(merged)
    return merged


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _empty_charts() -> Dict[str, Any]:
    """Return empty chart data structure.

    ``position_grid``
        *_GRID_CELLS × _GRID_CELLS* 2-D array counting bbox **centre**
        density.  Row = y-bin, Col = x-bin, both in normalised [0, 1).

    ``size_wh_grid``
        *_GRID_CELLS × _GRID_CELLS* 2-D array counting bbox **size**
        density.  Row = height-bin (top=small), Col = width-bin (left=small).

    ``bbox_samples``
        Up to *_MAX_BBOX_SAMPLES* randomly-sampled ``[cx, cy, w, h]``
        entries (reservoir sampling) for the overlay plot.
    """
    return {
        "position_grid": [[0] * _GRID_CELLS for _ in range(_GRID_CELLS)],
        "size_wh_grid": [[0] * _GRID_CELLS for _ in range(_GRID_CELLS)],
        "bbox_samples": [],
    }


def _empty_labels_summary() -> LabelsSummaryDict:
    return {
        "total_annotations": 0,
        "images_labeled": {"visible": 0, "lwir": 0},
        "by_channel": {"visible": 0, "lwir": 0},
        "by_source": {"auto": 0, "reviewed": 0, "manual": 0},
        "by_class": {},
        "charts": _empty_charts(),
    }


def _empty_class_bucket() -> Dict[str, Any]:
    return {
        "name": None,
        "total": 0,
        "by_source": {"auto": 0, "reviewed": 0, "manual": 0},
        "by_channel": {"visible": 0, "lwir": 0},
        "attributes": {},
        "charts": _empty_charts(),
    }


def _accumulate_charts(
    charts: Dict[str, Any],
    cx: float, cy: float, w: float, h: float,
    ann_index: int,
) -> None:
    """Accumulate one bbox into chart grids and bbox sample reservoir.

    Args:
        charts: The ``charts`` sub-dict to update (global or per-class).
        cx, cy: Normalised centre coordinates [0, 1].
        w, h: Normalised width/height [0, 1].
        ann_index: Running annotation counter (1-based) for reservoir sampling.
    """
    n = _GRID_CELLS

    # Position grid — centre density
    col = min(int(cx * n), n - 1)
    row = min(int(cy * n), n - 1)
    if 0 <= row < n and 0 <= col < n:
        charts["position_grid"][row][col] += 1

    # Size w×h grid — width on x, height on y
    wc = min(int(w * n), n - 1)
    hc = min(int(h * n), n - 1)
    if 0 <= hc < n and 0 <= wc < n:
        charts["size_wh_grid"][hc][wc] += 1

    # Bbox samples — reservoir sampling (Vitter's Algorithm R)
    samples = charts["bbox_samples"]
    entry = [round(cx, 4), round(cy, 4), round(w, 4), round(h, 4)]
    if len(samples) < _MAX_BBOX_SAMPLES:
        samples.append(entry)
    else:
        j = random.randint(0, ann_index - 1)
        if j < _MAX_BBOX_SAMPLES:
            samples[j] = entry


def _merge_charts(dest: Dict[str, Any], src: Dict[str, Any]) -> None:
    """Merge *src* chart bins into *dest* element-wise."""
    if not src:
        return
    # Merge 2-D grids (position_grid, size_wh_grid)
    for key in ("position_grid", "size_wh_grid"):
        s_grid = src.get(key, [])
        d_grid = dest.get(key)
        if not s_grid:
            continue
        if d_grid is None:
            dest[key] = [list(row) for row in s_grid]
            continue
        for r, s_row in enumerate(s_grid):
            if r < len(d_grid):
                for c, v in enumerate(s_row):
                    if c < len(d_grid[r]):
                        d_grid[r][c] += v
    # Merge bbox samples — concatenate then downsample if over cap
    s_samples = src.get("bbox_samples", [])
    d_samples = dest.setdefault("bbox_samples", [])
    d_samples.extend(s_samples)
    if len(d_samples) > _MAX_BBOX_SAMPLES:
        random.shuffle(d_samples)
        del d_samples[_MAX_BBOX_SAMPLES:]


def _stamp(d: Dict[str, Any]) -> None:
    ts = get_timestamp_fields()
    d["last_updated"] = ts["last_updated"]
    d["last_updated_str"] = ts["last_updated_str"]


def _load_class_name_map(
    labels_dir: Path,
    config_path: Optional[Path] = None,
) -> Dict[str, Optional[str]]:
    """Best-effort load of class id→name.

    Resolution order:
        1. Explicit *config_path* (if given and exists).
        2. Walk up from *labels_dir* looking for ``labels_config.yaml``
           (workspace-level config).
    """
    from backend.services.labels.label_storage import find_labels_config

    paths_to_try: list[Path] = []
    if config_path and config_path.exists():
        paths_to_try.append(config_path)
    ws_config = find_labels_config(labels_dir.parent)
    if ws_config:
        paths_to_try.append(ws_config)

    for path in paths_to_try:
        try:
            data = load_yaml(path)
            if not isinstance(data, dict):
                continue
            name_map: Dict[str, Optional[str]] = {}
            for cls_data in data.get("classes", []):
                if isinstance(cls_data, dict) and "id" in cls_data:
                    name_map[str(cls_data["id"])] = cls_data.get("name")
            names = data.get("names", {})
            if isinstance(names, dict):
                for id_key, name in names.items():
                    name_map[str(id_key)] = str(name)
            if name_map:
                return name_map
        except Exception:  # noqa: BLE001
            continue
    return {}
