"""High-level cache and calibration persistence helpers."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import yaml

from backend.utils.cache import (
    CacheData,
    DatasetCache,
    ensure_dataset_entry,
    load_cache,
    save_cache,
    serialize_dataset_entry,
    deserialize_dataset_entry,
    touch_dataset,
    trim_cache,
)
from common.log_utils import log_debug, log_info, log_warning, log_perf

if TYPE_CHECKING:
    from backend.services.viewer_state import ViewerState


# Image labels cache (marks, auto_marks, reason_counts, overrides, archived)
DATASET_CACHE_FILENAME = ".image_labels.yaml"  # Hidden file


def _is_collection(dataset_path: Path) -> bool:
    """Check if a dataset path is actually a collection (has child datasets)."""
    if not dataset_path.exists() or not dataset_path.is_dir():
        return False

    # Check if any subdirectory is a dataset (has lwir and visible folders)
    for entry in dataset_path.iterdir():
        if not entry.is_dir():
            continue
        if (entry / "lwir").exists() and (entry / "visible").exists():
            return True
    return False


def _distribute_collection_marks(dataset_path: Path, entry: DatasetCache) -> None:
    """Distribute collection marks to child datasets based on base name prefixes.

    Collection marks have prefixed bases like "child_name/base".
    This extracts them and saves to the appropriate child's cache.
    """
    marks = entry.get("marks", {})
    auto_marks_dict = entry.get("auto_marks", {})

    if not marks and not auto_marks_dict:
        return

    debug = os.environ.get("DEBUG_CACHE", "").lower() in {"1", "true", "on"}

    # Group marks by child dataset
    child_marks: Dict[str, Dict[str, str]] = {}
    child_auto_marks: Dict[str, Dict[str, Set[str]]] = {}

    # Process regular marks
    for base, reason in marks.items():
        if "/" not in base:
            continue  # Not a prefixed mark, skip
        child_name, local_base = base.split("/", 1)
        if child_name not in child_marks:
            child_marks[child_name] = {}
        child_marks[child_name][local_base] = reason

    # Process auto_marks
    for reason, bases in auto_marks_dict.items():
        if not isinstance(bases, (list, set)):
            continue
        for base in bases:
            if "/" not in base:
                continue
            child_name, local_base = base.split("/", 1)
            if child_name not in child_auto_marks:
                child_auto_marks[child_name] = {}
            if reason not in child_auto_marks[child_name]:
                child_auto_marks[child_name][reason] = set()
            child_auto_marks[child_name][reason].add(local_base)

    # Save to each child's cache
    for child_name in set(child_marks.keys()) | set(child_auto_marks.keys()):
        child_path = dataset_path / child_name
        if not child_path.exists():
            if debug:
                log_warning(f"Collection child path not found: {child_path}", "CACHE")
            continue

        child_cache_path = child_path / DATASET_CACHE_FILENAME
        child_entry = load_dataset_cache_file(child_cache_path)

        # Track if we actually make changes
        has_changes = False

        # Merge marks (collection marks override child marks for same base)
        existing_marks = child_entry.get("marks", {})
        if isinstance(existing_marks, dict):
            existing_marks = dict(existing_marks)
        else:
            existing_marks = {}

        marks_to_add = child_marks.get(child_name, {})
        # Check if marks would actually add something new
        for base, reason in marks_to_add.items():
            if base not in existing_marks or existing_marks[base] != reason:
                has_changes = True
                break

        if has_changes:
            existing_marks.update(marks_to_add)
            child_entry["marks"] = existing_marks

        # Merge auto_marks
        existing_auto = child_entry.get("auto_marks", {})
        if not isinstance(existing_auto, dict):
            existing_auto = {}
        else:
            # Deserialize if needed
            existing_auto = {
                r: set(b) if isinstance(b, list) else b
                for r, b in existing_auto.items()
            }

        auto_to_add = child_auto_marks.get(child_name, {})
        # Check if auto_marks would add something new
        for reason, bases in auto_to_add.items():
            if reason not in existing_auto:
                has_changes = True
            elif not bases.issubset(existing_auto[reason]):
                has_changes = True
            if has_changes:
                break

        if has_changes:
            for reason, bases in auto_to_add.items():
                if reason in existing_auto:
                    existing_auto[reason].update(bases)
                else:
                    existing_auto[reason] = bases.copy()
            child_entry["auto_marks"] = serialize_auto_marks(existing_auto)

        # Only save and notify if there were actual changes
        if not has_changes:
            if debug:
                log_debug(f"Child {child_name}: no new marks to distribute (already has them)", "CACHE")
            continue

        # Update reason_counts
        reason_counts = child_entry.get("reason_counts", {})
        if not isinstance(reason_counts, dict):
            reason_counts = {}

        for reason in set(marks_to_add.values()):
            reason_counts[reason] = reason_counts.get(reason, 0) + sum(
                1 for r in marks_to_add.values() if r == reason
            )

        child_entry["reason_counts"] = reason_counts

        # Save child cache
        save_dataset_cache_file(child_cache_path, child_entry)

        # Notify handler registry that child cache was updated
        from backend.services.handler_registry import get_handler_registry
        get_handler_registry().notify_cache_changed(child_path)

        if debug:
            mark_count = len(marks_to_add)
            auto_count = sum(len(bases) for bases in auto_to_add.values())
            log_info(f"Distributed to {child_name}: {mark_count} marks ({auto_count} auto)", "CACHE")


@dataclass(frozen=True)
class CachePersistPayload:
    cache_data: CacheData
    dataset_cache_path: Optional[Path]
    dataset_entry: DatasetCache


class CacheService:
    """Encapsulate signature/calibration persistence and dataset bookkeeping."""

    def __init__(self) -> None:
        t = time.perf_counter()
        self._cache: CacheData = load_cache()
        log_perf(f"load_cache {time.perf_counter() - t:.3f}s")

        self._dataset_path: Optional[str] = None
        self._dataset_cache_path: Optional[Path] = None
        self._dataset_entry: DatasetCache = empty_dataset_entry()
        self._workspace_root: Optional[Path] = None  # Workspace root to exclude from recent
        self.active_cache: Dict[str, Any] = {}  # In-memory snapshot of current dataset state
        if "dataset_kinds" not in self._cache or not isinstance(self._cache.get("dataset_kinds"), dict):
            self._cache["dataset_kinds"] = {}

    def set_workspace_root(self, workspace_root: Optional[Path]) -> None:
        """Set workspace root to exclude from recent history."""
        self._workspace_root = workspace_root

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def set_active_dataset(self, dataset_path: Optional[Path]) -> None:
        self._dataset_path = str(dataset_path) if dataset_path else None
        if self._dataset_path:
            # Don't record workspace root in recent history
            if self._workspace_root and dataset_path and dataset_path == self._workspace_root:
                log_debug(f"Skipping workspace root {dataset_path.name} from recent history", "CACHE")
                self._dataset_cache_path = Path(self._dataset_path) / DATASET_CACHE_FILENAME
                self._dataset_entry = load_dataset_cache_file(self._dataset_cache_path)
                return

            # Determine what to record in recent history
            # Priority: If loading a collection directly, record it.
            # If loading a child dataset, record its parent collection.
            path_to_record = self._dataset_path
            if dataset_path:
                parent = dataset_path.parent
                is_parent_workspace = self._workspace_root and parent == self._workspace_root

                # Case 1: Loading a collection directly (current path IS a collection)
                if _is_collection(dataset_path):
                    path_to_record = str(dataset_path)
                    log_debug(f"Recording collection {dataset_path.name} (loaded directly)", "CACHE")
                # Case 2: Loading a child dataset whose parent is a collection
                elif parent and parent != dataset_path and _is_collection(parent) and not is_parent_workspace:
                    path_to_record = str(parent)
                    log_debug(f"Recording collection {parent.name} instead of child {dataset_path.name}", "CACHE")
                # Case 3: Loading a standalone dataset
                else:
                    log_debug(f"Recording standalone dataset {dataset_path.name}", "CACHE")

            self._cache["last_dataset"] = path_to_record
            ensure_dataset_entry(self._cache, path_to_record)
            touch_dataset(self._cache, path_to_record)
            trim_cache(self._cache)
            self._dataset_cache_path = Path(self._dataset_path) / DATASET_CACHE_FILENAME
            self._dataset_entry = load_dataset_cache_file(self._dataset_cache_path)

            # If image_labels.yaml doesn't exist but .summary_cache.yaml does, load sweep_flags from summary cache
            if not self._dataset_entry or not self._dataset_entry.get("sweep_flags"):
                from backend.services.dataset_handler import SUMMARY_CACHE_FILENAME
                summary_cache_path = Path(self._dataset_path) / SUMMARY_CACHE_FILENAME
                if summary_cache_path.exists():
                    summary_data = load_dataset_cache_file(summary_cache_path)
                    if summary_data and isinstance(summary_data, dict):
                        # Merge sweep_flags from summary into entry
                        sweep_flags = summary_data.get("sweep_flags", {})
                        if sweep_flags and any(sweep_flags.values()):  # Only if there are actual flags set
                            if not self._dataset_entry:
                                self._dataset_entry = empty_dataset_entry()
                            if not self._dataset_entry.get("sweep_flags"):
                                self._dataset_entry["sweep_flags"] = {}
                            self._dataset_entry["sweep_flags"].update(sweep_flags)
                            log_debug(f"Loaded sweep_flags from summary: {sweep_flags}", "CACHE")

            # Persist changes to recent datasets
            self.save()
        else:
            self._dataset_cache_path = None
            self._dataset_entry = empty_dataset_entry()

    def last_dataset(self) -> Optional[str]:
        value = self._cache.get("last_dataset")
        return value if isinstance(value, str) else None

    def record_dataset_kind(self, dataset_path: Path, kind: str) -> None:
        kinds = self._cache.get("dataset_kinds")
        if not isinstance(kinds, dict):
            kinds = {}
            self._cache["dataset_kinds"] = kinds
        kinds[str(dataset_path)] = kind
        self.save()  # Persist changes

    def dataset_kind(self, dataset_path: Path) -> str:
        """Get dataset kind. Returns 'dataset', 'collection', or 'unknown'."""
        kinds = self._cache.get("dataset_kinds")
        if not isinstance(kinds, dict):
            return "unknown"
        value = kinds.get(str(dataset_path))
        return value if isinstance(value, str) else "unknown"

    def set_dataset_kind(self, dataset_path: Path, kind: str) -> None:
        """Set dataset kind ('dataset' or 'collection')."""
        if "dataset_kinds" not in self._cache:
            self._cache["dataset_kinds"] = {}
        self._cache["dataset_kinds"][str(dataset_path)] = kind


    def _ensure_preferences(self) -> Dict[str, Any]:
        prefs = self._cache.get("preferences")
        if not isinstance(prefs, dict):
            prefs = {}
            self._cache["preferences"] = prefs
        return prefs

    def get_preferences(self) -> Dict[str, Any]:
        return dict(self._ensure_preferences())

    def get_preference(self, key: str, default: Any = None) -> Any:
        prefs = self._ensure_preferences()
        return prefs.get(key, default)

    def set_preference(self, key: str, value: Any) -> None:
        self.set_preferences(**{key: value})

    def set_preferences(self, **kwargs: Any) -> None:
        prefs = self._ensure_preferences()
        changed = False
        for key, value in kwargs.items():
            if prefs.get(key) == value:
                continue
            prefs[key] = value
            changed = True
        if changed:
            self.save()

    def clear_dataset_cache(self, dataset_path: Path) -> None:
        path_str = str(dataset_path)
        datasets = self._cache.get("datasets")
        if isinstance(datasets, dict):
            datasets.pop(path_str, None)
        cache_file = dataset_path / DATASET_CACHE_FILENAME
        try:
            cache_file.unlink()
        except OSError:
            pass
        if self._dataset_path == path_str:
            self._dataset_entry = empty_dataset_entry()
        self.save()

    def save(self) -> None:
        save_cache(self._cache)

    def recent_datasets(self, limit: int = 5) -> List[str]:
        datasets = self._cache.get("datasets")
        if not isinstance(datasets, dict):
            return []
        ordered = list(datasets.keys())

        # Filter: only collections and standalone datasets (not children of collections)
        filtered = []
        for path_str in ordered:
            path = Path(path_str)
            if not path.exists():
                continue
            parent = path.parent
            is_parent_workspace = self._workspace_root and parent == self._workspace_root
            if parent and _is_collection(parent) and not is_parent_workspace:
                continue
            if self._workspace_root and path == self._workspace_root:
                continue
            filtered.append(path_str)

        return list(reversed(filtered[-limit:]))

    # ------------------------------------------------------------------
    # Dataset snapshots
    # ------------------------------------------------------------------
    def load_dataset_entry(self) -> DatasetCache:
        if not self._dataset_path:
            return empty_dataset_entry()
        return self._dataset_entry

    def snapshot_state(
        self,
        state: "ViewerState",
        archived_entries: Dict[str, Dict[str, Any]],
        total_pairs: int,
    ) -> None:
        """Snapshot state to cache. Uses ViewerState.cache_data as single source of truth."""
        cd = state.cache_data

        # Build active_cache for runtime access
        self.active_cache = {
            "marks": cd["marks"],
            "reason_counts": cd["reason_counts"],
            "auto_marks": {k: list(v) for k, v in cd["auto_marks"].items()},
            "calibration_marked": list(state.calibration_marked),
            "calibration_results": state.calibration_results,
            "calibration_corners": state.calibration_corners,
            "reproj_errors": cd["reproj_errors"],
            "extrinsic_errors": cd["extrinsic_errors"],
            "overrides": list(cd["overrides"]),
            "archived": archived_entries,
            "signatures": self.active_cache.get("signatures", {}),
            "total_pairs": total_pairs,
            "sweep_flags": cd.get("sweep_flags", {}),
        }

        if not self._dataset_path:
            return

        entry = ensure_dataset_entry(self._cache, self._dataset_path)
        if isinstance(self._dataset_entry, dict) and isinstance(self._dataset_entry.get("note"), str):
            entry["note"] = self._dataset_entry.get("note", "")
        self._write_snapshot_entry(entry, state, archived_entries, total_pairs)
        touch_dataset(self._cache, self._dataset_path)
        trim_cache(self._cache)
        self._cache["last_dataset"] = self._dataset_path

        if self._dataset_cache_path:
            dataset_entry = deepcopy(self._dataset_entry) if self._dataset_entry else empty_dataset_entry()
            self._write_snapshot_entry(dataset_entry, state, archived_entries, total_pairs)
            self._dataset_entry = dataset_entry

    def _write_snapshot_entry(
        self,
        entry: DatasetCache,
        state: "ViewerState",
        archived_entries: Dict[str, Dict[str, Any]],
        total_pairs: int,
    ) -> None:
        """Write state to entry dict. Extracts data from ViewerState.cache_data."""
        cd = state.cache_data

        # Marks
        entry["marks"] = {k: v.strip() for k, v in cd["marks"].items()
                          if isinstance(k, str) and isinstance(v, str) and v.strip()}
        entry["reason_counts"] = {k: int(v) for k, v in cd["reason_counts"].items()
                                  if isinstance(k, str) and isinstance(v, (int, float))}
        entry["auto_counts"] = {k: len(v) for k, v in cd["auto_marks"].items()
                                if isinstance(k, str) and isinstance(v, set)}
        entry["auto_marks"] = serialize_auto_marks(cd["auto_marks"])

        # Log manual vs auto breakdown for debugging
        manual_count = len(cd["marks"]) - sum(len(bases) for bases in cd["auto_marks"].values())
        auto_count = sum(len(bases) for bases in cd["auto_marks"].values())
        if manual_count > 0 or auto_count > 0:
            log_info(f"Saving marks: {len(cd['marks'])} total ({manual_count} manual, {auto_count} auto)", "CACHE")

        # Calibration (serialize the whole calibration dict, preserving auto flag)
        entry["calibration"] = serialize_calibration(cd.get("calibration", {}))

        # Errors (direct dict access)
        entry["reproj_errors"] = {
            ch: {b: float(e) for b, e in bucket.items()
                 if isinstance(b, str) and isinstance(e, (int, float))}
            for ch in ("lwir", "visible")
            if (bucket := cd["reproj_errors"].get(ch, {})) and isinstance(bucket, dict)
        }
        entry["extrinsic_errors"] = {b: float(e) for b, e in cd["extrinsic_errors"].items()
                                      if isinstance(b, str) and isinstance(e, (int, float))}

        # Other fields
        entry["overrides"] = sorted(cd["overrides"])
        entry["archived"] = serialize_archived_entries(archived_entries)
        entry["total_pairs"] = int(total_pairs)

        sweep_flags = cd.get("sweep_flags")
        if sweep_flags:
            entry["sweep_flags"] = dict(sweep_flags)


    def build_persist_payload(self) -> CachePersistPayload:
        cache_copy = deepcopy(self._cache)
        entry_copy = deepcopy(self._dataset_entry)
        return CachePersistPayload(cache_copy, self._dataset_cache_path, entry_copy)

    def get_last_collection(self) -> Optional[str]:
        value = self.get_preference("last_collection")
        return value if isinstance(value, str) else None

    def set_last_collection(self, collection_path: Optional[Path]) -> None:
        self.set_preference("last_collection", str(collection_path) if collection_path else None)

    def recent_collections(self, limit: int = 5) -> List[str]:
        raw = self.get_preference("recent_collections", [])
        if not isinstance(raw, list):
            return []
        entries = [p for p in raw if isinstance(p, str)]
        return entries[:limit]

    def touch_collection(self, collection_path: Path, *, limit: int = 10) -> None:
        prefs = self._ensure_preferences()
        raw = prefs.get("recent_collections")
        entries: List[str] = []
        if isinstance(raw, list):
            entries = [p for p in raw if isinstance(p, str)]
        path_str = str(collection_path)
        entries = [p for p in entries if p != path_str]
        entries.insert(0, path_str)
        entries = entries[: max(1, int(limit))]
        prefs["recent_collections"] = entries
        prefs["last_collection"] = path_str
        self.save()


# ----------------------------------------------------------------------
# Serialization helpers
# ----------------------------------------------------------------------
# TODO(refactor): Reduce serialization to only complex types (bytes, numpy)
# Simple dicts (marks, reason_counts, etc) should be used directly with type validation


def _serialize_bytes_to_hex(data: Optional[bytes]) -> Optional[str]:
    """Helper: Serialize bytes to hex string (only complex type serialization needed)."""
    return data.hex() if data is not None else None


def _deserialize_hex_to_bytes(data: Any) -> Optional[bytes]:
    """Helper: Deserialize hex string to bytes."""
    if isinstance(data, str):
        try:
            return bytes.fromhex(data)
        except ValueError:
            return None
    return None


def serialize_signatures(signatures: Dict[str, Dict[str, Optional[bytes]]]) -> Dict[str, Dict[str, Optional[str]]]:
    """✅ NECESSARY: Handles bytes type (bytes -> hex)."""
    payload: Dict[str, Dict[str, Optional[str]]] = {}
    for base, bucket in signatures.items():
        payload[base] = {
            type_dir: _serialize_bytes_to_hex(signature)
            for type_dir, signature in bucket.items()
        }
    return payload


def deserialize_signatures(raw: Any) -> Dict[str, Dict[str, Optional[bytes]]]:
    """✅ NECESSARY: Handles bytes type (hex -> bytes)."""
    cache: Dict[str, Dict[str, Optional[bytes]]] = {}
    if not isinstance(raw, dict):
        return cache
    for base, bucket in raw.items():
        if not isinstance(base, str) or not isinstance(bucket, dict):
            continue
        decoded: Dict[str, Optional[bytes]] = {}
        for type_dir, value in bucket.items():
            if not isinstance(type_dir, str):
                continue
            decoded[type_dir] = _deserialize_hex_to_bytes(value) if value else None
        if decoded:
            cache[base] = decoded
    return cache


def serialize_auto_marks(auto_marks: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """⚠️ SEMI-NECESSARY: Set -> List for YAML compatibility (trivial conversion)."""
    if not isinstance(auto_marks, dict):
        return {}
    return {k: list(v) for k, v in auto_marks.items() if isinstance(k, str) and isinstance(v, set)}


def deserialize_auto_marks(raw: Any) -> Dict[str, Set[str]]:
    """⚠️ SEMI-NECESSARY: List -> Set for YAML compatibility (trivial conversion)."""
    if not isinstance(raw, dict):
        return {}
    result: Dict[str, Set[str]] = {}
    for reason, bases in raw.items():
        if isinstance(reason, str) and isinstance(bases, list):
            result[reason] = {b for b in bases if isinstance(b, str)}
    return result


def serialize_archived_entries(entries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Serialize archived entries. Only corners need conversion (numpy arrays), rest passes through."""
    payload: Dict[str, Any] = {}
    for base, data in entries.items():
        if not isinstance(base, str) or not isinstance(data, dict):
            continue
        # Pass through all fields, only serialize corners if present
        entry = dict(data)  # shallow copy
        if "calibration_corners" in entry:
            entry["calibration_corners"] = _serialize_corners_bucket(entry.get("calibration_corners"))
        payload[base] = entry
    return payload


def deserialize_archived_entries(raw: Any) -> Dict[str, Dict[str, Any]]:
    """Deserialize archived entries. Only corners need conversion (numpy arrays), rest passes through."""
    if not isinstance(raw, dict):
        return {}
    restored: Dict[str, Dict[str, Any]] = {}
    for base, data in raw.items():
        if not isinstance(base, str) or not isinstance(data, dict):
            continue
        # Pass through all fields, only deserialize corners if present
        entry = dict(data)  # shallow copy
        if "calibration_corners" in entry:
            entry["calibration_corners"] = _deserialize_corners_bucket(entry.get("calibration_corners"))
        restored[base] = entry
    return restored


def serialize_calibration(
    calibration_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert ViewerState calibration dict to YAML per-base dict structure.

    Preserves: marked, auto, outlier_*, results
    NOTE: Corners deliberately excluded - stored in calibration/*.yaml files separately.
    """
    payload: Dict[str, Any] = {}

    if not isinstance(calibration_dict, dict):
        return payload

    for base, entry in calibration_dict.items():
        if not isinstance(base, str) or not isinstance(entry, dict):
            continue
        # Only serialize if there's meaningful data
        if not entry.get("marked") and not entry.get("results"):
            continue
        payload[base] = {
            "marked": bool(entry.get("marked", False)),
            "auto": bool(entry.get("auto", False)),  # Preserve auto flag
            "outlier_lwir": bool(entry.get("outlier_lwir", False)),
            "outlier_visible": bool(entry.get("outlier_visible", False)),
            "outlier_stereo": bool(entry.get("outlier_stereo", False)),
            "results": entry.get("results", {}),
        }
    return payload


def deserialize_calibration(raw: Any) -> Dict[str, Dict[str, Any]]:
    """Convert YAML per-base dict structure to ViewerState calibration dict.

    Returns dict with structure: {base: {marked, auto, outlier_*, results}}
    NOTE: Corners always loaded on-demand from calibration/*.yaml files.
    """
    calibration: Dict[str, Dict[str, Any]] = {}

    if not isinstance(raw, dict):
        return calibration

    for base, entry in raw.items():
        if not isinstance(base, str) or not isinstance(entry, dict):
            continue

        # Extract results if valid
        entry_results = entry.get("results", {})
        results = {}
        if isinstance(entry_results, dict):
            results = {
                k: v for k, v in entry_results.items()
                if k in ("lwir", "visible") and (isinstance(v, bool) or v is None)
            }

        calibration[base] = {
            "marked": bool(entry.get("marked", False)),
            "auto": bool(entry.get("auto", False)),  # Preserve auto flag (default False for legacy)
            "outlier_lwir": bool(entry.get("outlier_lwir", False)),
            "outlier_visible": bool(entry.get("outlier_visible", False)),
            "outlier_stereo": bool(entry.get("outlier_stereo", False)),
            "results": results,
        }

    return calibration


def _serialize_corners_bucket(
    bucket: Optional[Dict[str, Optional[List[List[float]]]]],
) -> Dict[str, Any]:
    if not isinstance(bucket, dict):
        return {}
    payload: Dict[str, Any] = {}
    for type_dir, points in bucket.items():
        if type_dir not in {"lwir", "visible"}:
            continue
        if not points:
            payload[type_dir] = []
            continue
        serialized: List[List[float]] = []
        for pair in points:
            if (
                isinstance(pair, list)
                and len(pair) == 2
                and all(isinstance(coord, (int, float)) for coord in pair)
            ):
                serialized.append([float(pair[0]), float(pair[1])])
        payload[type_dir] = serialized
    return payload


def _deserialize_corners_bucket(raw: Any) -> Dict[str, Optional[List[List[float]]]]:
    """Deserialize corner points. Uses List[List[float]] instead of tuples for YAML compatibility."""
    if not isinstance(raw, dict):
        return {}
    bucket: Dict[str, Optional[List[List[float]]]] = {}
    for type_dir, value in raw.items():
        if type_dir not in {"lwir", "visible"}:
            continue
        if value is None:
            bucket[type_dir] = None
            continue
        if not isinstance(value, list):
            continue
        points: List[List[float]] = []
        for pair in value:
            if (
                isinstance(pair, list)
                and len(pair) == 2
                and all(isinstance(coord, (int, float)) for coord in pair)
            ):
                points.append([float(pair[0]), float(pair[1])])
        bucket[type_dir] = points
    return bucket


def serialize_matrices(matrices: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """✅ NECESSARY: Handles numpy array serialization for camera matrices."""
    payload: Dict[str, Any] = {}
    for key, data in matrices.items():
        if data:
            entry: Dict[str, Any] = {
                "camera_matrix": data.get("camera_matrix"),
                "distortion": data.get("distortion"),
            }
            if data.get("image_size"):
                entry["image_size"] = data.get("image_size")
            if data.get("samples") is not None:
                entry["samples"] = data.get("samples")
            if data.get("reprojection_error") is not None:
                entry["reprojection_error"] = data.get("reprojection_error")
            payload[key] = entry
    return payload


def deserialize_matrices(raw: Any) -> Dict[str, Optional[Dict[str, Any]]]:
    """✅ NECESSARY: Handles numpy array deserialization for camera matrices."""
    matrices: Dict[str, Optional[Dict[str, Any]]] = {"lwir": None, "visible": None}
    if not isinstance(raw, dict):
        return matrices
    for key in matrices.keys():
        entry = raw.get(key)
        if not isinstance(entry, dict):
            continue
        camera = entry.get("camera_matrix")
        distortion = entry.get("distortion")
        if is_matrix3x3(camera) and is_distortion_vector(distortion):
            matrix_data: Dict[str, Any] = {
                "camera_matrix": camera,
                "distortion": distortion,
            }
            if entry.get("image_size"):
                matrix_data["image_size"] = entry.get("image_size")
            if entry.get("samples") is not None:
                matrix_data["samples"] = entry.get("samples")
            if entry.get("reprojection_error") is not None:
                matrix_data["reprojection_error"] = entry.get("reprojection_error")
            matrices[key] = matrix_data
    return matrices



def is_matrix3x3(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False
    return all(
        isinstance(row, (list, tuple))
        and len(row) == 3
        and all(isinstance(num, (int, float)) for num in row)
        for row in value
    )


def is_distortion_vector(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or not value:
        return False
    return all(isinstance(num, (int, float)) for num in value)


def is_vector3(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False
    return all(isinstance(num, (int, float)) for num in value)


def empty_dataset_entry() -> DatasetCache:
    return {
        "marks": {},
        "reason_counts": {},
        "auto_counts": {},
        "auto_marks": {},
        "signatures": {},
        "calibration": {},
        "matrices": {},
        "extrinsic": {},
        "reproj_errors": {},
        "extrinsic_errors": {},
        "overrides": [],
        "note": "",
        "archived": {},
        "total_pairs": 0,
        "sweep_flags": {
            "duplicates": False,
            "missing": False,
            "quality": False,
            "patterns": False,
        },
    }


def load_dataset_cache_file(cache_path: Path) -> DatasetCache:
    """Load dataset cache file."""
    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError):  # noqa: PERF203
        return empty_dataset_entry()
    return deserialize_dataset_entry(raw)


def save_dataset_cache_file(cache_path: Path, entry: DatasetCache) -> None:
    payload = serialize_dataset_entry(entry)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    try:
        with open(cache_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, allow_unicode=False, sort_keys=True)
    except OSError:
        return



__all__ = [
    "CachePersistPayload",
    "CacheService",
    "serialize_signatures",
    "deserialize_signatures",
    "serialize_calibration",
    "deserialize_calibration",
    "is_matrix3x3",
    "is_distortion_vector",
    "is_vector3",
    "empty_dataset_entry",
    "load_dataset_cache_file",
    "save_dataset_cache_file",
]
