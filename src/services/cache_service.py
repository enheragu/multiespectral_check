"""High-level cache and calibration persistence helpers."""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from utils.cache import (
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


DATASET_CACHE_FILENAME = ".reviewer_cache.yaml"


@dataclass(frozen=True)
class CachePersistPayload:
    cache_data: CacheData
    dataset_cache_path: Optional[Path]
    dataset_entry: DatasetCache


class CacheService:
    """Encapsulate signature/calibration persistence and dataset bookkeeping."""

    def __init__(self) -> None:
        self._cache: CacheData = load_cache()
        self._dataset_path: Optional[str] = None
        self._dataset_cache_path: Optional[Path] = None
        self._dataset_entry: DatasetCache = empty_dataset_entry()

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def set_active_dataset(self, dataset_path: Optional[Path]) -> None:
        self._dataset_path = str(dataset_path) if dataset_path else None
        if self._dataset_path:
            self._cache["last_dataset"] = self._dataset_path
            touch_dataset(self._cache, self._dataset_path)
            trim_cache(self._cache)
            self._dataset_cache_path = Path(self._dataset_path) / DATASET_CACHE_FILENAME
            self._dataset_entry = load_dataset_cache_file(self._dataset_cache_path)
        else:
            self._dataset_cache_path = None
            self._dataset_entry = empty_dataset_entry()

    def last_dataset(self) -> Optional[str]:
        value = self._cache.get("last_dataset")
        return value if isinstance(value, str) else None

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

    def save(self) -> None:
        save_cache(self._cache)

    # ------------------------------------------------------------------
    # Dataset snapshots
    # ------------------------------------------------------------------
    def load_dataset_entry(self) -> DatasetCache:
        if not self._dataset_path:
            return empty_dataset_entry()
        return self._dataset_entry

    def snapshot_state(
        self,
        marks: Dict[str, str],
        signatures: Dict[str, Dict[str, Optional[bytes]]],
        calibration_marked: Set[str],
        calibration_outliers: Set[str],
        calibration_results: Dict[str, Dict[str, Optional[bool]]],
        calibration_corners: Dict[str, Dict[str, Optional[List[Tuple[float, float]]]]],
        calibration_warnings: Dict[str, Dict[str, str]],
        matrices: Dict[str, Optional[Dict[str, Any]]],
        reproj_errors: Dict[str, Dict[str, float]],
        extrinsic_errors: Dict[str, float],
        overrides: Set[str],
        extrinsic: Optional[Dict[str, Any]],
    ) -> None:
        if not self._dataset_path:
            return
        entry = ensure_dataset_entry(self._cache, self._dataset_path)
        self._write_snapshot_entry(
            entry,
            marks,
            signatures,
            calibration_marked,
            calibration_outliers,
            calibration_results,
            calibration_corners,
            calibration_warnings,
            matrices,
            reproj_errors,
            extrinsic_errors,
            overrides,
            extrinsic,
        )
        touch_dataset(self._cache, self._dataset_path)
        trim_cache(self._cache)
        self._cache["last_dataset"] = self._dataset_path
        if self._dataset_cache_path:
            dataset_entry = empty_dataset_entry()
            self._write_snapshot_entry(
                dataset_entry,
                marks,
                signatures,
                calibration_marked,
                calibration_outliers,
                calibration_results,
                calibration_corners,
                calibration_warnings,
                matrices,
                reproj_errors,
                extrinsic_errors,
                overrides,
                extrinsic,
            )
            self._dataset_entry = dataset_entry

    def _write_snapshot_entry(
        self,
        entry: DatasetCache,
        marks: Dict[str, str],
        signatures: Dict[str, Dict[str, Optional[bytes]]],
        calibration_marked: Set[str],
        calibration_outliers: Set[str],
        calibration_results: Dict[str, Dict[str, Optional[bool]]],
        calibration_corners: Dict[str, Dict[str, Optional[List[Tuple[float, float]]]]],
        calibration_warnings: Dict[str, Dict[str, str]],
        matrices: Dict[str, Optional[Dict[str, Any]]],
        reproj_errors: Dict[str, Dict[str, float]],
        extrinsic_errors: Dict[str, float],
        overrides: Set[str],
        extrinsic: Optional[Dict[str, Any]],
    ) -> None:
        entry["marks"] = serialize_marks(marks)
        entry["signatures"] = serialize_signatures(signatures)
        entry["calibration"] = serialize_calibration(
            calibration_marked,
            calibration_outliers,
            calibration_results,
            calibration_corners,
            calibration_warnings,
        )
        entry["matrices"] = serialize_matrices(matrices)
        entry["extrinsic"] = serialize_extrinsic(extrinsic)
        entry["reproj_errors"] = serialize_reprojection_errors(reproj_errors)
        entry["extrinsic_errors"] = serialize_extrinsic_errors(extrinsic_errors)
        entry["overrides"] = sorted(overrides)

    def build_persist_payload(self) -> CachePersistPayload:
        cache_copy = deepcopy(self._cache)
        entry_copy = deepcopy(self._dataset_entry)
        return CachePersistPayload(cache_copy, self._dataset_cache_path, entry_copy)


# ----------------------------------------------------------------------
# Serialization helpers
# ----------------------------------------------------------------------


def serialize_signatures(signatures: Dict[str, Dict[str, Optional[bytes]]]) -> Dict[str, Dict[str, Optional[str]]]:
    payload: Dict[str, Dict[str, Optional[str]]] = {}
    for base, bucket in signatures.items():
        payload[base] = {
            type_dir: signature.hex() if signature is not None else None
            for type_dir, signature in bucket.items()
        }
    return payload


def serialize_marks(marks: Dict[str, str]) -> Dict[str, str]:
    payload: Dict[str, str] = {}
    for base, reason in marks.items():
        if not isinstance(base, str) or not isinstance(reason, str):
            continue
        trimmed = reason.strip()
        if not trimmed:
            continue
        payload[base] = trimmed
    return payload


def deserialize_signatures(raw: Any) -> Dict[str, Dict[str, Optional[bytes]]]:
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
            if isinstance(value, str):
                try:
                    decoded[type_dir] = bytes.fromhex(value)
                except ValueError:
                    decoded[type_dir] = None
            elif value is None:
                decoded[type_dir] = None
        if decoded:
            cache[base] = decoded
    return cache


def deserialize_marks(raw: Any) -> Dict[str, str]:
    marks: Dict[str, str] = {}
    if not isinstance(raw, dict):
        return marks
    for base, reason in raw.items():
        if not isinstance(base, str) or not isinstance(reason, str):
            continue
        trimmed = reason.strip()
        if not trimmed:
            continue
        marks[base] = trimmed
    return marks


def serialize_calibration(
    calibration_marked: Set[str],
    calibration_outliers: Set[str],
    calibration_results: Dict[str, Dict[str, Optional[bool]]],
    calibration_corners: Dict[str, Dict[str, Optional[List[Tuple[float, float]]]]],
    calibration_warnings: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    bases = (
        set(calibration_marked)
        | set(calibration_outliers)
        | set(calibration_results.keys())
        | set(calibration_corners.keys())
        | set(calibration_warnings.keys())
    )
    for base in bases:
        entry: Dict[str, Any] = {
            "marked": base in calibration_marked,
            "outlier": base in calibration_outliers,
            "results": calibration_results.get(base, {}),
            "corners": _serialize_corners_bucket(calibration_corners.get(base)),
        }
        warnings = calibration_warnings.get(base)
        if isinstance(warnings, dict) and warnings:
            entry["warnings"] = {k: str(v) for k, v in warnings.items() if isinstance(k, str) and isinstance(v, str)}
        payload[base] = entry
    return payload


def deserialize_calibration(
    raw: Any,
) -> Tuple[
    Set[str],
    Set[str],
    Dict[str, Dict[str, Optional[bool]]],
    Dict[str, Dict[str, Optional[List[Tuple[float, float]]]]],
    Dict[str, Dict[str, str]],
]:
    marked: Set[str] = set()
    outliers: Set[str] = set()
    results: Dict[str, Dict[str, Optional[bool]]] = {}
    corners: Dict[str, Dict[str, Optional[List[Tuple[float, float]]]]] = {}
    warnings: Dict[str, Dict[str, str]] = {}
    if not isinstance(raw, dict):
        return marked, outliers, results, corners, warnings
    for base, entry in raw.items():
        if not isinstance(base, str) or not isinstance(entry, dict):
            continue
        if entry.get("marked"):
            marked.add(base)
        if entry.get("outlier"):
            outliers.add(base)
        entry_results = entry.get("results", {})
        if isinstance(entry_results, dict):
            filtered: Dict[str, Optional[bool]] = {}
            for type_dir, value in entry_results.items():
                if type_dir not in {"lwir", "visible"}:
                    continue
                if isinstance(value, bool) or value is None:
                    filtered[type_dir] = value
            if filtered:
                results[base] = filtered
        bucket = _deserialize_corners_bucket(entry.get("corners"))
        if bucket:
            corners[base] = bucket
        entry_warnings = entry.get("warnings")
        if isinstance(entry_warnings, dict):
            filtered: Dict[str, str] = {}
            for type_dir, message in entry_warnings.items():
                if type_dir not in {"lwir", "visible"}:
                    continue
                if isinstance(message, str) and message.strip():
                    filtered[type_dir] = message.strip()
            if filtered:
                warnings[base] = filtered
    return marked, outliers, results, corners, warnings


def serialize_reprojection_errors(errors: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    payload: Dict[str, Dict[str, float]] = {}
    for channel in ("lwir", "visible"):
        bucket = errors.get(channel, {}) if isinstance(errors, dict) else {}
        if not isinstance(bucket, dict):
            continue
        filtered = {
            base: float(err)
            for base, err in bucket.items()
            if isinstance(base, str) and isinstance(err, (int, float))
        }
        if filtered:
            payload[channel] = filtered
    return payload


def deserialize_reprojection_errors(raw: Any) -> Dict[str, Dict[str, float]]:
    errors: Dict[str, Dict[str, float]] = {"lwir": {}, "visible": {}}
    if not isinstance(raw, dict):
        return errors
    for channel in ("lwir", "visible"):
        bucket = raw.get(channel)
        if not isinstance(bucket, dict):
            continue
        filtered = {
            base: float(err)
            for base, err in bucket.items()
            if isinstance(base, str) and isinstance(err, (int, float))
        }
        errors[channel] = filtered
    return errors


def serialize_extrinsic_errors(errors: Dict[str, float]) -> Dict[str, float]:
    if not isinstance(errors, dict):
        return {}
    return {
        base: float(err)
        for base, err in errors.items()
        if isinstance(base, str) and isinstance(err, (int, float))
    }


def deserialize_extrinsic_errors(raw: Any) -> Dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    return {
        base: float(err)
        for base, err in raw.items()
        if isinstance(base, str) and isinstance(err, (int, float))
    }


def _serialize_corners_bucket(
    bucket: Optional[Dict[str, Optional[List[Tuple[float, float]]]]],
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
                isinstance(pair, (list, tuple))
                and len(pair) == 2
                and all(isinstance(coord, (int, float)) for coord in pair)
            ):
                u, v = float(pair[0]), float(pair[1])
                serialized.append([u, v])
        payload[type_dir] = serialized
    return payload


def _deserialize_corners_bucket(raw: Any) -> Dict[str, Optional[List[Tuple[float, float]]]]:
    if not isinstance(raw, dict):
        return {}
    bucket: Dict[str, Optional[List[Tuple[float, float]]]] = {}
    for type_dir, value in raw.items():
        if type_dir not in {"lwir", "visible"}:
            continue
        if value is None:
            bucket[type_dir] = None
            continue
        if not isinstance(value, list):
            continue
        points: List[Tuple[float, float]] = []
        for pair in value:
            if (
                isinstance(pair, (list, tuple))
                and len(pair) == 2
                and all(isinstance(coord, (int, float)) for coord in pair)
            ):
                points.append((float(pair[0]), float(pair[1])))
        bucket[type_dir] = points
    return bucket


def serialize_matrices(matrices: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, Any]:
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
    matrices = {"lwir": None, "visible": None}
    if not isinstance(raw, dict):
        return matrices
    for key in matrices.keys():
        entry = raw.get(key)
        if not isinstance(entry, dict):
            continue
        camera = entry.get("camera_matrix")
        distortion = entry.get("distortion")
        if is_matrix3x3(camera) and is_distortion_vector(distortion):
            matrices[key] = {
                "camera_matrix": camera,
                "distortion": distortion,
            }
            if entry.get("image_size"):
                matrices[key]["image_size"] = entry.get("image_size")
            if entry.get("samples") is not None:
                matrices[key]["samples"] = entry.get("samples")
            if entry.get("reprojection_error") is not None:
                matrices[key]["reprojection_error"] = entry.get("reprojection_error")
    return matrices


def serialize_extrinsic(extrinsic: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not extrinsic:
        return {}
    payload: Dict[str, Any] = {}
    rotation = extrinsic.get("rotation")
    if is_matrix3x3(rotation):
        payload["rotation"] = rotation
    translation = extrinsic.get("translation")
    if is_vector3(translation):
        payload["translation"] = [float(num) for num in translation]
    essential = extrinsic.get("essential_matrix")
    if is_matrix3x3(essential):
        payload["essential_matrix"] = essential
    fundamental = extrinsic.get("fundamental_matrix")
    if is_matrix3x3(fundamental):
        payload["fundamental_matrix"] = fundamental
    baseline = extrinsic.get("baseline") or extrinsic.get("baseline_mm")
    if isinstance(baseline, (int, float)):
        payload["baseline"] = float(baseline)
    samples = extrinsic.get("samples")
    if isinstance(samples, int):
        payload["samples"] = samples
    reproj = extrinsic.get("reprojection_error")
    if isinstance(reproj, (int, float)):
        payload["reprojection_error"] = float(reproj)
    updated = extrinsic.get("updated_at")
    if isinstance(updated, str):
        payload["updated_at"] = updated
    return payload


def deserialize_extrinsic(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    rotation = raw.get("rotation")
    translation = raw.get("translation")
    if not is_matrix3x3(rotation) or not is_vector3(translation):
        return None
    result: Dict[str, Any] = {
        "rotation": rotation,
        "translation": [float(num) for num in translation],
    }
    if is_matrix3x3(raw.get("essential_matrix")):
        result["essential_matrix"] = raw["essential_matrix"]
    if is_matrix3x3(raw.get("fundamental_matrix")):
        result["fundamental_matrix"] = raw["fundamental_matrix"]
    baseline = raw.get("baseline") or raw.get("baseline_mm")
    if isinstance(baseline, (int, float)):
        result["baseline"] = float(baseline)
    samples = raw.get("samples")
    if isinstance(samples, int):
        result["samples"] = samples
    reproj = raw.get("reprojection_error")
    if isinstance(reproj, (int, float)):
        result["reprojection_error"] = float(reproj)
    updated = raw.get("updated_at")
    if isinstance(updated, str):
        result["updated_at"] = updated
    return result


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
        "signatures": {},
        "calibration": {},
        "matrices": {},
        "extrinsic": {},
        "reproj_errors": {},
        "extrinsic_errors": {},
        "overrides": [],
    }


def load_dataset_cache_file(cache_path: Path) -> DatasetCache:
    if not cache_path.exists():
        return empty_dataset_entry()
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
    "serialize_marks",
    "deserialize_marks",
    "serialize_calibration",
    "deserialize_calibration",
    "serialize_reprojection_errors",
    "deserialize_reprojection_errors",
    "serialize_extrinsic_errors",
    "deserialize_extrinsic_errors",
    "serialize_matrices",
    "deserialize_matrices",
    "serialize_extrinsic",
    "deserialize_extrinsic",
    "is_matrix3x3",
    "is_distortion_vector",
    "is_vector3",
    "empty_dataset_entry",
    "load_dataset_cache_file",
    "save_dataset_cache_file",
]
