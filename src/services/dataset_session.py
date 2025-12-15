"""Encapsulate dataset lifecycle, cache persistence, and destructive actions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPixmap

from dataset_loader import DatasetLoader
from services.cache_service import (
    CachePersistPayload,
    CacheService,
    deserialize_calibration,
    deserialize_extrinsic_errors,
    deserialize_extrinsic,
    deserialize_matrices,
    deserialize_marks,
    deserialize_reprojection_errors,
    deserialize_signatures,
)
from services.lru_index import LRUIndex
from services.viewer_state import ViewerState
from utils.calibration import undistort_pixmap
from utils.duplicates import (
    SIGNATURE_THRESHOLD,
    get_signature,
    signature_distance,
    store_signature,
)
from utils.reasons import REASON_DUPLICATE, REASON_MISSING_PAIR, REASON_USER


PIXMAP_CACHE_LIMIT = 24


@dataclass
class DeleteOutcome:
    moved: int
    failed: List[str]
    dataset_available: bool


class DatasetSession:
    def __init__(self) -> None:
        self.loader: Optional[DatasetLoader] = None
        self.dataset_path: Optional[Path] = None
        self.state = ViewerState()
        self.cache_service = CacheService()
        self.cache_dirty = False
        self._pixmap_cache_order = LRUIndex(PIXMAP_CACHE_LIMIT)

    # ------------------------------------------------------------------
    # Dataset lifecycle
    # ------------------------------------------------------------------
    def last_dataset(self) -> Optional[str]:
        return self.cache_service.last_dataset()

    def reset_state(self) -> None:
        self.state.reset()
        self._pixmap_cache_order.clear()

    def load(self, dir_path: Path) -> bool:
        loader = DatasetLoader(str(dir_path))
        if not loader.load_dataset():
            self.loader = None
            self.dataset_path = None
            self.cache_service.set_active_dataset(None)
            self.state.reset()
            self._pixmap_cache_order.clear()
            return False
        self.loader = loader
        self.dataset_path = dir_path
        self.state.reset()
        self._pixmap_cache_order.clear()
        self.cache_service.set_active_dataset(dir_path)
        self._hydrate_from_cache()
        self._filter_state_by_loader()
        self._auto_mark_missing_pairs()
        self.mark_cache_dirty()
        return True

    def total_pairs(self) -> int:
        return len(self.loader.image_bases) if self.loader else 0

    def has_images(self) -> bool:
        return bool(self.loader and self.loader.image_bases)

    def get_base(self, index: int) -> Optional[str]:
        if not self.loader or not self.loader.image_bases:
            return None
        if index < 0 or index >= len(self.loader.image_bases):
            return None
        return self.loader.image_bases[index]

    def calibration_filter_position(self, current_index: int) -> Tuple[int, int]:
        if not self.loader or not self.loader.image_bases or not self.state.calibration_marked:
            return 0, 0
        filtered_total = 0
        filtered_index = 0
        for idx, base in enumerate(self.loader.image_bases):
            if base not in self.state.calibration_marked:
                continue
            filtered_total += 1
            if idx == current_index:
                filtered_index = filtered_total
        return filtered_index, filtered_total

    def get_metadata_text(self, base: str, type_dir: str) -> str:
        if not self.loader:
            return "No metadata found"
        metadata = self.loader.get_metadata(base, type_dir)
        if not metadata:
            return "No metadata found"
        lines = [f"{key}: {value}" for key, value in metadata.items()]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Cache coordination
    # ------------------------------------------------------------------
    def mark_cache_dirty(self) -> None:
        self.cache_dirty = True

    def snapshot_cache_payload(self) -> Optional[CachePersistPayload]:
        if not self.cache_dirty:
            return None
        if not self.dataset_path:
            self.cache_dirty = False
            return None
        self.cache_service.snapshot_state(
            self.state.marked_for_delete,
            self.state.signatures,
            self.state.calibration_marked,
            self.state.calibration_outliers,
            self.state.calibration_results,
            self.state.calibration_corners,
            self.state.calibration_warnings,
            self.state.calibration_matrices,
            self.state.calibration_reproj_errors,
            self.state.extrinsic_pair_errors,
            self.state.auto_override,
            self.state.calibration_extrinsic,
        )
        payload = self.cache_service.build_persist_payload()
        self.cache_dirty = False
        return payload

    # ------------------------------------------------------------------
    # Dangerous operations
    # ------------------------------------------------------------------
    def delete_marked_entries(self) -> DeleteOutcome:
        if not self.loader or not self.state.marked_for_delete:
            return DeleteOutcome(0, [], bool(self.loader and self.loader.image_bases))
        failed: List[str] = []
        moved = 0
        for base, reason in list(self.state.marked_for_delete.items()):
            if not self.loader.delete_entry(base, reason):
                failed.append(base)
                continue
            self.state.marked_for_delete.pop(base, None)
            self._evict_pixmap_cache_entry(base)
            self.state.calibration_marked.discard(base)
            self.state.calibration_outliers.discard(base)
            self.state.calibration_results.pop(base, None)
            self.state.calibration_corners.pop(base, None)
            self.state.calibration_warnings.pop(base, None)
            for bucket in self.state.calibration_reproj_errors.values():
                bucket.pop(base, None)
            self.state.extrinsic_pair_errors.pop(base, None)
            self.state.remove_calibration_entry(base)
            self.state.auto_override.discard(base)
            moved += 1
        dataset_available = self.loader.load_dataset()
        self._clear_pixmap_cache()
        self.state.signatures = {}
        if dataset_available:
            self._filter_state_by_loader()
            self._auto_mark_missing_pairs()
        else:
            self.state.calibration_marked.clear()
            self.state.calibration_outliers.clear()
            self.state.calibration_results.clear()
            self.state.calibration_corners.clear()
            self.state.calibration_warnings.clear()
            self.state.extrinsic_pair_errors.clear()
            self.state.auto_override.clear()
        self.state.rebuild_reason_counts()
        self.state.rebuild_calibration_summary()
        self.mark_cache_dirty()
        return DeleteOutcome(moved, failed, dataset_available)

    def restore_from_trash(self) -> int:
        if not self.loader:
            return 0
        restored_pairs = self.loader.restore_from_trash()
        if restored_pairs == 0:
            return 0
        self._clear_pixmap_cache()
        self.state.signatures = {}
        self.state.clear_markings()
        self.state.calibration_marked.clear()
        self.state.calibration_outliers.clear()
        self.state.calibration_results.clear()
        self.state.calibration_corners.clear()
        self.state.calibration_warnings.clear()
        self.state.extrinsic_pair_errors.clear()
        self._filter_state_by_loader()
        self._auto_mark_missing_pairs()
        self.state.rebuild_calibration_summary()
        self.mark_cache_dirty()
        return restored_pairs

    def count_trash_pairs(self) -> int:
        if not self.loader:
            return 0
        return self.loader.count_trash_pairs()

    def apply_signatures(
        self,
        current_index: int,
        lwir_signature: Optional[bytes],
        vis_signature: Optional[bytes],
    ) -> Tuple[bool, bool]:
        """Persist provided signatures and auto-mark duplicates if needed."""
        if not self.loader or not self.loader.image_bases:
            return False, False
        if current_index < 0 or current_index >= len(self.loader.image_bases):
            return False, False
        base = self.loader.image_bases[current_index]
        bucket = self.state.signatures.setdefault(base, {})
        cache_changed = (
            bucket.get("lwir") != lwir_signature
            or bucket.get("visible") != vis_signature
        )
        store_signature(self.state.signatures, base, "lwir", lwir_signature)
        store_signature(self.state.signatures, base, "visible", vis_signature)
        if current_index == 0:
            return cache_changed, False
        prev_base = self.loader.image_bases[current_index - 1]
        if prev_base == base:
            return cache_changed, False
        prev_lwir = get_signature(self.state.signatures, prev_base, "lwir")
        prev_vis = get_signature(self.state.signatures, prev_base, "visible")
        if not (lwir_signature and vis_signature and prev_lwir and prev_vis):
            return cache_changed, False
        lwir_diff = signature_distance(lwir_signature, prev_lwir)
        vis_diff = signature_distance(vis_signature, prev_vis)
        if (
            lwir_diff <= SIGNATURE_THRESHOLD
            and vis_diff <= SIGNATURE_THRESHOLD
            and base not in self.state.auto_override
            and base not in self.state.marked_for_delete
        ):
            if self.state.set_mark_reason(base, REASON_DUPLICATE, REASON_USER):
                return True, True
        return cache_changed, False

    # ------------------------------------------------------------------
    # Pixmap helpers
    # ------------------------------------------------------------------
    def prepare_display_pair(
        self,
        base: str,
        view_rectified: bool,
    ) -> Tuple[Optional[QPixmap], Optional[QPixmap]]:
        lwir, vis = self.load_raw_pixmaps(base)
        lwir = self._prepare_base_pixmap(lwir, "lwir", view_rectified)
        vis = self._prepare_base_pixmap(vis, "visible", view_rectified)
        return self._normalize_display_pair(lwir, vis)

    def load_raw_pixmaps(self, base: str) -> Tuple[Optional[QPixmap], Optional[QPixmap]]:
        return (
            self._get_or_load_pixmap(base, "lwir"),
            self._get_or_load_pixmap(base, "visible"),
        )

    def _get_or_load_pixmap(self, base: str, type_dir: str) -> Optional[QPixmap]:
        if not self.loader:
            return None
        cache = self.state.pixmap_cache.setdefault(base, {})
        if type_dir in cache:
            self._record_pixmap_use(base)
            return cache[type_dir]
        image_path = self.loader.get_image_path(base, type_dir)
        if image_path and image_path.exists():
            loaded = QPixmap(str(image_path))
            pixmap = loaded if not loaded.isNull() else None
        else:
            pixmap = None
        cache[type_dir] = pixmap
        self._record_pixmap_use(base)
        self._enforce_pixmap_cache_limit()
        return pixmap

    def _prepare_base_pixmap(
        self,
        pixmap: Optional[QPixmap],
        type_dir: str,
        view_rectified: bool,
    ) -> Optional[QPixmap]:
        if not pixmap or pixmap.isNull():
            return None
        return self._apply_rectification_if_needed(pixmap, type_dir, view_rectified)

    def _apply_rectification_if_needed(
        self,
        pixmap: Optional[QPixmap],
        type_dir: str,
        view_rectified: bool,
    ) -> Optional[QPixmap]:
        if not view_rectified or not pixmap:
            return pixmap
        matrices = self.state.calibration_matrices.get(type_dir)
        if not matrices:
            return pixmap
        corrected = undistort_pixmap(
            pixmap,
            matrices.get("camera_matrix"),
            matrices.get("distortion"),
        )
        return corrected or pixmap

    def _normalize_display_pair(
        self,
        lwir_pixmap: Optional[QPixmap],
        vis_pixmap: Optional[QPixmap],
    ) -> Tuple[Optional[QPixmap], Optional[QPixmap]]:
        pixmaps = [p for p in (lwir_pixmap, vis_pixmap) if p and not p.isNull()]
        if not pixmaps:
            return lwir_pixmap, vis_pixmap
        target_width = max(p.width() for p in pixmaps)
        target_height = max(p.height() for p in pixmaps)
        return (
            self._scale_to_canvas(lwir_pixmap, target_width, target_height),
            self._scale_to_canvas(vis_pixmap, target_width, target_height),
        )

    def _scale_to_canvas(
        self,
        pixmap: Optional[QPixmap],
        width: int,
        height: int,
    ) -> Optional[QPixmap]:
        if not pixmap or pixmap.isNull():
            return pixmap
        if pixmap.width() == width and pixmap.height() == height:
            return pixmap
        scaled = pixmap.scaled(
            width,
            height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        canvas = QPixmap(width, height)
        canvas.fill(Qt.GlobalColor.white)
        painter = QPainter(canvas)
        x = (width - scaled.width()) // 2
        y = (height - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        painter.end()
        return canvas

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _hydrate_from_cache(self) -> None:
        entry = self.cache_service.load_dataset_entry()
        self.state.signatures = deserialize_signatures(entry.get("signatures", {}))
        self.state.marked_for_delete = deserialize_marks(entry.get("marks", {}))
        marked, outliers, results, corners, warnings = deserialize_calibration(entry.get("calibration", {}))
        self.state.calibration_marked = marked
        self.state.calibration_outliers = outliers
        self.state.calibration_results = results
        self.state.calibration_corners = corners
        self.state.calibration_warnings = warnings
        self.state.calibration_reproj_errors = deserialize_reprojection_errors(entry.get("reproj_errors"))
        self.state.extrinsic_pair_errors = deserialize_extrinsic_errors(entry.get("extrinsic_errors"))
        self.state.calibration_matrices = deserialize_matrices(entry.get("matrices", {}))
        self.state.calibration_extrinsic = deserialize_extrinsic(entry.get("extrinsic"))
        overrides = entry.get("overrides", [])
        if isinstance(overrides, list):
            self.state.auto_override = {base for base in overrides if isinstance(base, str)}
        else:
            self.state.auto_override.clear()
        self.state.rebuild_reason_counts()
        self.state.rebuild_calibration_summary()

    def _auto_mark_missing_pairs(self) -> None:
        if not self.loader:
            return
        for base, channels in self.loader.channel_map.items():
            if (
                base in self.state.auto_override
                or base in self.state.marked_for_delete
                or base in self.state.calibration_outliers
            ):
                continue
            if "lwir" not in channels or "visible" not in channels:
                self.state.set_mark_reason(base, REASON_MISSING_PAIR, REASON_USER)

    def _filter_state_by_loader(self) -> None:
        if not self.loader:
            return
        valid_bases: Set[str] = set(self.loader.image_bases)
        self.state.marked_for_delete = {
            base: reason
            for base, reason in self.state.marked_for_delete.items()
            if base in valid_bases and isinstance(reason, str) and reason
        }
        self.state.signatures = {
            base: bucket
            for base, bucket in self.state.signatures.items()
            if base in valid_bases
        }
        self.state.calibration_marked.intersection_update(valid_bases)
        self.state.calibration_outliers.intersection_update(valid_bases)
        self.state.calibration_results = {
            base: result for base, result in self.state.calibration_results.items() if base in valid_bases
        }
        self.state.calibration_corners = {
            base: data for base, data in self.state.calibration_corners.items() if base in valid_bases
        }
        self.state.calibration_warnings = {
            base: data for base, data in self.state.calibration_warnings.items() if base in valid_bases
        }
        self.state.calibration_reproj_errors = {
            channel: {
                base: err for base, err in (bucket or {}).items() if base in valid_bases and isinstance(err, (int, float))
            }
            for channel, bucket in getattr(self.state, "calibration_reproj_errors", {}).items()
        }
        self.state.extrinsic_pair_errors = {
            base: err for base, err in self.state.extrinsic_pair_errors.items() if base in valid_bases
        }
        self.state.auto_override.intersection_update(valid_bases)
        self.state.rebuild_reason_counts()
        self.state.rebuild_calibration_summary()

    def build_outlier_rows(self, bases: Iterable[str]) -> List[Dict[str, Any]]:
        lwir_errs = self.state.calibration_reproj_errors.get("lwir", {})
        vis_errs = self.state.calibration_reproj_errors.get("visible", {})
        stereo_errs = self.state.extrinsic_pair_errors
        rows: List[Dict[str, Any]] = []
        for base in bases:
            calib_results = self.state.calibration_results.get(base, {})
            rows.append(
                {
                    "base": base,
                    "lwir": lwir_errs.get(base) if isinstance(lwir_errs, dict) else None,
                    "visible": vis_errs.get(base) if isinstance(vis_errs, dict) else None,
                    "stereo": stereo_errs.get(base) if isinstance(stereo_errs, dict) else None,
                    "included": base in self.state.calibration_marked and base not in self.state.calibration_outliers,
                    "detect_lwir": calib_results.get("lwir"),
                    "detect_visible": calib_results.get("visible"),
                }
            )
        return rows

    def _record_pixmap_use(self, base: Optional[str]) -> None:
        if not base:
            return
        evicted = self._pixmap_cache_order.touch(base)
        for key in evicted:
            self.state.pixmap_cache.pop(key, None)

    def _enforce_pixmap_cache_limit(self) -> None:
        while len(self.state.pixmap_cache) > PIXMAP_CACHE_LIMIT:
            evicted = self._pixmap_cache_order.pop_oldest()
            if evicted is None:
                break
            self.state.pixmap_cache.pop(evicted, None)

    def _evict_pixmap_cache_entry(self, base: str) -> None:
        self.state.pixmap_cache.pop(base, None)
        self._pixmap_cache_order.remove(base)

    def _clear_pixmap_cache(self) -> None:
        self.state.pixmap_cache.clear()
        self._pixmap_cache_order.clear()
