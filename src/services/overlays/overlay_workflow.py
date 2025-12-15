"""Draw and cache per-image overlays (labels, reasons, calibration cues) for the viewer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import QPoint
from PyQt6.QtGui import QColor, QPainter, QPixmap

from services.lru_index import LRUIndex
from utils.overlays import (
    draw_calibration_overlay,
    draw_overlay_labels,
    draw_label_boxes,
    draw_reason_overlay,
    paint_rule_of_thirds,
)
from utils.reasons import REASON_STYLES

LabelOverlay = Tuple[str, float, float, float, float, QColor]
CALIBRATION_BORDER_COLOR = QColor("#00ffea")
WARNING_LABEL_COLOR = QColor("#ffb347")


@dataclass
class OverlayCacheEntry:
    signature: Tuple[Any, ...]
    pixmap: QPixmap


class OverlayWorkflow:
    def __init__(self, cache_limit: int = 24) -> None:
        self.cache_limit = cache_limit
        self._overlay_cache: Dict[str, Dict[str, OverlayCacheEntry]] = {}
        self._overlay_cache_order = LRUIndex(cache_limit)

    def invalidate(self, base: Optional[str] = None) -> None:
        if base is None:
            self._overlay_cache.clear()
            self._overlay_cache_order.clear()
            return
        self._overlay_cache.pop(base, None)
        self._overlay_cache_order.remove(base)

    def is_cached(self, base: str) -> bool:
        bucket = self._overlay_cache.get(base)
        if not bucket:
            return False
        for channel in ("lwir", "visible"):
            entry = bucket.get(channel)
            if not entry or not entry.pixmap or entry.pixmap.isNull():
                return False
        return True

    def _corner_signature(self, corner_points: Optional[List[Tuple[float, float]]]) -> Optional[Tuple[Tuple[float, float], ...]]:
        if not corner_points:
            return None
        return tuple((round(u, 4), round(v, 4)) for u, v in corner_points)

    def build_signature(
        self,
        view_rectified: bool,
        show_grid: bool,
        reason: Optional[str],
        calibration: bool,
        calibration_detected: Optional[bool],
        corner_points: Optional[List[Tuple[float, float]]],
        warning_text: Optional[str],
        label_sig: Optional[Tuple[Any, ...]] = None,
    ) -> Tuple[Any, ...]:
        return (
            view_rectified,
            show_grid,
            reason or "",
            calibration,
            calibration_detected,
            self._corner_signature(corner_points),
            (warning_text or "")[:64],
            label_sig,
        )

    def _get_cached_overlay(
        self,
        base: str,
        channel: str,
        signature: Tuple[Any, ...],
    ) -> Optional[QPixmap]:
        bucket = self._overlay_cache.get(base)
        if not bucket:
            return None
        entry = bucket.get(channel)
        if not entry:
            return None
        if entry.signature != signature or not entry.pixmap or entry.pixmap.isNull():
            return None
        return entry.pixmap

    def _track_overlay_cache_use(self, base: Optional[str]) -> None:
        if not base:
            return
        evicted = self._overlay_cache_order.touch(base)
        for key in evicted:
            self._overlay_cache.pop(key, None)
        self._enforce_overlay_cache_limit()

    def _enforce_overlay_cache_limit(self) -> None:
        while len(self._overlay_cache) > self.cache_limit:
            evicted = self._overlay_cache_order.pop_oldest()
            if evicted is None:
                break
            self._overlay_cache.pop(evicted, None)

    def render(
        self,
        base: str,
        channel: str,
        pixmap: Optional[QPixmap],
        *,
        view_rectified: bool,
        show_grid: bool,
        reason: Optional[str],
        calibration: bool,
        calibration_detected: Optional[bool],
        corner_points: Optional[List[Tuple[float, float]]],
        warning_text: Optional[str],
        label_boxes: List[LabelOverlay],
        label_sig: Optional[Tuple[Any, ...]],
    ) -> Optional[QPixmap]:
        if not pixmap or pixmap.isNull():
            return None
        signature = self.build_signature(
            view_rectified,
            show_grid,
            reason,
            calibration,
            calibration_detected,
            corner_points,
            warning_text,
            label_sig,
        )
        cached = self._get_cached_overlay(base, channel, signature)
        if cached is not None:
            return cached

        base_pix = pixmap.copy()
        base_w, base_h = base_pix.width(), base_pix.height()
        overlay_pen_width = max(2, int(max(base_w, base_h) / 200))
        painter = QPainter(base_pix)

        if show_grid:
            paint_rule_of_thirds(painter, base_w, base_h)

        label_entries = []
        if reason:
            style = REASON_STYLES.get(reason, {"color": QColor("red"), "text": reason})
            draw_reason_overlay(painter, base_pix, style["color"], style["text"], overlay_pen_width)
            label_entries.append((style["text"], style["color"]))
        if calibration:
            draw_calibration_overlay(painter, base_pix, CALIBRATION_BORDER_COLOR, max(3, overlay_pen_width + 1))
            if warning_text:
                status = "Chessboard discarded"
            elif calibration_detected is not None:
                status = "Chessboard detected" if calibration_detected else "Chessboard missing"
            else:
                status = "Calibration candidate"
            label_entries.append((status, CALIBRATION_BORDER_COLOR))
        if warning_text:
            trimmed = warning_text if len(warning_text) <= 60 else f"{warning_text[:57]}…"
            label_entries.append((f"Suspect corners: {trimmed}", WARNING_LABEL_COLOR))
        if label_entries:
            draw_overlay_labels(painter, base_pix.width(), base_pix.height(), label_entries)
        if label_boxes:
            draw_label_boxes(painter, base_pix.width(), base_pix.height(), label_boxes)
        if corner_points:
            dot_color = WARNING_LABEL_COLOR if warning_text else CALIBRATION_BORDER_COLOR
            painter.setPen(dot_color)
            painter.setBrush(dot_color)
            radius = max(3, overlay_pen_width)
            for u, v in corner_points:
                x = int(u * base_w)
                y = int(v * base_h)
                painter.drawEllipse(QPoint(x, y), radius, radius)

        painter.end()
        bucket = self._overlay_cache.setdefault(base, {})
        bucket[channel] = OverlayCacheEntry(signature, base_pix)
        self._track_overlay_cache_use(base)
        return base_pix
