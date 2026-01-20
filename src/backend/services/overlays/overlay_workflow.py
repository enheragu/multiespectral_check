"""Draw and cache per-image overlays (labels, reasons, calibration cues) for the viewer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QPoint
from PyQt6.QtGui import QColor, QPainter, QPixmap

from backend.services.lru_index import LRUIndex
from backend.utils.overlays import (
    draw_calibration_overlay,
    draw_overlay_labels,
    draw_label_boxes,
    draw_reason_overlay,
    paint_rule_of_thirds,
)
from common.reasons import REASON_STYLES
from common.log_utils import log_debug

if TYPE_CHECKING:
    from backend.utils.stereo_alignment import AlignmentTransform

LabelOverlay = Tuple[str, float, float, float, float, QColor]
CALIBRATION_BORDER_COLOR = QColor("#00ffea")
CALIBRATION_ERROR_COLOR = QColor("#dc3545")
WARNING_LABEL_COLOR = QColor("#ffb347")
SUBPIXEL_CORNER_COLOR = QColor("#ff8c00")  # Dark orange for subpixel corners


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

    def _corner_signature(self, corner_points: Optional[List[List[float]]]) -> Optional[Tuple[Tuple[float, float], ...]]:
        if not corner_points:
            return None
        return tuple((round(pt[0], 4), round(pt[1], 4)) for pt in corner_points)

    def build_signature(
        self,
        view_rectified: bool,
        grid_mode: str,
        reason: Optional[str],
        reason_label: Optional[str],
        calibration: bool,
        calibration_auto: bool,
        calibration_detected: Optional[bool],
        corner_points: Optional[List[List[float]]],
        corner_points_secondary: Optional[List[List[float]]] = None,
        corners_refined: bool = False,
        warning_text: Optional[str] = None,
        calibration_errors: Optional[Dict[str, Optional[float]]] = None,
        stereo_error: Optional[float] = None,
        thresholds: Optional[Dict[str, float]] = None,
        label_sig: Optional[Tuple[Any, ...]] = None,
        alignment_output_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Any, ...]:
        def _rounded_errors(errors: Optional[Dict[str, Optional[float]]]) -> Optional[Tuple[Tuple[str, float], ...]]:
            if not errors:
                return None
            items = []
            for key in sorted(errors.keys()):
                val = errors.get(key)
                if isinstance(val, (int, float)):
                    items.append((key, round(float(val), 4)))
            return tuple(items) if items else None

        def _rounded_thresholds(data: Optional[Dict[str, float]]) -> Optional[Tuple[Tuple[str, float], ...]]:
            if not data:
                return None
            return tuple(sorted((k, round(float(v), 4)) for k, v in data.items()))

        return (
            view_rectified,
            grid_mode,
            reason or "",
            (reason_label or "")[:64],
            calibration,
            calibration_auto,
            calibration_detected,
            self._corner_signature(corner_points),
            self._corner_signature(corner_points_secondary),
            corners_refined,
            (warning_text or "")[:64],
            _rounded_errors(calibration_errors),
            round(stereo_error, 4) if isinstance(stereo_error, (int, float)) else None,
            _rounded_thresholds(thresholds),
            label_sig,
            alignment_output_size,
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
            if key in self._overlay_cache:
                del self._overlay_cache[key]
        self._enforce_overlay_cache_limit()

    def _enforce_overlay_cache_limit(self) -> None:
        while len(self._overlay_cache) > self.cache_limit:
            evicted = self._overlay_cache_order.pop_oldest()
            if evicted is None:
                break
            if evicted in self._overlay_cache:
                del self._overlay_cache[evicted]

    def render(
        self,
        base: str,
        channel: str,
        pixmap: Optional[QPixmap],
        *,
        view_rectified: bool,
        grid_mode: str,
        reason: Optional[str],
        reason_label: Optional[str],
        calibration: bool,
        calibration_auto: bool,
        calibration_detected: Optional[bool],
        corner_points: Optional[List[List[float]]],
        corner_points_secondary: Optional[List[List[float]]] = None,
        corners_refined: bool = False,
        warning_text: Optional[str],
        calibration_errors: Optional[Dict[str, Optional[float]]],
        stereo_error: Optional[float],
        thresholds: Optional[Dict[str, float]],
        label_boxes: List[LabelOverlay],
        label_sig: Optional[Tuple[Any, ...]],
        alignment_transform: Optional["AlignmentTransform"] = None,
        original_size: Optional[Tuple[int, int]] = None,
        camera_matrix: Optional[Any] = None,
        distortion: Optional[Any] = None,
    ) -> Optional[QPixmap]:
        if not pixmap or pixmap.isNull():
            return None
        signature = self.build_signature(
            view_rectified,
            grid_mode,
            reason,
            reason_label,
            calibration,
            calibration_auto,
            calibration_detected,
            corner_points,
            corner_points_secondary,
            corners_refined,
            warning_text,
            calibration_errors,
            stereo_error,
            thresholds,
            label_sig,
            alignment_output_size=alignment_transform.output_size if alignment_transform else None,
        )
        cached = self._get_cached_overlay(base, channel, signature)
        if cached is not None:
            return cached

        base_pix = pixmap.copy()
        base_w, base_h = base_pix.width(), base_pix.height()
        overlay_pen_width = max(2, int(max(base_w, base_h) / 200))
        painter = QPainter(base_pix)

        if grid_mode and grid_mode != "off":
            paint_rule_of_thirds(painter, base_w, base_h, detailed=(grid_mode == "detailed"))

        label_entries: List[Tuple[str, QColor]] = []
        if reason:
            style = REASON_STYLES.get(reason, {"color": QColor("red"), "text": reason})
            label_color: QColor = style.get("color", QColor("red"))  # type: ignore[assignment]
            label_text = reason_label or str(style.get("text", reason))
            draw_reason_overlay(painter, base_pix, label_color, label_text, overlay_pen_width)
            label_entries.append((label_text, label_color))
        if calibration:
            draw_calibration_overlay(painter, base_pix, CALIBRATION_BORDER_COLOR, max(3, overlay_pen_width + 1))
            # Build calibration status with Auto/User prefix and refined tag
            prefix = "Auto" if calibration_auto else "Manual"
            refined_tag = " (refined)" if corners_refined else ""
            if warning_text:
                status = f"{prefix}: Calibration. Chessboard discarded"
            elif calibration_detected is not None:
                status = f"{prefix}: Calibration{refined_tag}"
            else:
                status = f"{prefix}: Calibration"
            label_entries.append((status, CALIBRATION_BORDER_COLOR))
        if warning_text:
            trimmed = warning_text if len(warning_text) <= 60 else f"{warning_text[:57]}…"
            prefix = "Discarded" if calibration else "Warning"
            label_entries.append((f"{prefix}: {trimmed}", WARNING_LABEL_COLOR))
        if calibration_errors:
            channel_map = {"lwir": "LWIR", "visible": "Visible"}
            value = calibration_errors.get(channel)
            if isinstance(value, (int, float)):
                threshold = (thresholds or {}).get(channel, 0.0)
                color = CALIBRATION_ERROR_COLOR if threshold and float(value) > threshold else CALIBRATION_BORDER_COLOR
                label_entries.append((f"{channel_map.get(channel, channel).upper()} {float(value):.3f} px", color))
        if isinstance(stereo_error, (int, float)):
            threshold = (thresholds or {}).get("stereo", 0.0)
            color = CALIBRATION_ERROR_COLOR if threshold and float(stereo_error) > threshold else CALIBRATION_BORDER_COLOR
            label_entries.append((f"Stereo {float(stereo_error):.3f} px", color))
        if label_entries:
            draw_overlay_labels(painter, base_pix.width(), base_pix.height(), label_entries)
        if label_boxes:
            draw_label_boxes(painter, base_pix.width(), base_pix.height(), label_boxes)

        # Draw corners - helper function for rendering a set of corners
        def draw_corner_set(
            points: List[List[float]],
            color: QColor,
            shape: str = "circle",  # "circle" or "cross"
        ) -> None:
            painter.setPen(color)
            painter.setBrush(color)
            r = max(3, overlay_pen_width)
            orig_w, orig_h = original_size  # type: ignore[misc]

            if alignment_transform:
                corners_array = np.array(points, dtype=np.float32)
                if channel == "visible":
                    output_coords = alignment_transform.transform_vis_corners_complete(
                        corners_array, original_size, view_rectified, camera_matrix, distortion,
                    )
                else:
                    output_coords = alignment_transform.transform_lwir_corners_complete(
                        corners_array, original_size, view_rectified, camera_matrix, distortion,
                    )
                for i in range(len(points)):
                    x, y = int(output_coords[i, 0]), int(output_coords[i, 1])
                    if shape == "circle":
                        painter.drawEllipse(QPoint(x, y), r, r)
                    else:  # cross
                        painter.drawLine(x - r, y, x + r, y)
                        painter.drawLine(x, y - r, x, y + r)
            else:
                # No alignment - batch process all corners
                scale_x = base_w / orig_w if orig_w > 0 else 1.0
                scale_y = base_h / orig_h if orig_h > 0 else 1.0

                # Denormalize all points at once
                coords = np.array([[u * orig_w, v * orig_h] for u, v in points], dtype=np.float32)

                # Apply undistort if needed (vectorized)
                if view_rectified and camera_matrix is not None and distortion is not None:
                    try:
                        cam = np.array(camera_matrix, dtype=np.float32)
                        dist = np.array(distortion, dtype=np.float32).reshape(-1)
                        new_cam, _ = cv2.getOptimalNewCameraMatrix(
                            cam, dist, (orig_w, orig_h), 1, (orig_w, orig_h)
                        )
                        # Reshape for undistortPoints: (N, 1, 2)
                        pts_reshaped = coords.reshape(-1, 1, 2)
                        undistorted = cv2.undistortPoints(pts_reshaped, cam, dist, P=new_cam)
                        coords = undistorted.reshape(-1, 2)
                    except Exception:
                        pass

                # Scale and draw
                for orig_x, orig_y in coords:
                    x, y = int(orig_x * scale_x), int(orig_y * scale_y)
                    if shape == "circle":
                        painter.drawEllipse(QPoint(x, y), r, r)
                    else:  # cross
                        painter.drawLine(x - r, y, x + r, y)
                        painter.drawLine(x, y - r, x, y + r)

        # Draw secondary corners first (original) so primary (subpixel) draws on top
        if corner_points_secondary and original_size:
            # Secondary corners: blue circles (original detection)
            draw_corner_set(corner_points_secondary, QColor("#4a90d9"), "circle")

        if corner_points and original_size:
            # Primary corners: use orange for subpixel (crosses), cyan for original (circles)
            if corners_refined:
                dot_color = SUBPIXEL_CORNER_COLOR  # Orange for subpixel
                shape = "cross"
            else:
                dot_color = WARNING_LABEL_COLOR if warning_text else CALIBRATION_BORDER_COLOR
                shape = "circle"
            # Log diagnostic info for first corner only
            u0, v0 = corner_points[0]
            log_debug(
                f"Corner transform [{channel}]: orig_size={original_size}, base={base_w}x{base_h}, "
                f"has_align={alignment_transform is not None}, view_rectified={view_rectified}, "
                f"corner0=({u0:.4f},{v0:.4f}), refined={corners_refined}",
                "OVERLAY"
            )
            draw_corner_set(corner_points, dot_color, shape)

        painter.end()
        bucket = self._overlay_cache.setdefault(base, {})
        bucket[channel] = OverlayCacheEntry(signature, base_pix)
        self._track_overlay_cache_use(base)
        return base_pix
