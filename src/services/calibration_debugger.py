from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QColor, QPainter, QPixmap

from services.dataset_session import DatasetSession
from utils.calibration import (
    CalibrationCorners,
    CalibrationDebugBundle,
    CalibrationResult,
    analyze_pair,
)

CHANNEL_ORDER = ("lwir", "visible")


class CalibrationDebugger:
    """Handle calibration analysis, status formatting, and debug artifact creation."""

    def __init__(
        self,
        session: DatasetSession,
        chessboard_size: Tuple[int, int],
        debug_dir_name: str = "_calibration_debug",
    ) -> None:
        self.session = session
        self.chessboard_size = chessboard_size
        self.debug_dir_name = debug_dir_name

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------
    def analyze(
        self,
        base: str,
        debug_enabled: bool,
    ) -> Tuple[CalibrationResult, CalibrationCorners, CalibrationDebugBundle]:
        lwir_pixmap, vis_pixmap = self.session.load_raw_pixmaps(base)
        results, corners, _, debug_bundle = analyze_pair(
            base,
            lwir_pixmap,
            vis_pixmap,
            self.chessboard_size,
            debug=debug_enabled,
        )
        return results, corners, debug_bundle

    def needs_corner_refresh(
        self,
        results: Optional[CalibrationResult],
        corners: Optional[CalibrationCorners],
    ) -> bool:
        if not results:
            return True
        for key in CHANNEL_ORDER:
            detected = results.get(key)
            if detected is True and not ((corners or {}).get(key)):
                return True
        return False

    # ------------------------------------------------------------------
    # Status + debug bundle helpers
    # ------------------------------------------------------------------
    def format_status(
        self,
        base: str,
        results: Optional[CalibrationResult],
        corners: Optional[CalibrationCorners],
        *,
        include_counts: bool,
    ) -> str:
        expected = self.chessboard_size[0] * self.chessboard_size[1]
        status_parts: List[str] = []
        for label, key in (("LWIR", "lwir"), ("Visible", "visible")):
            result = (results or {}).get(key)
            if result is True:
                entry = f"{label}: detected"
            elif result is False:
                entry = f"{label}: not found"
            else:
                continue
            if include_counts:
                count = len(((corners or {}).get(key)) or [])
                entry += f" ({count}/{expected} corners)"
            status_parts.append(entry)
        if not status_parts:
            status_parts.append("Chessboard detection skipped")
        return f"Calibration tagged for {base} ({'; '.join(status_parts)})"

    def build_cached_debug_bundle(
        self,
        base: str,
        results: Optional[CalibrationResult],
        corners: Optional[CalibrationCorners],
    ) -> Optional[CalibrationDebugBundle]:
        if not results:
            return None
        lwir_pixmap, vis_pixmap = self.session.load_raw_pixmaps(base)
        bundle: CalibrationDebugBundle = {
            "lwir": self._render_payload(
                lwir_pixmap,
                results.get("lwir"),
                (corners or {}).get("lwir"),
            ),
            "visible": self._render_payload(
                vis_pixmap,
                results.get("visible"),
                (corners or {}).get("visible"),
            ),
        }
        if not any(bundle.values()):
            return None
        return bundle

    def save_debug_bundle(
        self,
        base: str,
        bundle: Optional[CalibrationDebugBundle],
    ) -> Tuple[Optional[Path], List[str]]:
        if not bundle:
            return None, []
        lines = [f"[calibration-debug] {base}"]
        saved_paths: List[Path] = []
        for channel in CHANNEL_ORDER:
            payload = bundle.get(channel) if isinstance(bundle, dict) else None
            if not payload:
                lines.append(f"  - {channel}: no frame")
                continue
            detected = payload.get("detected")
            if detected is True:
                status_label = "detected"
            elif detected is False:
                status_label = "missing"
            else:
                status_label = "unknown"
            found = payload.get("corners_found", 0)
            expected = payload.get("expected_corners", 0)
            lines.append(f"  - {channel}: {status_label} ({found}/{expected} corners)")
            saved = self._save_snapshot(base, channel, payload.get("pixmap"))
            if saved:
                lines.append(f"    saved: {saved}")
                saved_paths.append(saved)
        final_path = saved_paths[-1] if saved_paths else None
        return final_path, lines

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _render_payload(
        self,
        pixmap: Optional[QPixmap],
        detected: Optional[bool],
        corner_points: Optional[List[Tuple[float, float]]],
    ) -> Optional[Dict[str, Any]]:
        if not pixmap or pixmap.isNull():
            return None
        canvas = pixmap.copy()
        painter = QPainter(canvas)
        pen_color = QColor("#00c853") if detected else QColor("#ff5252")
        count = len(corner_points or [])
        expected = self.chessboard_size[0] * self.chessboard_size[1]
        if corner_points:
            painter.setPen(pen_color)
            painter.setBrush(pen_color)
            radius = max(3, int(max(canvas.width(), canvas.height()) / 200))
            width = canvas.width()
            height = canvas.height()
            for u, v in corner_points:
                x = int(u * width)
                y = int(v * height)
                painter.drawEllipse(QPoint(x, y), radius, radius)
        painter.setBrush(QColor(0, 0, 0, 160))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(10, 10, 360, 48)
        painter.setPen(QColor("white"))
        if detected is True:
            status_label = "detected"
        elif detected is False:
            status_label = "missing"
        else:
            status_label = "unknown"
        painter.drawText(20, 42, f"{status_label} ({count}/{expected} corners)")
        painter.end()
        return {
            "detected": detected,
            "corners_found": count,
            "expected_corners": expected,
            "pixmap": canvas,
        }

    def _save_snapshot(
        self,
        base: str,
        channel: str,
        pixmap: Optional[QPixmap],
    ) -> Optional[Path]:
        if not pixmap or pixmap.isNull():
            return None
        root = self.session.dataset_path or Path.cwd()
        debug_dir = root / self.debug_dir_name
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        safe_base = base.replace("/", "_")
        file_path = debug_dir / f"{safe_base}_{channel}_debug.png"
        if pixmap.save(str(file_path), "PNG"):
            return file_path
        return None