"""Zoom/pan enabled QScrollArea for synchronized image viewing."""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QPoint, QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QLabel, QScrollArea


class ZoomPanView(QScrollArea):
    """Scrollable label that supports Ctrl+wheel zoom and drag panning."""

    transformChanged = pyqtSignal(float, float, float)
    contextRequested = pyqtSignal(QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(False)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._label = QLabel("Select a dataset")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("background: #f0f0f0; border: 2px solid gray;")
        self._label.setMinimumSize(400, 400)
        self.setWidget(self._label)

        self._scale = 1.0
        self._min_scale = 0.2
        self._max_scale = 6.0
        self._base_pixmap: Optional[QPixmap] = None
        self._drag_origin: Optional[QPointF] = None
        self._suppress_emit = False

    def set_placeholder(self, text: str) -> None:
        self._base_pixmap = None
        self._scale = 1.0
        self._label.setText(text)
        self._label.setPixmap(QPixmap())
        self._label.adjustSize()
        self.horizontalScrollBar().setValue(0)
        self.verticalScrollBar().setValue(0)
        self._emit_transform()

    def set_pixmap(self, pixmap: Optional[QPixmap]) -> None:
        self._base_pixmap = pixmap
        self._scale = 1.0
        if pixmap and not pixmap.isNull():
            self._label.setText("")
            self.reset_transform()
            return
        self.set_placeholder("Image error")

    def reset_transform(self) -> None:
        if not self._base_pixmap or self._base_pixmap.isNull():
            return
        viewport = self.viewport().size()
        if viewport.width() <= 0 or viewport.height() <= 0:
            self._scale = 1.0
        else:
            fit_x = viewport.width() / self._base_pixmap.width()
            fit_y = viewport.height() / self._base_pixmap.height()
            target_scale = min(fit_x, fit_y)
            self._scale = max(self._min_scale, min(self._max_scale, target_scale))
        self._apply_scale()
        self.horizontalScrollBar().setValue(0)
        self.verticalScrollBar().setValue(0)
        self._emit_transform()

    def wheelEvent(self, event):  # noqa: N802 (Qt API name)
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta != 0:
                factor = 1.15 if delta > 0 else 0.85
                self._set_scale(self._scale * factor)
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._drag_origin is not None:
            delta = event.position() - self._drag_origin
            self._drag_origin = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            self._emit_transform()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._drag_origin is not None:
            self._drag_origin = None
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):  # noqa: N802
        if self._drag_origin is not None:
            self._drag_origin = None
            self.unsetCursor()
        super().leaveEvent(event)

    def contextMenuEvent(self, event):  # noqa: N802
        self.contextRequested.emit(event.globalPos())
        event.accept()

    def apply_external_transform(self, scale: float, h_ratio: float, v_ratio: float) -> None:
        if not self._base_pixmap or self._base_pixmap.isNull():
            return
        self._suppress_emit = True
        self._set_scale(scale, emit=False)
        self._set_scroll_ratio(self.horizontalScrollBar(), h_ratio)
        self._set_scroll_ratio(self.verticalScrollBar(), v_ratio)
        self._suppress_emit = False

    def current_transform(self) -> tuple[float, float, float]:
        return (
            self._scale,
            self._ratio(self.horizontalScrollBar()),
            self._ratio(self.verticalScrollBar()),
        )

    def _set_scale(self, scale: float, emit: bool = True) -> None:
        if not self._base_pixmap or self._base_pixmap.isNull():
            return
        scale = max(self._min_scale, min(self._max_scale, scale))
        if abs(scale - self._scale) < 1e-3:
            return
        self._scale = scale
        self._apply_scale()
        if emit:
            self._emit_transform()

    def _apply_scale(self) -> None:
        if not self._base_pixmap or self._base_pixmap.isNull():
            self._label.setPixmap(QPixmap())
            return
        width = max(1, int(self._base_pixmap.width() * self._scale))
        height = max(1, int(self._base_pixmap.height() * self._scale))
        scaled = self._base_pixmap.scaled(
            width,
            height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)
        self._label.resize(scaled.size())

    def _set_scroll_ratio(self, scrollbar, ratio: float) -> None:  # type: ignore[override]
        ratio = max(0.0, min(1.0, ratio))
        maximum = scrollbar.maximum()
        scrollbar.setValue(int(ratio * maximum) if maximum > 0 else 0)

    def _ratio(self, scrollbar) -> float:  # type: ignore[override]
        maximum = scrollbar.maximum()
        if maximum <= 0:
            return 0.0
        return scrollbar.value() / maximum

    def _emit_transform(self) -> None:
        if self._suppress_emit:
            return
        self.transformChanged.emit(*self.current_transform())

