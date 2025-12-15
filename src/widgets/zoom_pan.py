"""Zoom/pan enabled QScrollArea for synchronized image viewing."""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QPoint, QPointF, Qt, QRect, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QLabel, QScrollArea, QRubberBand


class ZoomPanView(QScrollArea):
    """Scrollable label that supports Ctrl+wheel zoom and drag panning."""

    transformChanged = pyqtSignal(float, float, float)
    contextRequested = pyqtSignal(QPoint)
    labelBoxDefined = pyqtSignal(float, float, float, float)
    labelSelectionCanceled = pyqtSignal()
    labelDeleteRequested = pyqtSignal(float, float, QPoint)

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
        self._labeling_mode = False
        self._label_start: Optional[QPointF] = None
        self._label_start_px: Optional[QPoint] = None
        self._label_current_px: Optional[QPoint] = None
        self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)

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
        if self._labeling_mode and event.button() == Qt.MouseButton.LeftButton:
            pos = self._image_pos(event)
            pos_px = self._label_pos(event)
            if pos and pos_px:
                self._label_start = pos
                self._label_start_px = pos_px
                self._label_current_px = pos_px
                self._update_rubber_band()
            event.accept()
            return
        if self._labeling_mode and event.button() == Qt.MouseButton.RightButton:
            pos = self._image_pos(event)
            if pos:
                self.labelDeleteRequested.emit(pos.x(), pos.y(), event.globalPosition().toPoint())
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._labeling_mode:
            if self._label_start is not None:
                pos_px = self._label_pos(event)
                if pos_px:
                    self._label_current_px = pos_px
                    self._update_rubber_band()
            event.accept()
            return
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
        if self._labeling_mode and event.button() == Qt.MouseButton.LeftButton:
            end = self._image_pos(event)
            if self._label_start and end:
                self._emit_box(self._label_start, end)
            self._label_start = None
            self._label_start_px = None
            self._label_current_px = None
            self._rubber_band.hide()
            event.accept()
            return
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

    def cancel_label_selection(self) -> None:
        self._label_start = None
        self._label_start_px = None
        self._label_current_px = None
        self._rubber_band.hide()
        self.labelSelectionCanceled.emit()

    def set_labeling_mode(self, enabled: bool) -> None:
        self._labeling_mode = enabled
        self._label_start = None
        self._label_start_px = None
        self._label_current_px = None
        self._rubber_band.hide()
        if not enabled:
            self.unsetCursor()

    def _image_pos(self, event) -> Optional[QPointF]:
        if not self._label or not self._label.pixmap() or self._label.pixmap().isNull():
            return None
        label_pos = self._label.mapFrom(self.viewport(), event.position().toPoint())
        pix = self._label.pixmap()
        if label_pos.x() < 0 or label_pos.y() < 0 or label_pos.x() > pix.width() or label_pos.y() > pix.height():
            return None
        x_norm = label_pos.x() / pix.width()
        y_norm = label_pos.y() / pix.height()
        return QPointF(x_norm, y_norm)

    def _label_pos(self, event) -> Optional[QPoint]:
        if not self._label or not self._label.pixmap() or self._label.pixmap().isNull():
            return None
        label_pos = self._label.mapFrom(self.viewport(), event.position().toPoint())
        pix = self._label.pixmap()
        if label_pos.x() < 0 or label_pos.y() < 0 or label_pos.x() > pix.width() or label_pos.y() > pix.height():
            return None
        return label_pos

    def _emit_box(self, p1: QPointF, p2: QPointF) -> None:
        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()
        left = min(x1, x2)
        right = max(x1, x2)
        top = min(y1, y2)
        bottom = max(y1, y2)
        if right - left <= 0 or bottom - top <= 0:
            return
        self.labelBoxDefined.emit(left, top, right, bottom)

    def _update_rubber_band(self) -> None:
        if self._label_start_px is None or self._label_current_px is None:
            self._rubber_band.hide()
            return
        # Map label coords to viewport coords for the rubber band geometry
        top_left = self._label.mapTo(self.viewport(), self._label_start_px)
        bottom_right = self._label.mapTo(self.viewport(), self._label_current_px)
        rect = QRect(top_left, bottom_right).normalized()
        self._rubber_band.setGeometry(rect)
        self._rubber_band.show()

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

