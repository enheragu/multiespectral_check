"""Zoom/pan enabled QScrollArea for synchronized image viewing."""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt, QRect, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QColor, QPen, QFont
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
        self._label.setMinimumSize(100, 100)  # Further reduced to allow narrower window
        self.setWidget(self._label)

        # Install event filter for overlay painting
        self._label.installEventFilter(self)

        # Allow the scroll area itself to shrink
        self.setMinimumSize(100, 100)

        self._scale = 1.0
        self._fit_scale = 1.0
        self._min_scale = 0.2
        self._max_scale = 6.0
        self._base_pixmap: Optional[QPixmap] = None
        self._drag_origin: Optional[QPointF] = None
        self._suppress_emit = False
        self._labeling_mode = False
        self._auto_label_mode = False   # right-click edit only (no rubber-band)
        self._label_start: Optional[QPointF] = None
        self._label_start_px: Optional[QPoint] = None
        self._label_current_px: Optional[QPoint] = None
        self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self._pending_scale: Optional[float] = None

        # Edit highlight bbox (for live preview during editing)
        self._edit_highlight: Optional[tuple] = None  # (x1, y1, x2, y2) normalized

    def eventFilter(self, obj, event: QEvent) -> bool:
        """Paint highlight overlay after the label paints itself."""
        if obj is self._label and event.type() == QEvent.Type.Paint:
            # Let the label paint itself first
            self._label.event(event)
            # Then paint our highlight overlay
            if self._edit_highlight:
                self._paint_edit_highlight()
            return True
        return super().eventFilter(obj, event)

    def set_placeholder(self, text: str) -> None:
        self._base_pixmap = None
        self._scale = 1.0
        self._label.setText(text)
        self._label.setPixmap(QPixmap())
        self._label.adjustSize()

        h_bar = self.horizontalScrollBar()
        v_bar = self.verticalScrollBar()
        if h_bar is not None:
            h_bar.setValue(0)
        if v_bar is not None:
            v_bar.setValue(0)

        self._emit_transform()

    def set_pixmap(self, pixmap: Optional[QPixmap], peer_scale: Optional[float] = None) -> None:
        self._base_pixmap = pixmap
        self._pending_scale = peer_scale
        self._scale = peer_scale if peer_scale else 1.0
        if pixmap and not pixmap.isNull():
            self._label.setText("")
            self.reset_transform()
            return
        self.set_placeholder("Image error")

    def reset_transform(self) -> None:
        if not self._base_pixmap or self._base_pixmap.isNull():
            return

        viewport_widget = self.viewport()
        if viewport_widget is None:
            return

        viewport = viewport_widget.size()
        if self._pending_scale is not None:
            target_scale = max(self._min_scale, min(self._max_scale, self._pending_scale))
            self._pending_scale = None
        elif viewport.width() <= 0 or viewport.height() <= 0:
            target_scale = 1.0
        else:
            fit_x = viewport.width() / self._base_pixmap.width()
            fit_y = viewport.height() / self._base_pixmap.height()
            # Use MIN to fit in viewport without overflow
            target_scale = min(self._max_scale, max(self._min_scale, min(fit_x, fit_y)))
        # Remember the scale that fits the current pixmap into this viewport
        self._fit_scale = target_scale
        self._scale = target_scale
        self._apply_scale()

        h_bar = self.horizontalScrollBar()
        v_bar = self.verticalScrollBar()
        if h_bar is not None:
            h_bar.setValue(0)
        if v_bar is not None:
            v_bar.setValue(0)

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
        if (self._labeling_mode or self._auto_label_mode) and event.button() == Qt.MouseButton.RightButton:
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
                # Use clamp=True to allow drawing boxes at image edges
                pos_px = self._label_pos(event, clamp=True)
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
            # Use clamp=True to allow finishing boxes at image edges
            end = self._image_pos(event, clamp=True)
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
        # In labeling/auto-label mode, right-click is handled by
        # mousePressEvent (labelDeleteRequested)
        if self._labeling_mode or self._auto_label_mode:
            event.accept()
            return
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

    def set_auto_label_mode(self, enabled: bool) -> None:
        """Enable right-click label editing without rubber-band drawing."""
        self._auto_label_mode = enabled

    def set_edit_highlight(self, bbox: Optional[tuple]) -> None:
        """Set a bbox to highlight during editing (x1, y1, x2, y2 normalized).

        This draws a dashed rectangle with P1/P2 corner labels for live preview.
        """
        self._edit_highlight = bbox
        self._label.update()  # Trigger repaint

    def clear_edit_highlight(self) -> None:
        """Clear the edit highlight."""
        self._edit_highlight = None
        self._label.update()  # Trigger repaint

    def get_pixmap_size(self) -> Optional[tuple]:
        """Get the size of the current base pixmap (width, height), or None if no pixmap."""
        if self._base_pixmap and not self._base_pixmap.isNull():
            return (self._base_pixmap.width(), self._base_pixmap.height())
        return None
    def _image_pos(self, event, clamp: bool = False) -> Optional[QPointF]:
        """Get normalized image position from mouse event.

        Args:
            event: Mouse event
            clamp: If True, clamp position to image bounds instead of returning None

        Returns:
            Normalized position (0-1) or None if outside bounds and clamp=False
        """
        if not self._label or not self._label.pixmap() or self._label.pixmap().isNull():
            return None
        label_pos = self._label.mapFrom(self.viewport(), event.position().toPoint())
        pix = self._label.pixmap()

        if clamp:
            # Clamp to image bounds (for bbox drawing at edges)
            x = max(0, min(label_pos.x(), pix.width()))
            y = max(0, min(label_pos.y(), pix.height()))
            x_norm = x / pix.width()
            y_norm = y / pix.height()
            return QPointF(x_norm, y_norm)
        else:
            # Strict bounds check (for click detection)
            if label_pos.x() < 0 or label_pos.y() < 0 or label_pos.x() > pix.width() or label_pos.y() > pix.height():
                return None
            x_norm = label_pos.x() / pix.width()
            y_norm = label_pos.y() / pix.height()
            return QPointF(x_norm, y_norm)

    def _label_pos(self, event, clamp: bool = False) -> Optional[QPoint]:
        """Get pixel position on label from mouse event.

        Args:
            event: Mouse event
            clamp: If True, clamp position to image bounds instead of returning None

        Returns:
            Pixel position on label or None if outside bounds and clamp=False
        """
        if not self._label or not self._label.pixmap() or self._label.pixmap().isNull():
            return None
        label_pos = self._label.mapFrom(self.viewport(), event.position().toPoint())
        pix = self._label.pixmap()

        if clamp:
            # Clamp to image bounds
            x = max(0, min(label_pos.x(), pix.width()))
            y = max(0, min(label_pos.y(), pix.height()))
            return QPoint(int(x), int(y))
        else:
            if label_pos.x() < 0 or label_pos.y() < 0 or label_pos.x() > pix.width() or label_pos.y() > pix.height():
                return None
            return label_pos  # type: ignore[return-value]

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
        # Interpret incoming scale as a zoom factor relative to the sender's fit-to-view.
        # Apply the same relative zoom on top of our own fit scale.
        target = self._fit_scale * max(self._min_scale, min(self._max_scale, scale))
        self._set_scale(target, emit=False)
        self._set_scroll_ratio(self.horizontalScrollBar(), h_ratio)
        self._set_scroll_ratio(self.verticalScrollBar(), v_ratio)
        self._suppress_emit = False

    def current_transform(self) -> tuple[float, float, float]:
        # Emit relative zoom factor, not absolute scale, so peer views with different
        # resolutions/aspect ratios still occupy their viewport equally.
        zoom_factor = self._scale / self._fit_scale if self._fit_scale else self._scale
        return (zoom_factor, self._ratio(self.horizontalScrollBar()), self._ratio(self.verticalScrollBar()))

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
        # Trigger repaint of highlight overlay
        self._label.update()

    def _paint_edit_highlight(self) -> None:
        """Paint the edit highlight overlay with P1/P2 labels directly on the label widget."""
        if not self._edit_highlight or not self._label.pixmap():
            return

        x1, y1, x2, y2 = self._edit_highlight
        pix = self._label.pixmap()
        if pix.isNull():
            return

        img_w = pix.width()
        img_h = pix.height()

        # Convert normalized coords to pixel coords on current pixmap
        px1 = int(x1 * img_w)
        py1 = int(y1 * img_h)
        px2 = int(x2 * img_w)
        py2 = int(y2 * img_h)

        painter = QPainter(self._label)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw dashed rectangle
        pen = QPen(QColor(255, 165, 0))  # Orange
        pen.setWidth(2)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawRect(px1, py1, px2 - px1, py2 - py1)

        # Draw corner circles and labels
        font = QFont()
        font.setBold(True)
        font.setPointSize(10)
        painter.setFont(font)

        # P1 (top-left) - cyan
        p1_color = QColor(0, 200, 255)
        pen.setColor(p1_color)
        pen.setStyle(Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.setBrush(p1_color)
        painter.drawEllipse(QPoint(px1, py1), 6, 6)

        # P1 label
        painter.setPen(QPen(Qt.GlobalColor.white))
        painter.drawText(px1 + 10, py1 + 5, "P1")

        # P2 (bottom-right) - magenta
        p2_color = QColor(255, 100, 255)
        pen.setColor(p2_color)
        painter.setPen(pen)
        painter.setBrush(p2_color)
        painter.drawEllipse(QPoint(px2, py2), 6, 6)

        # P2 label
        painter.setPen(QPen(Qt.GlobalColor.white))
        painter.drawText(px2 - 25, py2 - 10, "P2")

        painter.end()

    def _set_scroll_ratio(self, scrollbar, ratio: float) -> None:  # type: ignore[override]
        ratio = max(0.0, min(1.0, ratio))
        maximum = scrollbar.maximum()
        scrollbar.setValue(int(ratio * maximum) if maximum > 0 else 0)

    def _ratio(self, scrollbar) -> float:  # type: ignore[override]
        maximum = scrollbar.maximum()
        if maximum <= 0:
            return 0.0
        return float(scrollbar.value() / maximum)

    def _emit_transform(self) -> None:
        if self._suppress_emit:
            return
        self.transformChanged.emit(*self.current_transform())

