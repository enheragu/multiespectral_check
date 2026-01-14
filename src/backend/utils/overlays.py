"""Helpers to paint grids, overlays, and labels on rendered pixmaps."""
from __future__ import annotations

from typing import List, Tuple, Sequence

from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPixmap


# Grid mode constants
GRID_OFF = "off"
GRID_THIRDS = "thirds"
GRID_DETAILED = "detailed"


def paint_rule_of_thirds(
    painter: QPainter,
    width: int,
    height: int,
    detailed: bool = False,
) -> None:
    """Paint rule of thirds grid, optionally with sub-divisions.

    Args:
        painter: QPainter to draw with
        width: Image width
        height: Image height
        detailed: If True, also draw sub-thirds within each third (9 divisions total)
    """
    base_pen_width = max(1, int(max(width, height) / 400))

    # Main thirds - more opaque
    main_pen = QPen(QColor(255, 255, 255, 180))
    main_pen.setWidth(base_pen_width)
    painter.setPen(main_pen)

    third_w = width / 3
    third_h = height / 3
    for idx in (1, 2):
        x = int(idx * third_w)
        painter.drawLine(x, 0, x, height)
        y = int(idx * third_h)
        painter.drawLine(0, y, width, y)

    # Sub-thirds - more subtle
    if detailed:
        sub_pen = QPen(QColor(255, 255, 255, 80))
        sub_pen.setWidth(max(1, base_pen_width - 1))
        painter.setPen(sub_pen)

        ninth_w = width / 9
        ninth_h = height / 9
        for idx in range(1, 9):
            # Skip the main third lines (3, 6)
            if idx % 3 == 0:
                continue
            x = int(idx * ninth_w)
            painter.drawLine(x, 0, x, height)
            y = int(idx * ninth_h)
            painter.drawLine(0, y, width, y)


def draw_overlay_labels(
    painter: QPainter,
    width: int,
    height: int,
    entries: List[Tuple[str, QColor]],
) -> None:
    font = QFont(painter.font())
    font.setPointSize(max(13, int(max(width, height) / 35)))
    painter.setFont(font)
    metrics = painter.fontMetrics()
    margin = max(12, int(max(width, height) / 50))
    line_height = metrics.height() + 4
    for idx, (text, color) in enumerate(entries):
        if not text:
            continue
        painter.setPen(color)
        text_width = metrics.horizontalAdvance(text)
        x = width - margin - text_width
        y = margin + line_height * (idx + 1)
        painter.drawText(x, y, text)


def draw_label_boxes(
    painter: QPainter,
    width: int,
    height: int,
    boxes: Sequence[Tuple[str, float, float, float, float, QColor]],
) -> None:
    """Draw YOLO-format boxes (normalized) with class text on top."""
    if not boxes:
        return
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    for cls_name, x_c, y_c, w_norm, h_norm, color in boxes:
        pen_width = max(2, int(max(width, height) / 200))
        inset = pen_width / 2.0
        pen = QPen(color)
        pen.setWidth(pen_width)
        painter.setPen(pen)
        abs_w = max(1.0, w_norm * width)
        abs_h = max(1.0, h_norm * height)
        abs_x = x_c * width - abs_w / 2.0
        abs_y = y_c * height - abs_h / 2.0
        rect = QRectF(abs_x + inset, abs_y + inset, abs_w - pen_width, abs_h - pen_width)
        painter.drawRect(rect)

        # Draw class label above the box
        font = QFont(painter.font())
        font.setPointSize(max(12, int(max(width, height) / 50)))
        painter.setFont(font)
        painter.setPen(QColor(color))
        metrics = painter.fontMetrics()
        text = cls_name or "?"
        text_w = metrics.horizontalAdvance(text)
        text_h = metrics.height()
        text_x = abs_x + inset
        text_y = max(inset + text_h, abs_y + inset + text_h)
        bg_x = int(round(text_x - 2))
        bg_y = int(round(text_y - text_h))
        bg_w = int(round(text_w + 4))
        bg_h = int(round(text_h))
        painter.fillRect(bg_x, bg_y, bg_w, bg_h, QColor(0, 0, 0, 160))
        painter.setPen(QColor("white"))
        painter.drawText(int(round(text_x)), int(round(text_y - metrics.descent() // 2)), text)


def draw_reason_overlay(
    painter: QPainter,
    canvas: QPixmap,
    color: QColor,
    text: str,
    pen_width: int,
) -> None:
    pen = QPen(color)
    pen.setWidth(pen_width)
    painter.setPen(pen)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    inset = pen_width / 2.0
    w = canvas.width()
    h = canvas.height()
    max_x = max(inset, w - inset - 1.0)
    max_y = max(inset, h - inset - 1.0)
    painter.drawLine(QPointF(inset, inset), QPointF(max_x, max_y))
    painter.drawLine(QPointF(inset, max_y), QPointF(max_x, inset))
    rect_width = max(1.0, w - pen_width - inset)
    rect_height = max(1.0, h - pen_width - inset)
    rect = QRectF(inset, inset, rect_width, rect_height)
    painter.drawRect(rect)


def draw_calibration_overlay(
    painter: QPainter,
    canvas: QPixmap,
    color: QColor,
    pen_width: int,
) -> None:
    calibration_pen = QPen(color)
    calibration_pen.setWidth(pen_width)
    painter.setPen(calibration_pen)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    inset = pen_width / 2.0  # align stroke on the pixel grid while staying at the edge
    rect_width = max(1.0, canvas.width() - pen_width - inset)
    rect_height = max(1.0, canvas.height() - pen_width - inset)
    rect = QRectF(inset, inset, rect_width, rect_height)
    painter.drawRect(rect)
