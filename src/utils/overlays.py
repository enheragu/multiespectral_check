"""Helpers to paint grids, overlays, and labels on rendered pixmaps."""
from __future__ import annotations

from typing import List, Tuple, Optional

from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPixmap


def paint_rule_of_thirds(painter: QPainter, width: int, height: int) -> None:
    grid_pen = QPen(QColor(255, 255, 255, 160))
    grid_pen.setWidth(max(1, int(max(width, height) / 400)))
    painter.setPen(grid_pen)
    third_w = width / 3
    third_h = height / 3
    for idx in (1, 2):
        x = int(idx * third_w)
        painter.drawLine(x, 0, x, height)
        y = int(idx * third_h)
        painter.drawLine(0, y, width, y)


def draw_overlay_labels(
    painter: QPainter,
    width: int,
    height: int,
    entries: List[Tuple[str, QColor]],
) -> None:
    font = QFont(painter.font())
    font.setPointSize(max(14, int(max(width, height) / 30)))
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
