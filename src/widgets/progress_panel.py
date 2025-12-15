"""Compact progress bar + cancel button widget."""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QWidget,
)

from services.progress_tracker import ProgressSnapshot
from widgets import style


class ProgressPanel(QWidget):
    """Displays the shared progress bar and an optional cancel button."""

    cancelRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._has_snapshot = False
        self._cancel_enabled = False
        self._cancel_tooltip = ""
        self._build_ui()
        self.hide()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 0, 0)
        layout.setSpacing(8)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumWidth(260)
        self.progress_bar.setFixedHeight(style.BUTTON_HEIGHT)
        self.progress_bar.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        )
        layout.addWidget(self.progress_bar)
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.setVisible(False)
        self.cancel_button.setFixedHeight(style.BUTTON_HEIGHT)
        self.cancel_button.setObjectName("progress_cancel_button")
        self.cancel_button.setStyleSheet(style.scoped_button_style(self.cancel_button.objectName()))
        self.cancel_button.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        )
        self.cancel_button.clicked.connect(self.cancelRequested.emit)
        layout.addWidget(self.cancel_button)

    def clear(self) -> None:
        self._has_snapshot = False
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")
        self.hide()
        self._apply_cancel_state()

    def set_snapshot(self, snapshot: Optional[ProgressSnapshot]) -> None:
        if snapshot is None:
            self.clear()
            return
        self._has_snapshot = True
        self.show()
        label_text = snapshot.label or ""
        if snapshot.busy or snapshot.total is None:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(1)
            self.progress_bar.setFormat(label_text)
        else:
            total = max(1, snapshot.total)
            value = max(0, min(snapshot.value, total))
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(value)
            suffix = f" {value}/{total}" if total else ""
            self.progress_bar.setFormat(f"{label_text}{suffix}".strip())
        self._apply_cancel_state()

    def set_cancel_state(self, enabled: bool, tooltip: str = "") -> None:
        self._cancel_enabled = enabled
        self._cancel_tooltip = tooltip
        self._apply_cancel_state()

    def _apply_cancel_state(self) -> None:
        if not self._has_snapshot:
            self.cancel_button.setVisible(False)
            return
        self.cancel_button.setVisible(True)
        self.cancel_button.setEnabled(self._cancel_enabled)
        if self._cancel_tooltip:
            self.cancel_button.setToolTip(self._cancel_tooltip)
        else:
            self.cancel_button.setToolTip("")
