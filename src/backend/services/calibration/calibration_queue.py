"""Debounced queue for calibration tasks to avoid spamming the solver on rapid marks."""

from __future__ import annotations

from typing import Callable, Optional, Set

from PyQt6.QtCore import QObject, QTimer


class DeferredCalibrationQueue(QObject):
    """Debounce calibration scheduling so repeated marks batch into one flush."""

    def __init__(
        self,
        *,
        parent: Optional[QObject],
        interval_ms: int,
        validator: Callable[[str], bool],
        scheduler: Callable[[str, bool], bool],
    ) -> None:
        super().__init__(parent)
        self._validator = validator
        self._scheduler = scheduler
        self._pending: Set[str] = set()
        self._pending_forced: Set[str] = set()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self.flush)

    def defer(self, base: Optional[str], *, force: bool = False) -> None:
        if not base or not self._validator(base):
            return
        self._pending.add(base)
        if force:
            self._pending_forced.add(base)
        if not self._timer.isActive():
            self._timer.start()

    def cancel(self, base: Optional[str]) -> None:
        if not base:
            return
        self._pending.discard(base)
        self._pending_forced.discard(base)
        if not self._pending and self._timer.isActive():
            self._timer.stop()

    def clear(self) -> None:
        self._pending.clear()
        self._pending_forced.clear()
        if self._timer.isActive():
            self._timer.stop()

    def flush(self) -> None:
        if not self._pending:
            return
        pending = list(self._pending)
        self._pending.clear()
        for base in pending:
            force = base in self._pending_forced
            self._pending_forced.discard(base)
            self._scheduler(base, force)
        if self._timer.isActive():
            self._timer.stop()
