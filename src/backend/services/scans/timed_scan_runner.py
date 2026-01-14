"""Reusable queue runner for timed background scans.

Manages a queue of indices, inflight tracking, and periodic scheduling with a QTimer.
Clients supply a scheduling function that receives an index and returns the base id
(or None) when the job is accepted. This keeps the runner generic across different
scan types (signatures, quality, patterns, etc.).
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Optional, Set

from PyQt6.QtCore import QObject, QTimer


class TimedScanRunner(QObject):
    def __init__(
        self,
        *,
        parent: Optional[QObject],
        max_inflight: int,
        timer_interval_ms: int,
        schedule_fn: Callable[[int], Optional[str]],
        on_progress: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(parent)
        self.max_inflight = max_inflight
        self.timer_interval_ms = timer_interval_ms
        self.schedule_fn = schedule_fn
        self.on_progress = on_progress

        self._queue: Deque[int] = deque()
        self._pending: Set[str] = set()
        self._total: int = 0
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._drain)

    # Public API -------------------------------------------------
    def reset(self) -> None:
        self._queue.clear()
        self._pending.clear()
        self._total = 0
        if self._timer.isActive():
            self._timer.stop()
        self._notify()

    def start(self, targets: list[int]) -> None:
        self.reset()
        if not targets:
            return
        self._queue.extend(targets)
        self._total = len(targets)
        self._notify()
        self._timer.start(0)

    def mark_completed(self, base: Optional[str]) -> None:
        if base:
            self._pending.discard(base)
        self._notify()
        if self._queue:
            self._timer.start(self.timer_interval_ms)

    # Properties -------------------------------------------------
    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def queued_count(self) -> int:
        return len(self._queue)

    @property
    def total(self) -> int:
        return self._total

    # Internal ---------------------------------------------------
    def _drain(self) -> None:
        if not self._queue:
            self._notify()
            return
        available = max(0, self.max_inflight - len(self._pending))
        scheduled = 0
        while available > 0 and self._queue:
            index = self._queue.popleft()
            base = self.schedule_fn(index)
            if base:
                self._pending.add(base)
                scheduled += 1
            available -= 1
        self._notify()
        if self._queue and scheduled == 0:
            self._timer.start(self.timer_interval_ms)

    def _notify(self) -> None:
        if self.on_progress:
            try:
                self.on_progress()
            except Exception:
                pass

