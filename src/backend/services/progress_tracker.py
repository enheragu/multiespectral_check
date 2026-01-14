"""Utility to orchestrate a shared progress bar across background tasks."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class ProgressSnapshot:
    label: str
    value: int = 0
    total: Optional[int] = None
    busy: bool = False


class ProgressTracker:
    """Keeps track of long-running tasks and exposes the top-most snapshot."""

    def __init__(self, callback: Callable[[Optional[ProgressSnapshot]], None]) -> None:
        self._tasks: "OrderedDict[str, ProgressSnapshot]" = OrderedDict()
        self._callback = callback

    def set_busy(self, task_id: str, label: str) -> None:
        snapshot = ProgressSnapshot(label=label, busy=True)
        self._tasks.pop(task_id, None)
        self._tasks[task_id] = snapshot
        self._emit()

    def start(self, task_id: str, label: str, total: int) -> None:
        snapshot = ProgressSnapshot(label=label, total=max(1, total), value=0, busy=False)
        self._tasks.pop(task_id, None)
        self._tasks[task_id] = snapshot
        self._emit()

    def start_task(self, task_id: str, label: str, total: int):
        """Convenience: start determinate task and return an updater callable."""
        self.start(task_id, label, total)

        def _update(done: int, override_total: Optional[int] = None) -> None:
            self.update(task_id, done, override_total)

        return _update

    def update(self, task_id: str, value: int, total: Optional[int] = None) -> None:
        snapshot = self._tasks.get(task_id)
        if not snapshot:
            return
        if total is not None and total > 0:
            snapshot.total = total
        if snapshot.total is not None:
            snapshot.value = max(0, min(value, snapshot.total))
            snapshot.busy = False
        self._emit()

    def finish(self, task_id: str) -> None:
        if task_id in self._tasks:
            self._tasks.pop(task_id, None)
            self._emit()

    def clear(self) -> None:
        if not self._tasks:
            return
        self._tasks.clear()
        self._emit()

    def _emit(self) -> None:
        snapshot = next(reversed(self._tasks.values())) if self._tasks else None
        self._callback(snapshot)