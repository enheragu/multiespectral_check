"""Schedules and manages calibration detection jobs across the dataset."""

from __future__ import annotations

from collections import deque
from threading import Event
from typing import Deque, Dict, Iterable, Optional, Tuple

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

from dataset_loader import DatasetLoader
from utils.calibration import analyze_pair_from_paths


class CalibrationTaskSignals(QObject):
    completed = pyqtSignal(int, str, dict, dict)
    failed = pyqtSignal(int, str, str)
    finished = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()


class CalibrationTask(QRunnable):
    def __init__(
        self,
        epoch: int,
        base: str,
        loader: DatasetLoader,
        pattern_size: Tuple[int, int],
    ) -> None:
        super().__init__()
        self.epoch = epoch
        self.base = base
        self.loader = loader
        self.pattern_size = pattern_size
        self.signals = CalibrationTaskSignals()
        self._cancel_event = Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def _ensure_not_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise RuntimeError("Calibration detection cancelled")

    def run(self) -> None:  # noqa: D401
        try:
            self._ensure_not_cancelled()
            lwir_path = self.loader.get_image_path(self.base, "lwir")
            vis_path = self.loader.get_image_path(self.base, "visible")
            if lwir_path is None or vis_path is None:
                raise FileNotFoundError(f"Missing image path(s) for {self.base}")
            self._ensure_not_cancelled()
            results, corners = analyze_pair_from_paths(
                self.base,
                lwir_path,
                vis_path,
                self.pattern_size,
            )
            self._ensure_not_cancelled()
            self.signals.completed.emit(self.epoch, self.base, results, corners)
        except Exception as exc:  # noqa: BLE001
            self.signals.failed.emit(self.epoch, self.base, str(exc))
        finally:
            self.signals.finished.emit(self.base)


class CalibrationController(QObject):
    """Manage background calibration detection tasks and emit results."""

    calibrationReady = pyqtSignal(str, dict, dict)
    calibrationFailed = pyqtSignal(str, str)
    activityChanged = pyqtSignal(int)

    def __init__(
        self,
        session,
        pattern_size: Tuple[int, int],
        thread_pool: Optional[QThreadPool] = None,
        max_workers: int = 2,
    ) -> None:
        super().__init__()
        self.session = session
        self.pattern_size = pattern_size
        self.thread_pool = thread_pool or QThreadPool.globalInstance()
        self.max_workers = max(1, max_workers)
        self._epoch = 0
        self._queue: Deque[str] = deque()
        self._running: Dict[str, CalibrationTask] = {}

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------
    def reset(self) -> None:
        self.cancel_all()

    def schedule(
        self,
        base: str,
        *,
        force: bool = False,
        priority: str = "normal",
    ) -> bool:
        if not self._can_schedule(base, force):
            return False
        if force:
            self._remove_from_queue(base)
        if priority == "high":
            self._queue.appendleft(base)
        else:
            self._queue.append(base)
        self._maybe_dispatch()
        self._emit_activity()
        return True

    def prefetch(
        self,
        bases: Iterable[str],
        *,
        force: bool = False,
    ) -> int:
        count = 0
        for base in bases:
            count += int(self.schedule(base, force=force))
        return count

    def cancel_all(self) -> None:
        if not (self._queue or self._running):
            self._epoch += 1
            self._emit_activity()
            return
        self._epoch += 1
        for task in self._running.values():
            task.cancel()
        self._queue.clear()
        self._running.clear()
        self._emit_activity()

    # --------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------
    def _can_schedule(self, base: Optional[str], force: bool) -> bool:
        if not base or not self.session.loader:
            return False
        if not force and (base in self._running or self._is_queued(base)):
            return False
        return True

    def _is_queued(self, base: str) -> bool:
        return base in self._queue

    def _remove_from_queue(self, base: str) -> None:
        if not self._queue:
            return
        self._queue = deque(item for item in self._queue if item != base)

    def _maybe_dispatch(self) -> None:
        if not self.session.loader:
            return
        while len(self._running) < self.max_workers and self._queue:
            base = self._queue.popleft()
            task = CalibrationTask(self._epoch, base, self.session.loader, self.pattern_size)
            self._register_task(task)
            self.thread_pool.start(task)

    def _register_task(self, task: CalibrationTask) -> None:
        base = task.base
        self._running[base] = task
        task.signals.completed.connect(self._handle_task_completed)
        task.signals.failed.connect(self._handle_task_failed)
        task.signals.finished.connect(self._handle_task_finished)
        self._emit_activity()

    def _handle_task_completed(
        self,
        epoch: int,
        base: str,
        results: dict,
        corners: dict,
    ) -> None:
        if epoch != self._epoch:
            return
        self.calibrationReady.emit(base, results, corners)

    def _handle_task_failed(self, epoch: int, base: str, message: str) -> None:
        if epoch != self._epoch:
            return
        self.calibrationFailed.emit(base, message)

    def _handle_task_finished(self, base: str) -> None:
        self._running.pop(base, None)
        self._maybe_dispatch()
        self._emit_activity()

    def _emit_activity(self) -> None:
        pending = len(self._queue) + len(self._running)
        self.activityChanged.emit(pending)
