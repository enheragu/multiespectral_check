"""Background sub-pixel refinement for detected chessboard corners."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Dict, Iterable, List, Optional, Tuple

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

from services.dataset_session import DatasetSession
from utils.calibration import refine_corners_from_path


@dataclass
class _RefinementChannelPayload:
    image_path: Path
    corners: List[Tuple[float, float]]


class _RefinementTaskSignals(QObject):
    completed = pyqtSignal(str, dict)
    failed = pyqtSignal(str, str)
    finished = pyqtSignal(str, bool)

    def __init__(self) -> None:
        super().__init__()


class _CalibrationRefineTask(QRunnable):
    def __init__(
        self,
        base: str,
        pattern_size: Tuple[int, int],
        payload: Dict[str, _RefinementChannelPayload],
    ) -> None:
        super().__init__()
        self.base = base
        self.pattern_size = pattern_size
        self.payload = payload
        self.signals = _RefinementTaskSignals()
        self._cancel_event = Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def _ensure_not_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise RuntimeError("Corner refinement cancelled")

    def run(self) -> None:  # noqa: D401
        try:
            refined: Dict[str, List[Tuple[float, float]]] = {}
            for channel, data in self.payload.items():
                self._ensure_not_cancelled()
                result = refine_corners_from_path(data.image_path, self.pattern_size, data.corners)
                if result:
                    refined[channel] = result
            if refined:
                self._ensure_not_cancelled()
                self.signals.completed.emit(self.base, refined)
                self.signals.finished.emit(self.base, True)
            else:
                self.signals.failed.emit(self.base, "Could not refine corners")
                self.signals.finished.emit(self.base, False)
        except Exception as exc:  # noqa: BLE001
            self.signals.failed.emit(self.base, str(exc))
            self.signals.finished.emit(self.base, False)


class CalibrationRefiner(QObject):
    """Schedule background sub-pixel refinement jobs for calibration candidates."""

    refinementReady = pyqtSignal(str, dict)
    refinementFailed = pyqtSignal(str, str)
    batchFinished = pyqtSignal(int, int)

    def __init__(
        self,
        session: DatasetSession,
        pattern_size: Tuple[int, int],
        thread_pool: Optional[QThreadPool] = None,
    ) -> None:
        super().__init__()
        self.session = session
        self.pattern_size = pattern_size
        self.thread_pool = thread_pool or QThreadPool.globalInstance()
        self._pending = 0
        self._succeeded = 0
        self._failed = 0
        self._tasks: Dict[str, _CalibrationRefineTask] = {}

    def refine(self, bases: Iterable[str]) -> int:
        loader = self.session.loader
        if not loader:
            return 0
        jobs: List[Tuple[str, Dict[str, _RefinementChannelPayload]]] = []
        for base in bases:
            bucket = self.session.state.calibration_corners.get(base, {})
            if not bucket:
                continue
            channels: Dict[str, _RefinementChannelPayload] = {}
            for channel in ("lwir", "visible"):
                corners = bucket.get(channel)
                if not corners:
                    continue
                image_path = loader.get_image_path(base, channel)
                if not image_path or not image_path.exists():
                    continue
                channels[channel] = _RefinementChannelPayload(image_path=image_path, corners=list(corners))
            if channels:
                jobs.append((base, channels))
        if not jobs:
            return 0
        self._pending = len(jobs)
        self._succeeded = 0
        self._failed = 0
        self._tasks.clear()
        for base, payload in jobs:
            task = _CalibrationRefineTask(base, self.pattern_size, payload)
            self._register_task(task)
            self.thread_pool.start(task)
        return self._pending

    def cancel(self) -> bool:
        if not self._tasks:
            return False
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()
        self._pending = 0
        self._succeeded = 0
        self._failed = 0
        self.batchFinished.emit(0, 0)
        return True

    def _register_task(self, task: _CalibrationRefineTask) -> None:
        task.signals.completed.connect(self._handle_task_completed)
        task.signals.failed.connect(self._handle_task_failed)
        task.signals.finished.connect(self._handle_task_finished)
        self._tasks[task.base] = task

    def _handle_task_completed(self, base: str, refined: dict) -> None:
        self.refinementReady.emit(base, refined)

    def _handle_task_failed(self, base: str, message: str) -> None:
        self.refinementFailed.emit(base, message)

    def _handle_task_finished(self, base: str, succeeded: bool) -> None:
        self._tasks.pop(base, None)
        if self._pending <= 0:
            return
        if succeeded:
            self._succeeded += 1
        else:
            self._failed += 1
        self._pending -= 1
        if self._pending <= 0:
            self.batchFinished.emit(self._succeeded, self._failed)
