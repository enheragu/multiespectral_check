"""Background sub-pixel refinement for detected chessboard corners."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Dict, Iterable, List, Optional, Tuple

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

from backend.services.dataset_session import DatasetSession
from backend.utils.calibration import refine_corners_from_path
from common.log_utils import log_debug, log_info, log_warning


@dataclass
class _RefinementChannelPayload:
    image_path: Path
    corners: List[List[float]]


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
            refined: Dict[str, List[List[float]]] = {}
            for channel, data in self.payload.items():
                self._ensure_not_cancelled()
                result = refine_corners_from_path(data.image_path, self.pattern_size, data.corners)
                if result:
                    refined[channel] = result
            if refined:
                self._ensure_not_cancelled()
                try:
                    log_debug(f"✅ Refinement completed for {self.base}: {list(refined.keys())}", "REFINER")
                    self.signals.completed.emit(self.base, refined)
                    self.signals.finished.emit(self.base, True)
                except RuntimeError:
                    pass  # Signals object may be deleted if GUI closed
            else:
                try:
                    log_debug(f"⚠️ Refinement produced no results for {self.base}", "REFINER")
                    self.signals.failed.emit(self.base, "Could not refine corners")
                    self.signals.finished.emit(self.base, False)
                except RuntimeError:
                    pass  # Signals object may be deleted if GUI closed
        except Exception as exc:  # noqa: BLE001
            try:
                log_warning(f"❌ Refinement failed for {self.base}: {exc}", "REFINER")
                self.signals.failed.emit(self.base, str(exc))
                self.signals.finished.emit(self.base, False)
            except RuntimeError:
                pass  # Signals object may be deleted if GUI closed


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
        log_debug(f"[CalibrationRefiner] refine() called with bases={list(bases)[:5]}...")
        # Support both loader and collection
        if not self.session.loader and not self.session.collection:
            log_warning("[CalibrationRefiner] No loader or collection in session, returning 0")
            return 0
        jobs: List[Tuple[str, Dict[str, _RefinementChannelPayload]]] = []
        bases_list = list(bases)
        log_debug(f"[CalibrationRefiner] Processing {len(bases_list)} bases")
        for base in bases_list:
            # Use get_corners() for lazy loading from disk
            bucket = self.session.get_corners(base) or {}
            if not bucket:
                log_debug(f"[CalibrationRefiner] No corners for base={base}")
                continue
            channels: Dict[str, _RefinementChannelPayload] = {}
            for channel in ("lwir", "visible"):
                corners = bucket.get(channel)
                if not corners:
                    continue
                # Use session.get_image_path which supports both loader and collection
                image_path = self.session.get_image_path(base, channel)
                if not image_path or not image_path.exists():
                    log_debug(f"[CalibrationRefiner] No image path for base={base} channel={channel}")
                    continue
                channels[channel] = _RefinementChannelPayload(image_path=image_path, corners=list(corners))
            if channels:
                jobs.append((base, channels))
        if not jobs:
            log_warning("[CalibrationRefiner] No jobs to process, returning 0")
            return 0
        log_info(f"[CalibrationRefiner] Starting {len(jobs)} refinement jobs")
        self._pending = len(jobs)
        self._succeeded = 0
        self._failed = 0
        self._tasks.clear()
        for base, payload in jobs:
            task = _CalibrationRefineTask(base, self.pattern_size, payload)
            self._register_task(task)
            if self.thread_pool is not None:
                self.thread_pool.start(task)
            else:
                log_warning("[CalibrationRefiner] No thread_pool available!")
        log_info(f"[CalibrationRefiner] Returning pending={self._pending}")
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
