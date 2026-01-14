"""Schedules and manages calibration detection jobs across the dataset."""

from __future__ import annotations

from threading import Event
import time
from typing import Dict, Iterable, Optional, Tuple

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

from backend.dataset_loader import DatasetLoader
from backend.services.base_queue_controller import BaseQueueController
from backend.utils.calibration import analyze_pair_from_paths
from common.log_utils import log_debug, log_info, log_warning, log_error, log_perf



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
        loader: Optional[DatasetLoader],
        pattern_size: Tuple[int, int],
        local_base: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.epoch = epoch
        self.base = base  # Full base (may be namespaced for collections)
        self.loader = loader
        self.pattern_size = pattern_size
        self.local_base = local_base or base  # Local base for file access
        self.signals = CalibrationTaskSignals()
        self._cancel_event = Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def _ensure_not_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise RuntimeError("Calibration detection cancelled")

    def run(self) -> None:  # noqa: D401
        start = time.perf_counter()
        try:
            self._ensure_not_cancelled()
            if not self.loader:
                raise RuntimeError(f"No loader available for {self.base}")
            # Use local_base for file access (handles namespaced bases in collections)
            lwir_path = self.loader.get_image_path(self.local_base, "lwir")
            vis_path = self.loader.get_image_path(self.local_base, "visible")
            if lwir_path is None or vis_path is None:
                raise FileNotFoundError(f"Missing image path(s) for {self.base}")
            self._ensure_not_cancelled()
            results, corners = analyze_pair_from_paths(
                self.base,  # Use full base for result tracking
                lwir_path,
                vis_path,
                self.pattern_size,
            )
            self._ensure_not_cancelled()
            try:
                self.signals.completed.emit(self.epoch, self.base, results, corners)
            except RuntimeError:
                # Signals object may be deleted if GUI closed
                pass
        except Exception as exc:  # noqa: BLE001
            try:
                self.signals.failed.emit(self.epoch, self.base, str(exc))
            except RuntimeError:
                # Signals object may be deleted if GUI closed
                pass
        finally:
            log_perf(f"detect base={self.base} in {time.perf_counter()-start:.3f}s")
            try:
                self.signals.finished.emit(self.base)
            except RuntimeError:
                # Signals object may be deleted if GUI closed
                pass


class CalibrationController(BaseQueueController[str, CalibrationTask]):
    """Manage background calibration detection tasks and emit results.

    Uses BaseQueueController for queue management, eliminating ~80 lines of duplicated code.
    """

    calibrationReady = pyqtSignal(str, dict, dict)
    calibrationFailed = pyqtSignal(str, str)

    def __init__(
        self,
        session,
        pattern_size: Tuple[int, int],
        thread_pool: Optional[QThreadPool] = None,
        max_workers: int = 2,
    ) -> None:
        super().__init__(thread_pool=thread_pool, max_concurrent=max_workers)
        self.session = session
        self.pattern_size = pattern_size

    # --------------------------------------------------------------
    # BaseQueueController abstract method implementations
    # --------------------------------------------------------------

    def _get_loader_for_base(self, base: str) -> Optional[DatasetLoader]:
        """Get the appropriate loader for a base (handles collections with namespaced bases)."""
        # Direct dataset
        if self.session.loader:
            return self.session.loader
        # Collection - need to find child loader for namespaced base
        if self.session.collection:
            return self.session.collection.get_loader_for_base(base)
        return None

    def _create_task(self, base: str) -> CalibrationTask:
        """Create calibration task for the given base."""
        loader = self._get_loader_for_base(base)
        # For collections, we need to use the local base name (without namespace)
        local_base = base
        if self.session.collection and "/" in base:
            local_base = base.split("/", 1)[1]
        task = CalibrationTask(
            self._epoch,
            base,  # Keep full base for tracking
            loader,
            self.pattern_size,
            local_base=local_base,  # Pass local base for file access
        )
        task.signals.completed.connect(self._handle_task_completed)
        task.signals.failed.connect(self._handle_task_failed)
        task.signals.finished.connect(self._handle_task_finished)
        return task

    def _can_schedule_item(self, base: str, force: bool) -> bool:
        """Check if base can be scheduled."""
        if not base:
            return False
        # Check we have either loader or collection
        if not self.session.loader and not self.session.collection:
            return False
        if not force and (base in self._running or self._is_queued(base)):
            return False
        return True

    def _get_item_key(self, base: str) -> str:
        """Get key for base (base itself is the key)."""
        return base

    def _cancel_task(self, task: CalibrationTask) -> None:
        """Cancel a calibration task."""
        task.cancel()

    # --------------------------------------------------------------
    # Public API extensions
    # --------------------------------------------------------------

    def prefetch(
        self,
        bases: Iterable[str],
        *,
        force: bool = False,
    ) -> int:
        """Schedule multiple bases for detection.

        Args:
            bases: Image bases to detect
            force: Force reschedule even if queued

        Returns:
            Number of bases successfully scheduled
        """
        count = 0
        for base in bases:
            count += int(self.schedule(base, force=force))
        return count

    # --------------------------------------------------------------
    # Signal handlers
    # --------------------------------------------------------------

    def _handle_task_completed(
        self,
        epoch: int,
        base: str,
        results: Dict,
        corners: Dict,
    ) -> None:
        if epoch != self._epoch:
            return
        self.calibrationReady.emit(base, results, corners)

    def _handle_task_failed(self, epoch: int, base: str, message: str) -> None:
        if epoch != self._epoch:
            return
        self.calibrationFailed.emit(base, message)

    def _handle_task_finished(self, base: str) -> None:
        self._mark_task_finished(base)
