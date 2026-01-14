"""Runs duplicate-signature jobs in the thread pool and emits progress signals."""

from __future__ import annotations

from threading import Event
from typing import Optional, Tuple

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

from backend.dataset_loader import DatasetLoader
from backend.services.indexed_queue_controller import IndexedQueueController
from backend.utils.duplicates import compute_signature_from_path


class SignatureTaskSignals(QObject):
    completed = pyqtSignal(int, int, str, object, object)
    failed = pyqtSignal(int, str, str)
    finished = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()


class SignatureTask(QRunnable):
    def __init__(self, epoch: int, index: int, base: str, loader: DatasetLoader):
        super().__init__()
        self.epoch = epoch
        self.index = index
        self.base = base
        self.loader = loader
        self.signals = SignatureTaskSignals()
        self._cancel_event = Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def _ensure_not_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise RuntimeError("Signature sweep cancelled")

    def run(self) -> None:
        try:
            self._ensure_not_cancelled()
            if not self.loader:
                self.signals.completed.emit(self.epoch, self.index, self.base, None, None)
                return
            lwir_path = self.loader.get_image_path(self.base, "lwir")
            vis_path = self.loader.get_image_path(self.base, "visible")
            self._ensure_not_cancelled()
            lwir_sig = compute_signature_from_path(lwir_path)
            self._ensure_not_cancelled()
            vis_sig = compute_signature_from_path(vis_path)
            self._ensure_not_cancelled()
            self.signals.completed.emit(self.epoch, self.index, self.base, lwir_sig, vis_sig)
        except Exception as exc:  # noqa: BLE001
            self.signals.failed.emit(self.epoch, self.base, str(exc))
        finally:
            self.signals.finished.emit(self)


class SignatureController(IndexedQueueController[str, SignatureTask]):
    """Controller for signature scan tasks with indexed queue management."""

    signatureReady = pyqtSignal(int, str, object, object)
    signatureFailed = pyqtSignal(str, str)

    def __init__(self, thread_pool: Optional[QThreadPool] = None) -> None:
        super().__init__(thread_pool=thread_pool, max_concurrent=4)
        self.loader: Optional[DatasetLoader] = None

    def set_loader(self, loader: Optional[DatasetLoader]) -> None:
        """Set the dataset loader for signature scans."""
        self.loader = loader

    def schedule_for_scan(self, index: int, base: Optional[str], loader: Optional[DatasetLoader]) -> bool:
        """Schedule signature scan for given index and base.

        Args:
            index: Image index in dataset
            base: Image base name
            loader: Dataset loader to use

        Returns:
            True if scheduled successfully
        """
        if not loader or not base or index < 0:
            return False
        self.loader = loader
        return self.schedule_indexed(index, base, force=False)

    # --------------------------------------------------------------
    # IndexedQueueController abstract method implementations
    # --------------------------------------------------------------

    def _create_task(self, indexed_item: Tuple[int, str]) -> SignatureTask:
        """Create signature task for indexed item."""
        index, base = indexed_item
        task = SignatureTask(self._epoch, index, base, self.loader)
        task.signals.completed.connect(self._handle_task_completed)
        task.signals.failed.connect(self._handle_task_failed)
        task.signals.finished.connect(self._handle_task_finished)
        return task

    def _can_schedule_item(self, indexed_item: Tuple[int, str], force: bool) -> bool:
        """Check if indexed item can be scheduled."""
        index, base = indexed_item
        if not base or not self.loader or index < 0:
            return False
        if not force and (indexed_item in self._running or self._is_queued(indexed_item)):
            return False
        return True

    def _cancel_task(self, task: SignatureTask) -> None:
        """Cancel a signature task."""
        task.cancel()

    # --------------------------------------------------------------
    # Signal handlers
    # --------------------------------------------------------------

    def _handle_task_completed(
        self,
        epoch: int,
        index: int,
        base: str,
        lwir_sig,
        vis_sig,
    ) -> None:
        if epoch != self._epoch:
            return
        self.signatureReady.emit(index, base, lwir_sig, vis_sig)

    def _handle_task_failed(self, epoch: int, base: str, message: str) -> None:
        if epoch != self._epoch:
            return
        self.signatureFailed.emit(base, message)

    def _handle_task_finished(self, task: SignatureTask) -> None:
        # Extract key from task to mark as finished
        key = (task.index, task.base)
        self._mark_task_finished(key)
