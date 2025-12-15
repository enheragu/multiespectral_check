from __future__ import annotations

from threading import Event
from typing import Optional, Set

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

from dataset_loader import DatasetLoader
from utils.duplicates import compute_signature_from_path


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


class SignatureController(QObject):
    signatureReady = pyqtSignal(int, str, object, object)
    signatureFailed = pyqtSignal(str, str)
    activityChanged = pyqtSignal(int)

    def __init__(self, thread_pool: Optional[QThreadPool] = None) -> None:
        super().__init__()
        self.thread_pool = thread_pool or QThreadPool.globalInstance()
        self._epoch = 0
        self._tasks: Set[SignatureTask] = set()

    def reset(self) -> None:
        self.cancel_all()

    def schedule(self, index: int, base: Optional[str], loader: Optional[DatasetLoader]) -> bool:
        if not loader or not base or index < 0:
            return False
        task = SignatureTask(self._epoch, index, base, loader)
        task.signals.completed.connect(self._handle_task_completed)
        task.signals.failed.connect(self._handle_task_failed)
        task.signals.finished.connect(self._handle_task_finished)
        self._tasks.add(task)
        self.thread_pool.start(task)
        self._emit_activity()
        return True

    def cancel_all(self) -> None:
        self._epoch += 1
        if not self._tasks:
            self._emit_activity()
            return
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()
        self._emit_activity()

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
        self._tasks.discard(task)
        self._emit_activity()

    def _emit_activity(self) -> None:
        self.activityChanged.emit(len(self._tasks))
