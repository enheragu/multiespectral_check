"""Runs blur/motion quality tasks in the thread pool and emits metrics per image."""
from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

from backend.dataset_loader import DatasetLoader
from backend.services.indexed_queue_controller import IndexedQueueController


@dataclass
class QualityMetrics:
    laplacian_var: Optional[float]
    anisotropy: Optional[float]


class QualityTaskSignals(QObject):
    completed = pyqtSignal(int, int, str, object, object)
    failed = pyqtSignal(int, str, str)
    finished = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()


class QualityTask(QRunnable):
    def __init__(self, epoch: int, index: int, base: str, loader: DatasetLoader) -> None:
        super().__init__()
        self.epoch = epoch
        self.index = index
        self.base = base
        self.loader = loader
        self.signals = QualityTaskSignals()
        self._cancel_event = Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def _ensure_not_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise RuntimeError("Quality sweep cancelled")

    def run(self) -> None:
        try:
            self._ensure_not_cancelled()
            lwir_metrics = self._compute_channel("lwir")
            self._ensure_not_cancelled()
            vis_metrics = self._compute_channel("visible")
            self._ensure_not_cancelled()
            self.signals.completed.emit(
                self.epoch,
                self.index,
                self.base,
                lwir_metrics,
                vis_metrics,
            )
        except Exception as exc:  # noqa: BLE001
            self.signals.failed.emit(self.epoch, self.base, str(exc))
        finally:
            self.signals.finished.emit(self)

    def _compute_channel(self, channel: str) -> QualityMetrics:
        path = self.loader.get_image_path(self.base, channel)
        if not path or not path.exists():
            return QualityMetrics(None, None)
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            return QualityMetrics(None, None)
        lap = cv2.Laplacian(img, cv2.CV_64F)
        lap_var = float(lap.var()) if lap.size else None
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        mean_gx = float(np.mean(np.abs(gx))) if gx.size else 0.0
        mean_gy = float(np.mean(np.abs(gy))) if gy.size else 0.0
        lo = min(mean_gx, mean_gy)
        hi = max(mean_gx, mean_gy)
        ratio = hi / (lo + 1e-6) if hi > 0.0 else None
        return QualityMetrics(lap_var, ratio)


class QualityController(IndexedQueueController[str, QualityTask]):
    """Controller for quality scan tasks with indexed queue management."""

    metricsReady = pyqtSignal(int, str, object, object)
    metricsFailed = pyqtSignal(str, str)

    def __init__(self, thread_pool: Optional[QThreadPool] = None) -> None:
        super().__init__(thread_pool=thread_pool, max_concurrent=4)
        self.loader: Optional[DatasetLoader] = None

    def set_loader(self, loader: Optional[DatasetLoader]) -> None:
        """Set the dataset loader for quality scans."""
        self.loader = loader

    def schedule(self, item: Tuple[int, str], *, force: bool = False, priority: str = "normal") -> bool:
        """Schedule quality scan for given index and base.

        Args:
            item: Tuple of (index, base)
            force: Force rescan even if already scanned
            priority: Priority level (not used for quality)

        Returns:
            bool: True if scheduled successfully
        """
        index, base = item
        if not self.loader or not base or index < 0:
            return False
        return self.schedule_indexed(index, base, force=force)

    # --------------------------------------------------------------
    # IndexedQueueController abstract method implementations
    # --------------------------------------------------------------

    def _create_task(self, indexed_item: Tuple[int, str]) -> QualityTask:
        """Create quality task for indexed item."""
        index, base = indexed_item
        task = QualityTask(self._epoch, index, base, self.loader)
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

    def _cancel_task(self, task: QualityTask) -> None:
        """Cancel a quality task."""
        task.cancel()

    # --------------------------------------------------------------
    # Signal handlers
    # --------------------------------------------------------------

    def _handle_task_completed(
        self,
        epoch: int,
        index: int,
        base: str,
        lwir_metrics: QualityMetrics,
        vis_metrics: QualityMetrics,
    ) -> None:
        if epoch != self._epoch:
            return
        self.metricsReady.emit(index, base, lwir_metrics, vis_metrics)

    def _handle_task_failed(self, epoch: int, base: str, message: str) -> None:
        if epoch != self._epoch:
            return
        self.metricsFailed.emit(base, message)

    def _handle_task_finished(self, task: QualityTask) -> None:
        # Extract key from task to mark as finished
        key = (task.index, task.base)
        self._mark_task_finished(key)
