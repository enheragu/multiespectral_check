"""Runs blur/motion sweeps across a dataset and auto-marks outliers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from backend.services.scans.timed_scan_runner import TimedScanRunner

from frontend.services.ui.cancel_controller import CancelController
from backend.services.progress_tracker import ProgressTracker
from backend.services.quality.quality_controller import QualityController, QualityMetrics
from backend.services.dataset_session import DatasetSession
from common.reasons import REASON_BLURRY, REASON_MOTION


@dataclass
class QualityRecord:
    base: str
    lwir: QualityMetrics
    visible: QualityMetrics


class QualityScanManager(QObject):
    finished = pyqtSignal(int, int)

    def __init__(
        self,
        *,
        parent: Optional[QObject],
        session: DatasetSession,
        controller: QualityController,
        progress_tracker: ProgressTracker,
        cancel_controller: CancelController,
        task_id: str,
        max_inflight: int,
        timer_interval_ms: int,
        on_cancel_state: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.state = session.state
        self.controller = controller
        self.progress_tracker = progress_tracker
        self.cancel_controller = cancel_controller
        self.task_id = task_id
        self.max_inflight = max_inflight
        self.timer_interval_ms = timer_interval_ms
        self._on_cancel_state = on_cancel_state

        self._queue_runner = TimedScanRunner(
            parent=self,
            max_inflight=max_inflight,
            timer_interval_ms=timer_interval_ms,
            schedule_fn=self._schedule_index,
            on_progress=self._update_progress,
        )
        self._pending = set()
        self._completed = set()
        self._records: List[QualityRecord] = []
        self._scan_target = 0
        self._progress_started = False

        self.controller.metricsReady.connect(self._handle_ready)
        self.controller.metricsFailed.connect(self._handle_failed)
        self.controller.activityChanged.connect(self._handle_activity_changed)

    # Public API -------------------------------------------------
    def reset(self) -> None:
        self.controller.reset()
        self._queue_runner.reset()
        self._pending.clear()
        self._completed.clear()
        self._records.clear()
        self._scan_target = 0
        self._progress_started = False
        self.progress_tracker.finish(self.task_id)
        self.cancel_controller.unregister(self.task_id)
        if self._on_cancel_state:
            self._on_cancel_state()

    def prime(self, *, force: bool = False) -> Tuple[str, int]:
        # Check if we have images (works for both datasets and collections)
        if not self.session.has_images():
            return ("no-dataset", 0)
        total_pairs = self.session.total_pairs()
        if total_pairs <= 0:
            return ("no-images", 0)
        self.reset()
        targets = list(range(total_pairs))
        self._queue_runner.start(targets)
        self._scan_target = len(targets)
        self._update_progress()
        return ("queued", len(targets))

    def cancel_all(self) -> None:
        self.controller.cancel_all()
        self.reset()

    # Internal helpers ------------------------------------------
    def _schedule_index(self, idx: int) -> Optional[str]:
        base = self.session.get_base(idx)
        if not base:
            return None
        loader = self.session.get_loader_for_base(base)
        if not loader:
            return None
        if base in self._pending or base in self._completed:
            return None
        # Set loader for this base's dataset, then schedule
        self.controller.set_loader(loader)
        if self.controller.schedule((idx, base)):
            self._pending.add(base)
            return base
        return None

    def _finalize_job(self, base: Optional[str], *, success: bool) -> None:
        if not base:
            return
        self._pending.discard(base)
        if success:
            self._completed.add(base)
        if not self._queue_runner.queued_count and not self._pending:
            self._finalize_marks()
        else:
            self._queue_runner._timer.start(self.timer_interval_ms)
        self._update_progress()

    def _handle_ready(self, index: int, base: str, lwir: QualityMetrics, vis: QualityMetrics) -> None:
        self._records.append(QualityRecord(base, lwir, vis))
        self._finalize_job(base, success=True)

    def _handle_failed(self, base: str, message: str) -> None:  # noqa: ARG002
        self._finalize_job(base, success=False)

    def _handle_activity_changed(self, pending: int) -> None:  # noqa: ARG002
        self._update_progress()

    def _update_progress(self) -> None:
        pending = len(self._pending) + self._queue_runner.queued_count
        total = self._scan_target or self._queue_runner.total
        done = max(0, min(total - pending, total))
        if pending <= 0 or total <= 0:
            if self._progress_started:
                self.progress_tracker.finish(self.task_id)
                self.cancel_controller.unregister(self.task_id)
            self._progress_started = False
            return
        if not self._progress_started:
            self.progress_tracker.start(self.task_id, "Scanning blur/motion", total)
            self.cancel_controller.register(self.task_id, self.cancel_all)
            self._progress_started = True
        else:
            self.progress_tracker.update(self.task_id, done, total)
        if self._on_cancel_state:
            self._on_cancel_state()

    # Marking logic ---------------------------------------------
    def _finalize_marks(self) -> None:
        if not self._records:
            self.finished.emit(0, 0)
            return
        lap_values = {"lwir": [], "visible": []}
        aniso_values = {"lwir": [], "visible": []}
        for rec in self._records:
            for channel, metrics in (("lwir", rec.lwir), ("visible", rec.visible)):
                if metrics and metrics.laplacian_var is not None:
                    lap_values[channel].append(metrics.laplacian_var)
                if metrics and metrics.anisotropy is not None:
                    aniso_values[channel].append(metrics.anisotropy)
        thresholds = {
            "lap": {
                ch: self._compute_blur_threshold(vals)
                for ch, vals in lap_values.items()
            },
            "aniso": {
                ch: self._compute_aniso_threshold(vals)
                for ch, vals in aniso_values.items()
            },
        }
        blurry_count = 0
        motion_count = 0
        for rec in self._records:
            reason = None
            for channel, metrics in (("visible", rec.visible), ("lwir", rec.lwir)):
                if not metrics or metrics.laplacian_var is None:
                    continue
                lap = metrics.laplacian_var
                aniso = metrics.anisotropy
                blur_thr = thresholds["lap"].get(channel)
                motion_thr = thresholds["aniso"].get(channel)
                is_blur = blur_thr is not None and lap <= blur_thr
                is_motion = motion_thr is not None and aniso is not None and aniso >= motion_thr
                if is_motion:
                    reason = REASON_MOTION
                    break
                if is_blur and reason is None:
                    reason = REASON_BLURRY
            if not reason:
                continue
            existing = self.state.cache_data["marks"].get(rec.base)
            if existing and existing != reason:
                continue
            if self.state.set_mark_reason(rec.base, reason, reason, auto=True):
                if reason == REASON_MOTION:
                    motion_count += 1
                else:
                    blurry_count += 1
        self.state.rebuild_reason_counts()
        self._records.clear()
        self.finished.emit(blurry_count, motion_count)

    """
        Automatic detection of blurred images
    """
    @staticmethod
    def _compute_blur_threshold(values: List[float]) -> Optional[float]:
        if not values:
            return None
        arr = np.array(values, dtype=float)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        sigma = 1.4826 * mad if mad > 0 else 0.0
        # Loosen blur cutoff: median minus 1.5 sigma, but never above 70% of median.
        thr = med - 1.5 * sigma
        thr = min(thr, med * 0.7)
        return max(thr, 0.0)

    """
        Automatic detection if image is moved or not based on anisotropy values.
    """
    @staticmethod
    def _compute_aniso_threshold(values: List[float]) -> Optional[float]:
        if not values:
            return None
        arr = np.array(values, dtype=float)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        sigma = 1.4826 * mad if mad > 0 else 0.0
        # Slightly more permissive motion cutoff.
        thr = med + 1.5 * sigma
        return max(thr, 1.3)
