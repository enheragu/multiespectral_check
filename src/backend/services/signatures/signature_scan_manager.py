"""Queues and tracks duplicate signature scans across a dataset.

Coordinates controller scheduling, progress reporting, and cancellation while emitting signals
for ready signature pairs and handling failures gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Set

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from backend.services.scans.timed_scan_runner import TimedScanRunner

from frontend.services.ui.cancel_controller import CancelController
from backend.services.progress_tracker import ProgressTracker
from backend.services.signatures.signature_controller import SignatureController
from backend.services.dataset_session import DatasetSession

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False


@dataclass
class PrimeResult:
    status: str
    queued: int = 0
    total: int = 0


class SignatureScanManager(QObject):
    signatureReady = pyqtSignal(int, str, object, object)
    signatureFailed = pyqtSignal(str, str)
    sweepCompleted = pyqtSignal()  # Emitted when full sweep finishes

    def __init__(
        self,
        *,
        parent: Optional[QObject],
        session: DatasetSession,
        controller: SignatureController,
        progress_tracker: ProgressTracker,
        cancel_controller: CancelController,
        task_id: str,
        max_inflight: int,
        timer_interval_ms: int,
        cancel_state_callback: Optional[Callable[[], None]] = None,
        tqdm_position: Optional[int] = None,
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
        self._cancel_state_callback = cancel_state_callback
        self._tqdm_position = tqdm_position

        self._pending: Set[str] = set()
        self._completed: Set[str] = set()

        self._queue_runner = TimedScanRunner(
            parent=self,
            max_inflight=max_inflight,
            timer_interval_ms=timer_interval_ms,
            schedule_fn=self._schedule_index,
            on_progress=self._update_progress,
        )
        self._force = False
        self._scan_target = 0
        self._progress_started = False
        self._activity_total = 0
        self._activity_done = 0
        self._tqdm_bar = None

        self.controller.signatureReady.connect(self._handle_ready)
        self.controller.signatureFailed.connect(self._handle_failed)
        self.controller.activityChanged.connect(self._handle_activity_changed)

    # Public API -------------------------------------------------
    def reset_epoch(self) -> None:
        self.controller.reset()
        self._reset_tracking()

    def reset_progress(self) -> None:
        self._pending.clear()
        self._completed.clear()
        self._queue_runner.reset()
        self._force = False
        self._scan_target = 0
        self._progress_started = False
        self._activity_total = 0
        self._activity_done = 0
        if self._tqdm_bar is not None:
            self._tqdm_bar.close()
            self._tqdm_bar = None
        self.progress_tracker.finish(self.task_id)
        self.cancel_controller.unregister(self.task_id)
        if self._cancel_state_callback:
            self._cancel_state_callback()

    def cancel_all(self) -> None:
        self.controller.cancel_all()
        self.reset_progress()

    def prime(self, *, force: bool = False) -> PrimeResult:
        # Check if we have images (works for both datasets and collections)
        if not self.session.has_images():
            return PrimeResult(status="no-dataset")
        total_pairs = self.session.total_pairs()
        if total_pairs <= 0:
            return PrimeResult(status="no-images")
        if force:
            self.reset_epoch()
            self.reset_progress()
        if force:
            targets = list(range(total_pairs))
        else:
            # Get image bases from session (works for both datasets and collections)
            all_bases = self.session.get_all_bases()
            targets = [
                idx
                for idx, base in enumerate(all_bases)
                if base and not self._has_cached_signatures(base)
            ]
        self._scan_target = len(targets)
        self._progress_started = False
        if not targets:
            self._update_progress()
            return PrimeResult(status="cached", queued=0, total=total_pairs)
        self._queue_runner.start(targets)
        self._force = force
        self._update_progress()
        return PrimeResult(status="queued", queued=len(targets), total=total_pairs)

    def schedule_index(self, index: int, *, force: bool = False) -> Optional[str]:
        """Public entry to schedule a single index (used by viewer on demand)."""
        self._force = force
        return self._schedule_index(index)

    def _schedule_index(self, index: int) -> Optional[str]:
        base = self.session.get_base(index)
        if base is None:
            return None
        loader = self.session.get_loader_for_base(base)
        if not loader:
            return None
        if not self._force:
            if base in self._pending or base in self._completed:
                return None
            if self._has_cached_signatures(base):
                self._completed.add(base)
                return None
        if self.controller.schedule_for_scan(index, base, loader):
            self._pending.add(base)
            return base
        return None

    # Internal helpers ------------------------------------------
    def _finalize_job(self, base: Optional[str], *, success: bool) -> None:
        if not base:
            return
        # Inform runner to free pending slot and advance queue
        self._queue_runner.mark_completed(base)
        self._pending.discard(base)
        if success:
            self._completed.add(base)
        else:
            self._completed.discard(base)
        if self._activity_total > 0:
            self._activity_done = min(self._activity_total, self._activity_done + 1)
        self._update_progress()
        if self._queue_runner.queued_count:
            self._queue_runner._timer.start(self.timer_interval_ms)

    def _has_cached_signatures(self, base: Optional[str]) -> bool:
        if not base:
            return False
        bucket = self.state.signatures.get(base)
        if not bucket:
            return False
        return bucket.get("lwir") is not None and bucket.get("visible") is not None

    def _reset_tracking(self) -> None:
        self._pending.clear()
        self._completed.clear()

    def _update_progress(self) -> None:
        pending = len(self._pending) + self._queue_runner.queued_count
        total = self._scan_target or self._queue_runner.total
        if pending <= 0 or total <= 0:
            was_running = self._progress_started
            if self._progress_started:
                self.progress_tracker.finish(self.task_id)
                self.cancel_controller.unregister(self.task_id)
                if self._tqdm_bar is not None:
                    self._tqdm_bar.close()
                    self._tqdm_bar = None
            self._progress_started = False
            self._scan_target = 0
            self._activity_total = 0
            self._activity_done = 0
            # Emit signal when sweep completes successfully
            if was_running and pending == 0 and total > 0:
                self.sweepCompleted.emit()
            return
        done = max(0, min(total - pending, total))
        if not self._progress_started:
            # Create tqdm bar for dataset-level duplicate scan
            dataset_name = self.session.dataset_path.name if self.session.dataset_path else "dataset"
            if _TQDM_AVAILABLE:
                self._tqdm_bar = tqdm(
                    total=total,
                    desc=f"Duplicates {dataset_name}",
                    unit="img",
                    leave=False,
                    dynamic_ncols=True,
                    position=self._tqdm_position,
                )
            self.progress_tracker.start(
                self.task_id,
                "Scanning duplicates",
                total,
            )
            self.cancel_controller.register(
                self.task_id,
                self.cancel_all,
            )
            self._progress_started = True
        else:
            # Update tqdm bar
            if self._tqdm_bar is not None:
                increment = done - self._tqdm_bar.n
                if increment > 0:
                    self._tqdm_bar.update(increment)
            self.progress_tracker.update(
                self.task_id,
                done,
                total,
            )
        self._activity_total = total
        self._activity_done = done
        if self._cancel_state_callback:
            self._cancel_state_callback()

    # Signal handlers -------------------------------------------
    def _handle_ready(self, index: int, base: str, lwir_sig, vis_sig) -> None:
        self._finalize_job(base, success=True)
        self.signatureReady.emit(index, base, lwir_sig, vis_sig)

    def _handle_failed(self, base: str, message: str) -> None:
        self._finalize_job(base, success=False)
        self.signatureFailed.emit(base, message)

    def _handle_activity_changed(self, pending: int) -> None:  # noqa: ARG002
        self._update_progress()
