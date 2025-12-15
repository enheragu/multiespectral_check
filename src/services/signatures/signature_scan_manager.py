"""Queues and tracks duplicate signature scans across a dataset.

Coordinates controller scheduling, progress reporting, and cancellation while emitting signals
for ready signature pairs and handling failures gracefully.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional, Set

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from services.ui.cancel_controller import CancelController
from services.progress_tracker import ProgressTracker
from services.signatures.signature_controller import SignatureController
from services.dataset_session import DatasetSession


@dataclass
class PrimeResult:
    status: str
    queued: int = 0
    total: int = 0


class SignatureScanManager(QObject):
    signatureReady = pyqtSignal(int, str, object, object)
    signatureFailed = pyqtSignal(str, str)

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

        self._pending: Set[str] = set()
        self._completed: Set[str] = set()
        self._queue: Deque[int] = deque()
        self._force = False
        self._scan_target = 0
        self._progress_started = False
        self._activity_total = 0
        self._activity_done = 0

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._drain_queue)

        self.controller.signatureReady.connect(self._handle_ready)
        self.controller.signatureFailed.connect(self._handle_failed)
        self.controller.activityChanged.connect(self._handle_activity_changed)

    # Public API -------------------------------------------------
    def reset_epoch(self) -> None:
        self.controller.reset()
        self._reset_tracking()

    def reset_progress(self) -> None:
        self._queue.clear()
        self._pending.clear()
        self._completed.clear()
        self._force = False
        self._scan_target = 0
        self._progress_started = False
        self._activity_total = 0
        self._activity_done = 0
        if self._timer.isActive():
            self._timer.stop()
        self.progress_tracker.finish(self.task_id)
        self.cancel_controller.unregister(self.task_id)
        if self._cancel_state_callback:
            self._cancel_state_callback()

    def cancel_all(self) -> None:
        self.controller.cancel_all()
        self.reset_progress()

    def prime(self, *, force: bool = False) -> PrimeResult:
        loader = self.session.loader
        if not loader:
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
            targets = [
                idx
                for idx, base in enumerate(loader.image_bases)
                if base and not self._has_cached_signatures(base)
            ]
        self._scan_target = len(targets)
        self._progress_started = False
        if not targets:
            self._update_progress()
            return PrimeResult(status="cached", queued=0, total=total_pairs)
        self._queue.extend(targets)
        self._force = force
        self._update_progress()
        self._timer.start(0)
        return PrimeResult(status="queued", queued=len(targets), total=total_pairs)

    def schedule_index(self, index: int, *, force: bool = False) -> bool:
        loader = self.session.loader
        base = self.session.get_base(index)
        if not loader or base is None:
            return False
        if not force:
            if base in self._pending or base in self._completed:
                return False
            if self._has_cached_signatures(base):
                self._completed.add(base)
                return False
        if self.controller.schedule(index, base, loader):
            self._pending.add(base)
            return True
        return False

    # Internal helpers ------------------------------------------
    def _drain_queue(self) -> None:
        if not self._queue:
            self._update_progress()
            return
        inflight = len(self._pending)
        available = max(0, self.max_inflight - inflight)
        scheduled = 0
        while available > 0 and self._queue:
            index = self._queue.popleft()
            if self.schedule_index(index, force=self._force):
                available -= 1
                scheduled += 1
        self._update_progress()
        if self._queue and scheduled == 0:
            self._timer.start(self.timer_interval_ms)

    def _finalize_job(self, base: Optional[str], *, success: bool) -> None:
        if not base:
            return
        self._pending.discard(base)
        if success:
            self._completed.add(base)
        else:
            self._completed.discard(base)
        if self._activity_total > 0:
            self._activity_done = min(self._activity_total, self._activity_done + 1)
        self._update_progress()
        if self._queue:
            self._timer.start(self.timer_interval_ms)

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
        pending = len(self._pending) + len(self._queue)
        total = self._scan_target
        if pending <= 0 or total <= 0:
            if self._progress_started:
                self.progress_tracker.finish(self.task_id)
                self.cancel_controller.unregister(self.task_id)
            self._progress_started = False
            self._scan_target = 0
            self._activity_total = 0
            self._activity_done = 0
            return
        done = max(0, min(total - pending, total))
        if not self._progress_started:
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
