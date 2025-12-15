"""Queue-style progress accounting with cancel hooks.

Maintains per-task progress state, registers cancel handlers, and updates the shared tracker so UI
components can stay synchronized with long-running background work.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from services.ui.cancel_controller import CancelController
from services.progress_tracker import ProgressTracker


@dataclass
class ProgressState:
    total: int = 0
    done: int = 0
    last_pending: int = 0


class ProgressQueueManager:
    def __init__(self, tracker: ProgressTracker, cancel_controller: CancelController) -> None:
        self.tracker = tracker
        self.cancel_controller = cancel_controller
        self._states: Dict[str, ProgressState] = {}

    def reset(self) -> None:
        self._states.clear()

    def update(
        self,
        *,
        pending: int,
        label: str,
        task_id: str,
        cancel_handler: Optional[Callable[[], None]] = None,
    ) -> None:
        state = self._states.get(task_id, ProgressState())
        total = state.total
        done = state.done
        last = state.last_pending

        if pending <= 0:
            self._states.pop(task_id, None)
            self.cancel_controller.unregister(task_id)
            self.tracker.finish(task_id)
            return

        if last <= 0 or total <= 0:
            state = ProgressState(total=pending, done=0, last_pending=pending)
            self._states[task_id] = state
            self.tracker.start(task_id, label, pending)
            if cancel_handler:
                self.cancel_controller.register(task_id, cancel_handler)
            return

        remaining = max(0, total - done)
        if pending > remaining:
            total = done + pending
        done = max(0, min(total - pending, total))
        state.total = total
        state.done = done
        state.last_pending = pending
        self.tracker.update(task_id, done, total)

    def finish(self, task_id: str) -> None:
        self._states.pop(task_id, None)
        self.cancel_controller.unregister(task_id)
        self.tracker.finish(task_id)
