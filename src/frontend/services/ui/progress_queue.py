"""Queue-style progress accounting with cancel hooks.

Maintains per-task progress state, registers cancel handlers, and updates the shared tracker so UI
components can stay synchronized with long-running background work.

Also manages terminal TQDM bars for operations that benefit from terminal feedback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, TYPE_CHECKING

from frontend.services.ui.cancel_controller import CancelController
from backend.services.progress_tracker import ProgressTracker

if TYPE_CHECKING:
    from tqdm import tqdm

# Check if tqdm is available
try:
    from tqdm import tqdm as tqdm_class
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm_class = None  # type: ignore


@dataclass
class ProgressState:
    total: int = 0
    done: int = 0
    last_pending: int = 0
    tqdm_bar: Optional["tqdm"] = field(default=None, repr=False)


class ProgressQueueManager:
    def __init__(
        self,
        tracker: ProgressTracker,
        cancel_controller: CancelController,
        *,
        use_tqdm: bool = True,
    ) -> None:
        self.tracker = tracker
        self.cancel_controller = cancel_controller
        self._states: Dict[str, ProgressState] = {}
        self._use_tqdm = use_tqdm and TQDM_AVAILABLE

    def reset(self) -> None:
        # Close any open tqdm bars
        for state in self._states.values():
            if state.tqdm_bar:
                state.tqdm_bar.close()
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
            # Task finished - close tqdm bar and cleanup
            if state.tqdm_bar:
                state.tqdm_bar.close()
            self._states.pop(task_id, None)
            self.cancel_controller.unregister(task_id)
            self.tracker.finish(task_id)
            return

        if last <= 0 or total <= 0:
            # New task starting
            state = ProgressState(total=pending, done=0, last_pending=pending)
            self._states[task_id] = state
            self.tracker.start(task_id, label, pending)
            if cancel_handler:
                self.cancel_controller.register(task_id, cancel_handler)
            # Create tqdm bar for terminal output
            if self._use_tqdm and TQDM_AVAILABLE and tqdm_class is not None:
                state.tqdm_bar = tqdm_class(
                    total=pending,
                    desc=label,
                    unit="img",
                    leave=False,
                    ncols=100,
                )
            return

        # Update progress
        remaining = max(0, total - done)
        if pending > remaining:
            total = done + pending
        old_done = done
        done = max(0, min(total - pending, total))
        state.total = total
        state.done = done
        state.last_pending = pending
        self.tracker.update(task_id, done, total)

        # Update tqdm bar
        if state.tqdm_bar:
            advance = done - old_done
            if advance > 0:
                state.tqdm_bar.update(advance)
            # Only refresh tqdm when total changes significantly (>5%) to reduce log spam
            # The main progress update is via update(advance), not refresh()
            old_total = state.tqdm_bar.total
            if old_total != total:
                state.tqdm_bar.total = total
                # Only refresh if there was actual progress or significant change
                if advance > 0 or (total - old_total) > max(1, old_total // 20):
                    state.tqdm_bar.refresh()

    def finish(self, task_id: str) -> None:
        state = self._states.pop(task_id, None)
        if state and state.tqdm_bar:
            state.tqdm_bar.close()
        self.cancel_controller.unregister(task_id)
        self.tracker.finish(task_id)
