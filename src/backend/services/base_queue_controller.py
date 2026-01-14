"""Abstract base controller for managing background task queues with epoch-based cancellation.

Provides unified queue management logic to eliminate code duplication across
calibration, signature, and quality controllers.
"""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections import deque
from typing import Deque, Dict, Generic, Hashable, Optional, TypeVar

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal
from PyQt6.sip import wrappertype


# Type variables for generic task management
TItem = TypeVar("TItem", bound=Hashable)  # Queue item type (str, int, etc.)
TTask = TypeVar("TTask", bound=QRunnable)  # Task runnable type


class QABCMeta(wrappertype, ABCMeta):
    """Combined metaclass for QObject + ABC + Generic."""
    pass


class BaseQueueController(QObject, Generic[TItem, TTask], metaclass=QABCMeta):
    """Abstract base for controllers that manage a queue of background tasks.

    Features:
    - Queue management with priority support
    - Epoch-based cancellation (increment epoch to invalidate old tasks)
    - Concurrent task limiting
    - Activity tracking via signals

    Subclasses must implement:
    - _create_task(item) -> TTask
    - _can_schedule_item(item, force) -> bool
    - _get_item_key(item) -> Hashable (for deduplication)
    """

    # Signal emitted when number of pending tasks changes
    activityChanged = pyqtSignal(int)

    def __init__(
        self,
        thread_pool: Optional[QThreadPool] = None,
        max_concurrent: int = 4,
    ) -> None:
        """Initialize queue controller.

        Args:
            thread_pool: Thread pool for executing tasks, or None for global
            max_concurrent: Maximum number of tasks running simultaneously
        """
        super().__init__()
        # Always set to valid pool (use global if None provided)
        pool = thread_pool if thread_pool is not None else QThreadPool.globalInstance()
        if pool is None:
            raise RuntimeError("Failed to get QThreadPool instance")
        self.thread_pool: QThreadPool = pool
        self.max_concurrent = max(1, max_concurrent)
        self._epoch = 0
        self._queue: Deque[TItem] = deque()
        self._running: Dict[Hashable, TTask] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset controller, cancelling all pending work."""
        self.cancel_all()

    def schedule(
        self,
        item: TItem,
        *,
        force: bool = False,
        priority: str = "normal",
    ) -> bool:
        """Schedule an item for background processing.

        Args:
            item: Item to process
            force: If True, reschedule even if already queued/running
            priority: "high" (front of queue) or "normal" (back of queue)

        Returns:
            True if item was scheduled, False if rejected
        """
        if not self._can_schedule_item(item, force):
            return False

        key = self._get_item_key(item)

        if force:
            self._remove_from_queue(key)

        if priority == "high":
            self._queue.appendleft(item)
        else:
            self._queue.append(item)

        self._maybe_dispatch()
        self._emit_activity()
        return True

    def cancel_all(self) -> None:
        """Cancel all queued and running tasks."""
        if not (self._queue or self._running):
            self._epoch += 1
            self._emit_activity()
            return

        self._epoch += 1

        # Cancel running tasks
        for task in self._running.values():
            self._cancel_task(task)

        self._queue.clear()
        self._running.clear()
        self._emit_activity()

    def pending_count(self) -> int:
        """Get total number of pending tasks (queued + running).

        Returns:
            Number of tasks in queue plus currently running
        """
        return len(self._queue) + len(self._running)

    def is_idle(self) -> bool:
        """Check if controller has no pending work.

        Returns:
            True if queue is empty and no tasks running
        """
        return not self._queue and not self._running

    # ------------------------------------------------------------------
    # Abstract methods - must be implemented by subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _create_task(self, item: TItem) -> TTask:
        """Create a runnable task for processing the item.

        The task should be connected to appropriate signal handlers.

        Args:
            item: Item to process

        Returns:
            QRunnable task ready to execute
        """
        pass

    @abstractmethod
    def _can_schedule_item(self, item: TItem, force: bool) -> bool:
        """Check if item can be scheduled.

        Subclass should validate the item and check if scheduling makes sense.

        Args:
            item: Item to check
            force: Whether this is a forced reschedule

        Returns:
            True if item can be scheduled
        """

    @abstractmethod
    def _get_item_key(self, item: TItem) -> Hashable:
        """Get unique key for deduplicating items.

        Args:
            item: Item to get key for

        Returns:
            Hashable key (often the item itself if it's already hashable)
        """

    @abstractmethod
    def _cancel_task(self, task: TTask) -> None:
        """Cancel a running task.

        Args:
            task: Task to cancel
        """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_queued(self, key: Hashable) -> bool:
        """Check if item with given key is in queue.

        Args:
            key: Item key to check

        Returns:
            True if item is queued
        """
        return any(self._get_item_key(item) == key for item in self._queue)

    def _remove_from_queue(self, key: Hashable) -> None:
        """Remove item with given key from queue.

        Args:
            key: Key of item to remove
        """
        if not self._queue:
            return
        self._queue = deque(
            item for item in self._queue
            if self._get_item_key(item) != key
        )

    def _maybe_dispatch(self) -> None:
        """Dispatch queued tasks if there's capacity."""
        while len(self._running) < self.max_concurrent and self._queue:
            item = self._queue.popleft()
            key = self._get_item_key(item)

            # Skip if already running (shouldn't happen, but defensive)
            if key in self._running:
                continue

            task = self._create_task(item)
            self._running[key] = task
            self.thread_pool.start(task)

    def _mark_task_finished(self, key: Hashable) -> None:
        """Mark task as finished and dispatch next queued task.

        Args:
            key: Key of finished task
        """
        self._running.pop(key, None)
        self._maybe_dispatch()
        self._emit_activity()

    def _emit_activity(self) -> None:
        """Emit activity changed signal with pending count."""
        self.activityChanged.emit(self.pending_count())
