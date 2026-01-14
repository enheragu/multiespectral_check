"""Queue controller for tasks indexed by (index, item) tuples.

Extends BaseQueueController to handle tasks where both an index and an item
are needed as the key, common for scan operations that track position.
"""
from __future__ import annotations

from typing import Generic, Hashable, Optional, Tuple, TypeVar

from PyQt6.QtCore import QThreadPool

from backend.services.base_queue_controller import BaseQueueController, TTask

# Type variable for the item type (e.g., str for base name)
TItem = TypeVar("TItem", bound=Hashable)


class IndexedQueueController(BaseQueueController[Tuple[int, TItem], TTask], Generic[TItem, TTask]):
    """Queue controller for tasks keyed by (index, item) tuples.

    This controller extends BaseQueueController to handle operations where both
    an index (position) and an item (e.g., base name) are needed. Common for
    scan operations that need to track both what to process and its position.

    Type Parameters:
        TItem: Type of the item (second element of tuple), must be hashable
        TTask: Type of the task runnable

    Example:
        class MyController(IndexedQueueController[str, MyTask]):
            def _create_task(self, indexed_item: Tuple[int, str]) -> MyTask:
                index, base = indexed_item
                return MyTask(self._epoch, index, base, self.loader)
    """

    def __init__(
        self,
        thread_pool: Optional[QThreadPool] = None,
        max_concurrent: int = 4,
    ) -> None:
        """Initialize indexed queue controller.

        Args:
            thread_pool: Thread pool for executing tasks, or None for global
            max_concurrent: Maximum number of concurrent tasks
        """
        super().__init__(thread_pool=thread_pool, max_concurrent=max_concurrent)

    def schedule_indexed(self, index: int, item: TItem, force: bool = False) -> bool:
        """Schedule a task with explicit index and item.

        Args:
            index: Position/index for the item
            item: Item to process
            force: If True, reschedule even if already queued

        Returns:
            True if scheduled successfully
        """
        return self.schedule((index, item), force=force)

    def _get_item_key(self, indexed_item: Tuple[int, TItem]) -> Hashable:
        """Get hashable key for indexed item.

        The tuple (index, item) is already hashable if item is hashable.

        Args:
            indexed_item: Tuple of (index, item)

        Returns:
            The tuple itself as the key
        """
        return indexed_item

    def _extract_index_item(self, indexed_item: Tuple[int, TItem]) -> Tuple[int, TItem]:
        """Extract index and item from indexed item.

        Helper method for subclasses to easily unpack the tuple.

        Args:
            indexed_item: Tuple of (index, item)

        Returns:
            Unpacked (index, item) tuple
        """
        return indexed_item
