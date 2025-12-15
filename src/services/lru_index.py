"""Small LRU helper to track usage ordering with a bounded size."""
from __future__ import annotations

from collections import deque
from typing import Deque, Hashable, Iterable, Optional


class LRUIndex:
    """Tracks keys in LRU order and evicts when over capacity."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, capacity)
        self._order: Deque[Hashable] = deque()

    def touch(self, key: Hashable) -> list[Hashable]:
        """Mark key as most recently used and return any evicted keys."""
        try:
            self._order.remove(key)
        except ValueError:
            pass
        self._order.append(key)
        return self._enforce()

    def remove(self, key: Optional[Hashable]) -> None:
        if key is None:
            return
        try:
            self._order.remove(key)
        except ValueError:
            pass

    def pop_oldest(self) -> Optional[Hashable]:
        if not self._order:
            return None
        return self._order.popleft()

    def _enforce(self) -> list[Hashable]:
        evicted: list[Hashable] = []
        while len(self._order) > self.capacity:
            evicted.append(self._order.popleft())
        return evicted

    def __iter__(self) -> Iterable[Hashable]:
        return iter(self._order)

    def clear(self) -> None:
        self._order.clear()
