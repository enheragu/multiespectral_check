"""Manages cancellable task handlers and their state for UI components."""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Optional, Set, Union


class CancelController:
    def __init__(self) -> None:
        # Handler can return None or bool (for compat with functions that return success status)
        self._handlers: "OrderedDict[str, Callable[[], Union[None, bool]]]" = OrderedDict()
        self._inflight: Set[str] = set()

    def register(self, task_id: str, handler: Callable[[], Union[None, bool]]) -> None:
        self._handlers.pop(task_id, None)
        self._handlers[task_id] = handler
        self._inflight.discard(task_id)

    def unregister(self, task_id: str) -> None:
        self._handlers.pop(task_id, None)
        self._inflight.discard(task_id)

    def handler_for(self, task_id: str) -> Optional[Callable[[], Union[None, bool]]]:
        return self._handlers.get(task_id)

    def active_task(self) -> Optional[str]:
        if not self._handlers:
            return None
        for task_id in reversed(list(self._handlers.keys())):
            if task_id not in self._inflight:
                return task_id
        return None

    def mark_inflight(self, task_id: str) -> None:
        if task_id in self._handlers:
            self._inflight.add(task_id)

    def is_inflight(self, task_id: str) -> bool:
        return task_id in self._inflight

    def clear(self) -> None:
        self._handlers.clear()
        self._inflight.clear()
