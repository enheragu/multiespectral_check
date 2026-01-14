"""Global registry for DatasetHandlers to enable cross-component coordination."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

from common.log_utils import log_debug

if TYPE_CHECKING:
    from backend.services.dataset_handler import DatasetHandler


class HandlerRegistry:
    """
    Global singleton registry for DatasetHandlers.

    Allows any component (workspace_dialog, image_viewer, etc.) to:
    - Register handlers when creating them
    - Retrieve existing handlers by path
    - Notify handlers when dataset cache changes
    """

    def __init__(self) -> None:
        self._handlers: Dict[Path, 'DatasetHandler'] = {}

    def register(self, dataset_path: Path, handler: 'DatasetHandler') -> None:
        """Register a handler for a dataset path."""
        self._handlers[dataset_path] = handler
        log_debug(f"Registered handler for {dataset_path.name}", "REGISTRY")

    def get(self, dataset_path: Path) -> Optional['DatasetHandler']:
        """Get handler for a dataset path if it exists."""
        return self._handlers.get(dataset_path)

    def get_or_create(self, dataset_path: Path) -> 'DatasetHandler':
        """Get existing handler or create a new one."""
        if dataset_path in self._handlers:
            return self._handlers[dataset_path]

        # Import here to avoid circular dependency
        from backend.services.dataset_handler import DatasetHandler

        handler = DatasetHandler(dataset_path)
        self.register(dataset_path, handler)
        return handler

    def notify_cache_changed(self, dataset_path: Path) -> None:
        """
        Notify handler that image_labels.yaml was written.
        Handler should rebuild .summary_cache.yaml from it.
        """
        handler = self.get(dataset_path)
        if handler:
            log_debug(f"Notifying handler {dataset_path.name} of cache change", "REGISTRY")
            handler.mark_dirty()
        log_debug(f"No handler registered for {dataset_path.name}, skipping notification", "REGISTRY")

    def flush_all(self) -> None:
        """Force flush all registered handlers."""
        log_debug(f"Flushing {len(self._handlers)} handlers", "REGISTRY")
        for handler in self._handlers.values():
            handler.force_flush()

    def clear(self) -> None:
        """Clear all registered handlers."""
        self._handlers.clear()


# Global singleton instance
_registry = HandlerRegistry()


def get_handler_registry() -> HandlerRegistry:
    """Get the global handler registry singleton."""
    return _registry
