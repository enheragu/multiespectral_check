"""Centralized coordinator for workspace, collection, and dataset cache management.

Ensures consistent cache invalidation and persistence across the entire workspace hierarchy.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Set
from threading import Lock

from backend.services.workspace_inspector import invalidate_workspace_cache
from common.log_utils import log_debug


class WorkspaceCacheCoordinator:
    """Manages cache invalidation for workspace, collections, and datasets.

    Ensures that when a dataset changes:
    1. Its own cache is marked for save
    2. Parent collection cache is invalidated
    3. Workspace cache is invalidated
    4. Prevents cascading invalidations during batch operations
    """

    def __init__(self):
        self._lock = Lock()
        self._dirty_datasets: Set[Path] = set()
        self._invalidated_collections: Set[Path] = set()
        self._workspace_path: Optional[Path] = None
        self._workspace_dirty = False

    def set_workspace(self, workspace_path: Path) -> None:
        """Set the current workspace path."""
        with self._lock:
            self._workspace_path = workspace_path

    def mark_dataset_dirty(self, dataset_path: Path) -> None:
        """Mark a dataset as needing cache save.

        This will also invalidate parent collection and workspace caches.
        """
        debug = os.environ.get("DEBUG_CACHE", "").lower() in {"1", "true", "on"}

        with self._lock:
            self._dirty_datasets.add(dataset_path)

            # Invalidate collection caches (aggregation) when a dataset changes.
            # Layouts supported:
            # - workspace/dataset
            # - workspace/collection/dataset
            # - workspace/collection (collection itself may also have a cache)
            if self._workspace_path:
                # Dataset inside a collection
                if dataset_path.parent != self._workspace_path and dataset_path.parent.parent == self._workspace_path:
                    collection_path = dataset_path.parent
                    if collection_path not in self._invalidated_collections:
                        self._invalidated_collections.add(collection_path)
                        collection_cache = collection_path / ".cache.yaml"
                        if collection_cache.exists():
                            if debug:
                                log_debug(f"Invalidating collection cache: {collection_path.name}", "CACHE_COORD")
                            try:
                                collection_cache.unlink()
                            except OSError:
                                pass

                # Collection path itself (direct child of workspace)
                if dataset_path.parent == self._workspace_path:
                    collection_path = dataset_path
                    if collection_path not in self._invalidated_collections:
                        collection_cache = collection_path / ".cache.yaml"
                        if collection_cache.exists():
                            self._invalidated_collections.add(collection_path)
                            if debug:
                                log_debug(f"Invalidating collection cache: {collection_path.name}", "CACHE_COORD")
                            try:
                                collection_cache.unlink()
                            except OSError:
                                pass

            # Mark workspace as dirty
            if not self._workspace_dirty:
                self._workspace_dirty = True
                if debug:
                    log_debug(f"Marking workspace as dirty: {self._workspace_path}", "CACHE_COORD")

    def flush_workspace(self) -> None:
        """Flush workspace cache invalidation.

        Should be called after dataset caches are saved.
        """
        debug = os.environ.get("DEBUG_CACHE", "").lower() in {"1", "true", "on"}

        with self._lock:
            if self._workspace_dirty and self._workspace_path:
                if debug:
                    log_debug(f"Flushing workspace cache: {self._workspace_path.name}", "CACHE_COORD")
                invalidate_workspace_cache(self._workspace_path)
                self._workspace_dirty = False
                self._dirty_datasets.clear()
                self._invalidated_collections.clear()

    def is_dirty(self) -> bool:
        """Check if there are pending cache changes."""
        with self._lock:
            return bool(self._dirty_datasets) or self._workspace_dirty

    def reset(self) -> None:
        """Reset all dirty flags (used when closing/switching workspaces)."""
        with self._lock:
            self._dirty_datasets.clear()
            self._invalidated_collections.clear()
            self._workspace_dirty = False


# Global singleton instance
_cache_coordinator = WorkspaceCacheCoordinator()


def get_cache_coordinator() -> WorkspaceCacheCoordinator:
    """Get the global cache coordinator instance."""
    return _cache_coordinator
