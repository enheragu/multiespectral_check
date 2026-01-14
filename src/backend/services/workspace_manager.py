"""WorkspaceManager: Orchestrates DatasetHandlers without direct file reading."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Callable, Set

from backend.services.dataset_handler import DatasetHandler
from backend.services.workspace_inspector import WorkspaceDatasetInfo
from common.timing import timed
from common.log_utils import log_debug, log_info, log_warning, is_debug_enabled


class WorkspaceManager:
    """
    Manages workspace-level operations by coordinating DatasetHandlers.

    Features:
    - Creates and manages handlers for each dataset
    - Aggregates stats on demand from handlers
    - No direct file reading (delegates to handlers)
    - Invalidates workspace cache on changes
    """

    def __init__(self, workspace_path: Path, on_dataset_changed: Optional[Callable[[Path], None]] = None) -> None:
        self.workspace_path = workspace_path
        self.handlers: Dict[Path, DatasetHandler] = {}
        self._on_dataset_changed = on_dataset_changed
        log_info(f"Initialized for {workspace_path}", "[WorkspaceManager]")

    @timed
    def scan_workspace(self) -> List[WorkspaceDatasetInfo]:
        """
        Scan workspace and create handlers for each dataset.
        Returns list of WorkspaceDatasetInfo with stats from summary caches.

        WARNING: This creates QObject handlers, must be called from main thread!
        """
        log_debug("===== scan_workspace START =====", "WORKSPACE_MGR")
        log_debug(f"Workspace path: {self.workspace_path}", "WORKSPACE_MGR")

        # Discover all dataset directories
        discovered_paths = self._discover_dataset_paths()

        log_debug(f"Discovered {len(discovered_paths)} paths:", "WORKSPACE_MGR")
        for p in discovered_paths:
            rel_path = p.relative_to(self.workspace_path) if p != self.workspace_path else '<workspace_root>'
            is_collection = self._is_collection(p)
            is_leaf = self._is_leaf_dataset(p)
            log_debug(f"  - {rel_path} (collection={is_collection}, leaf={is_leaf})", "WORKSPACE_MGR")

        # Create handlers for new datasets (QObjects - must be in main thread!)
        for dataset_path in discovered_paths:
            if dataset_path not in self.handlers:
                handler = DatasetHandler(dataset_path, workspace_manager=self)
                self.handlers[dataset_path] = handler

                # Register in global registry
                from backend.services.handler_registry import get_handler_registry
                get_handler_registry().register(dataset_path, handler)

                log_info(f"Created handler for {dataset_path.name}", "[WorkspaceManager]")

        # Remove handlers for deleted datasets
        current_paths = set(discovered_paths)
        removed_paths = [p for p in self.handlers.keys() if p not in current_paths]
        for path in removed_paths:
            log_debug(f"Removed handler for {path.name}", "WORKSPACE_MGR")
            del self.handlers[path]

        # Aggregate stats from all handlers
        infos: List[WorkspaceDatasetInfo] = []
        for handler in self.handlers.values():
            log_debug(f"\nLoading summary for {handler.dataset_path.name}...", "WORKSPACE_MGR")
            summary = handler.load_summary()
            is_collection = self._is_collection(handler.dataset_path)
            parent = self._get_parent_name(handler.dataset_path)
            log_debug(f"  {handler.dataset_path.name}: is_collection={is_collection}, parent={parent}", "WORKSPACE_MGR")
            info = WorkspaceDatasetInfo(
                name=handler.dataset_path.name,
                path=handler.dataset_path,
                stats=summary.to_stats(),
                note=summary.dataset_info.get("note", ""),
                is_collection=is_collection,
                parent=parent,
            )
            infos.append(info)

            log_debug(f"✓ {info.name}: pairs={info.stats.total_pairs}, "
                      f"calib={info.stats.calibration_marked}/{info.stats.calibration_both}, "
                      f"manual={info.stats.tagged_manual}, auto={info.stats.tagged_auto}, "
                      f"removed={info.stats.removed_total}, is_collection={info.is_collection}")

        # Aggregate children stats into collections
        self._aggregate_collection_stats(infos)

        log_debug(f"\nFinal result: {len(infos)} infos", "WORKSPACE_MGR")
        for info in infos:
            log_debug(f"  - {info.name}: is_collection={info.is_collection}, parent={info.parent}, pairs={info.stats.total_pairs}", "WORKSPACE_MGR")
        log_debug("===== scan_workspace END =====\n", "WORKSPACE_MGR")

        return infos

    def _aggregate_collection_stats(self, infos: List[WorkspaceDatasetInfo]) -> None:
        """Aggregate stats from children into their parent collections."""
        from backend.services.stats_manager import DatasetStats

        # Group by parent
        children_by_parent: Dict[str, List[WorkspaceDatasetInfo]] = {}
        for info in infos:
            if info.parent:
                children_by_parent.setdefault(info.parent, []).append(info)

        # Update collection stats by aggregating from children
        for info in infos:
            if info.is_collection:
                children = children_by_parent.get(info.name, [])
                if children:
                    # Merge children stats - collections should ONLY show aggregated data
                    merged_stats = DatasetStats()
                    for child in children:
                        merged_stats.merge(child.stats)

                    # REPLACE collection stats entirely with merged stats from children
                    info.stats = merged_stats

                    log_debug(f"Updated collection {info.name}: "
                              f"pairs={info.stats.total_pairs} (aggregated from {len(children)} children)")
                else:
                    # Collection with no children - might be empty, keep its own stats
                    if info.stats.total_pairs > 0:
                        log_warning(f"Collection {info.name} has no children but shows {info.stats.total_pairs} pairs", "WORKSPACE_MGR")

    def _discover_dataset_paths(self) -> List[Path]:
        """
        Discover all dataset directories in workspace by structure.
        - Leaf dataset: has 'visible' and 'lwir' subdirectories
        - Collection: has subdirectories that are datasets

        CRITICAL: Workspace root itself is NEVER included - only its children.
        """
        discovered: List[Path] = []

        # First pass: find all leaf datasets (directories with visible+lwir)
        for entry in self.workspace_path.iterdir():
            if not entry.is_dir() or self._is_noise_dir(entry):
                continue

            # Check if it's a leaf dataset
            if self._is_leaf_dataset(entry):
                discovered.append(entry)
            else:
                # Check children
                for child in entry.iterdir():
                    if child.is_dir() and not self._is_noise_dir(child) and self._is_leaf_dataset(child):
                        discovered.append(child)

        # Second pass: add collections (directories with dataset children)
        collections_to_add: Set[Path] = set()
        for path in discovered:
            parent = path.parent
            if parent != self.workspace_path:
                # Parent is a collection
                collections_to_add.add(parent)

        discovered.extend(sorted(collections_to_add))

        return discovered

    def _is_leaf_dataset(self, path: Path) -> bool:
        """Check if directory is a leaf dataset (has visible and lwir subdirs)."""
        return (path / "visible").is_dir() and (path / "lwir").is_dir()

    def _is_noise_dir(self, path: Path) -> bool:
        """Return True if path should be skipped when scanning workspace."""
        return path.name in {"to_delete", ".git", "__pycache__"}

    def _is_collection(self, dataset_path: Path) -> bool:
        """Check if dataset is a collection (has subdirectories that are datasets)."""
        # A collection has at least one child that is a leaf dataset
        for child in dataset_path.iterdir():
            if child.is_dir() and not self._is_noise_dir(child) and self._is_leaf_dataset(child):
                return True
        return False

    def _get_parent_name(self, dataset_path: Path) -> Optional[str]:
        """Get parent collection name if dataset is nested."""
        parent_dir = dataset_path.parent

        # If parent is workspace root, no parent collection
        if parent_dir == self.workspace_path:
            return None

        # Check if parent is within workspace and is a collection
        if not parent_dir.is_relative_to(self.workspace_path):
            return None

        # Parent must be a collection (has dataset children)
        if self._is_collection(parent_dir):
            return parent_dir.name

        return None

    def notify_dataset_changed(self, dataset_path: Path) -> None:
        """
        Called by handlers when dataset data changes.
        Invalidates workspace-level caches and calls callback.
        Thread-safe: Can be called from any thread.
        """
        from PyQt6.QtCore import QTimer

        if is_debug_enabled("workspace"):
            log_debug(f"Dataset changed: {dataset_path.name}", "WorkspaceManager")

        # Call callback for UI refresh if provided
        callback = self._on_dataset_changed
        if callback is not None:
            if is_debug_enabled("workspace"):
                log_debug(f"Scheduling table refresh for {dataset_path.name}", "WorkspaceManager")

            # Use QTimer.singleShot(0, ...) to invoke callback on main thread
            # This is thread-safe and works from any thread
            try:
                QTimer.singleShot(0, lambda cb=callback, p=dataset_path: cb(p))
            except RuntimeError as e:
                if is_debug_enabled("workspace"):
                    log_debug(f"Error scheduling callback: {e}", "WorkspaceManager")
        else:
            if is_debug_enabled("workspace"):
                log_debug("No on_dataset_changed callback registered", "WorkspaceManager")

        # TODO: Optionally invalidate .workspace_index.json
        # For now, we can skip workspace index entirely and aggregate on-demand

    def flush_all(self) -> None:
        """Force flush all handlers (called on GUI close)."""
        log_debug(f"Flushing all {len(self.handlers)} handlers", "WORKSPACE_MGR")

        for handler in self.handlers.values():
            handler.force_flush()

        log_debug("Flush complete", "WORKSPACE_MGR")

    def get_handler(self, dataset_path: Path) -> Optional[DatasetHandler]:
        """Get handler for specific dataset path."""
        return self.handlers.get(dataset_path)

    def create_handler(self, dataset_path: Path) -> DatasetHandler:
        """Create and register a new handler for dataset."""
        if dataset_path in self.handlers:
            return self.handlers[dataset_path]

        handler = DatasetHandler(dataset_path, workspace_manager=self)
        self.handlers[dataset_path] = handler

        log_debug(f"Created handler for {dataset_path.name}", "WORKSPACE_MGR")

        return handler

    def regenerate_all_summaries(self) -> int:
        """Force rebuild all summary caches from filesystem and full caches."""
        log_info(f"Regenerating all summary caches for {len(self.handlers)} handlers", "WORKSPACE_MGR")

        count = 0
        for handler in self.handlers.values():
            # Force rebuild
            handler.load_summary(force_rebuild=True)
            handler.force_flush()
            count += 1

        log_info(f"Regenerated {count} summary caches", "WORKSPACE_MGR")

        return count
