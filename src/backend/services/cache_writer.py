"""Background cache persistence helpers for dataset state and overlay cache."""

from __future__ import annotations

import threading

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

from backend.services.cache_service import CachePersistPayload, save_dataset_cache_file
from backend.utils.cache import save_cache
from common.log_utils import log_debug, log_info, log_warning, log_error


_CACHE_WRITE_LOCK = threading.Lock()


def write_cache_payload(payload: CachePersistPayload) -> None:
    """Write cache payloads to disk."""
    import os
    debug = os.environ.get("DEBUG_CACHE", "").lower() in {"1", "true", "on"}
    # Serialise cache writes to avoid overlapping writes when running sweeps in parallel.
    with _CACHE_WRITE_LOCK:
        save_cache(payload.cache_data)
        if payload.dataset_cache_path:
            # Regular dataset - save dataset cache file
            # NOTE: Collections don't save their own cache - DatasetSession.snapshot_cache_payload()
            # already distributed marks to children via Collection.distribute_to_children()
            if debug:
                dataset_path = payload.dataset_cache_path.parent
                log_debug(f"Writing dataset cache: {dataset_path.name}", "CACHE_WRITE")
            save_dataset_cache_file(payload.dataset_cache_path, payload.dataset_entry)

            # Notify handler registry that cache was written
            from backend.services.handler_registry import get_handler_registry
            dataset_path = payload.dataset_cache_path.parent
            get_handler_registry().notify_cache_changed(dataset_path)


class CacheFlushNotifier(QObject):
    finished = pyqtSignal()


class CacheFlushRunnable(QRunnable):
    def __init__(self, payload: CachePersistPayload, notifier: CacheFlushNotifier) -> None:
        super().__init__()
        self._payload = payload
        self._notifier = notifier

    def run(self) -> None:
        write_cache_payload(self._payload)
        try:
            self._notifier.finished.emit()
        except RuntimeError:
            # Notifier may be deleted if the UI closed while the flush was running.
            pass
