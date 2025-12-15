from __future__ import annotations

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

from services.cache_service import CachePersistPayload, save_dataset_cache_file
from utils.cache import save_cache


def write_cache_payload(payload: CachePersistPayload) -> None:
    """Write cache payloads to disk."""
    save_cache(payload.cache_data)
    if payload.dataset_cache_path:
        save_dataset_cache_file(payload.dataset_cache_path, payload.dataset_entry)


class CacheFlushNotifier(QObject):
    finished = pyqtSignal()


class CacheFlushRunnable(QRunnable):
    def __init__(self, payload: CachePersistPayload, notifier: CacheFlushNotifier) -> None:
        super().__init__()
        self._payload = payload
        self._notifier = notifier

    def run(self) -> None:
        write_cache_payload(self._payload)
        self._notifier.finished.emit()
