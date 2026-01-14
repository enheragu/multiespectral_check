"""Thread-safe cache flush coordinator to prevent race conditions.

Provides mutex-protected access to pending cache writes and flush state.
"""
from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from backend.services.cache_service import CachePersistPayload


class CacheFlushCoordinator:
    """Coordinates cache flush operations with thread-safety.

    Prevents race conditions when multiple threads trigger cache writes simultaneously.
    Uses a mutex to protect shared state.
    """

    def __init__(self) -> None:
        """Initialize coordinator with empty state."""
        self._lock = Lock()
        self._inflight: bool = False
        self._pending: Optional["CachePersistPayload"] = None

    def request_flush(self, payload: "CachePersistPayload") -> bool:
        """Request a cache flush, queuing if one is already running.

        Args:
            payload: Cache data to flush

        Returns:
            True if flush was started immediately, False if queued as pending
        """
        with self._lock:
            if self._inflight:
                # Flush in progress, queue this as pending (replaces any existing pending)
                self._pending = payload
                return False
            else:
                # No flush running, mark as inflight
                self._inflight = True
                return True

    def mark_completed(self) -> Optional["CachePersistPayload"]:
        """Mark current flush as completed and get next pending flush if any.

        Returns:
            Pending payload to flush next, or None if no work pending
        """
        with self._lock:
            pending = self._pending
            self._pending = None

            if pending is None:
                # No more work, clear inflight flag
                self._inflight = False
            # else: keep inflight=True since we're returning work to do

            return pending

    def has_pending_work(self) -> bool:
        """Check if there's work in progress or pending.

        Returns:
            True if flush is running or queued
        """
        with self._lock:
            return self._inflight or self._pending is not None

    def is_idle(self) -> bool:
        """Check if coordinator is completely idle.

        Returns:
            True if no flush running and no pending work
        """
        with self._lock:
            return not self._inflight and self._pending is None

    def cancel_pending(self) -> None:
        """Cancel any pending flush without affecting current inflight flush."""
        with self._lock:
            self._pending = None

    def reset(self) -> None:
        """Reset coordinator state (for shutdown or testing).

        Warning: Only call this when you're sure no flush is actually running.
        """
        with self._lock:
            self._inflight = False
            self._pending = None
