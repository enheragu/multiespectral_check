"""Centralized thread pool management to avoid resource saturation.

Provides unified worker limits calculation and coordinated thread pools across the application.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from PyQt6.QtCore import QThreadPool


@dataclass(frozen=True)
class WorkerLimits:
    """Calculated worker limits based on CPU count and application needs."""

    cpu_count: int
    calibration_detect: int
    workspace_scan: int
    workspace_sweep: int
    signature_scan_inflight: int
    quality_scan_inflight: int
    enhancement: int

    @classmethod
    def calculate(cls, cpu_override: Optional[int] = None) -> "WorkerLimits":
        """Calculate optimal worker limits based on available CPUs.

        Strategy:
        - Reserve 1-2 cores for UI thread and OS
        - Distribute remaining cores among heavyweight operations
        - Limit total concurrent workers to avoid thrashing

        Args:
            cpu_override: Override detected CPU count (for testing/tuning)

        Returns:
            WorkerLimits with calculated values
        """
        cpu = cpu_override or os.cpu_count() or 4

        # Reserve cores for UI and system
        available = max(2, cpu - 1)

        # Heavyweight operations (calibration, scanning)
        heavyweight_pool = max(2, min(6, available // 2))

        # Lightweight inflight limits (IO-bound)
        lightweight_limit = max(4, min(8, available))

        # Image enhancement (CPU-bound but low priority)
        enhancement_pool = max(1, min(2, available // 4))

        return cls(
            cpu_count=cpu,
            calibration_detect=heavyweight_pool,
            workspace_scan=heavyweight_pool,
            workspace_sweep=heavyweight_pool,
            signature_scan_inflight=lightweight_limit,
            quality_scan_inflight=lightweight_limit,
            enhancement=enhancement_pool,
        )

    def total_potential_workers(self) -> int:
        """Calculate maximum potential concurrent workers.

        This is a theoretical maximum if all systems are active simultaneously.
        """
        return (
            self.calibration_detect +
            self.workspace_scan +
            self.signature_scan_inflight +
            self.quality_scan_inflight +
            self.enhancement
        )


class ThreadPoolManager:
    """Manages application-wide thread pools with resource coordination.

    Provides:
    - Single global QThreadPool instance
    - Dedicated calibration pool with limits
    - Worker limits configuration
    - Resource usage visibility
    """

    def __init__(self, worker_limits: Optional[WorkerLimits] = None) -> None:
        """Initialize thread pool manager.

        Args:
            worker_limits: Pre-calculated limits, or None to auto-calculate
        """
        self.limits = worker_limits or WorkerLimits.calculate()

        # Global pool - used for most background work
        pool = QThreadPool.globalInstance()
        if pool is None:
            raise RuntimeError("QThreadPool.globalInstance() returned None")
        self._global_pool = pool
        # Don't exceed reasonable limits even on high-core machines
        self._global_pool.setMaxThreadCount(min(16, max(4, self.limits.cpu_count)))

        # Dedicated calibration pool to isolate heavyweight work
        self._calibration_pool: Optional[QThreadPool] = None

    def global_pool(self) -> QThreadPool:
        """Get the global thread pool for general background work."""
        return self._global_pool

    def calibration_pool(self, parent=None) -> QThreadPool:
        """Get or create dedicated calibration thread pool.

        Args:
            parent: Qt parent object for the pool

        Returns:
            QThreadPool configured for calibration work
        """
        if self._calibration_pool is None:
            pool = QThreadPool(parent)
            pool.setMaxThreadCount(self.limits.calibration_detect)
            self._calibration_pool = pool

        # Should never be None after creation above
        assert self._calibration_pool is not None, "Calibration pool creation failed"
        return self._calibration_pool

    def get_limits(self) -> WorkerLimits:
        """Get current worker limits configuration."""
        return self.limits

    def active_thread_count(self) -> int:
        """Get total active threads across all managed pools.

        Returns:
            Number of currently active threads
        """
        count = self._global_pool.activeThreadCount()
        pool = self._calibration_pool
        if pool is not None:
            count += pool.activeThreadCount()
        return count

    def max_thread_count(self) -> int:
        """Get maximum possible threads across all pools.

        Returns:
            Maximum thread count configured
        """
        count = self._global_pool.maxThreadCount()
        pool = self._calibration_pool
        if pool is not None:
            count += pool.maxThreadCount()
        return count


# Global singleton instance
_POOL_MANAGER: Optional[ThreadPoolManager] = None


def get_thread_pool_manager(reset: bool = False) -> ThreadPoolManager:
    """Get or create the global ThreadPoolManager singleton.

    Args:
        reset: If True, recreate the manager (for testing)

    Returns:
        ThreadPoolManager instance
    """
    global _POOL_MANAGER
    if _POOL_MANAGER is None or reset:
        _POOL_MANAGER = ThreadPoolManager()
    return _POOL_MANAGER
