"""Unified progress reporting for GUI and terminal.

Provides a single interface for progress reporting that:
- Emits to GUI via ProgressTracker (when available)
- Emits to terminal via tqdm (when available)
- Supports cancellation checking
- Supports nested progress (workspace → collection → dataset)

Usage:
    # In a sweep method:
    def run_duplicate_sweep(self, reporter: Optional[ProgressReporter] = None):
        for idx, base in enumerate(self.loader.image_bases):
            if reporter and reporter.is_cancelled():
                break
            # ... do work ...
            if reporter:
                reporter.advance()
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tqdm import tqdm
    from backend.services.progress_tracker import ProgressTracker


class ProgressReporter:
    """Unified progress reporter for both GUI and terminal output.

    Design Philosophy:
    - Single interface, multiple outputs (GUI + terminal)
    - No code duplication in sweep methods
    - Supports cancellation without passing callbacks everywhere
    - Minimal API: start, advance, finish, is_cancelled
    """

    def __init__(
        self,
        *,
        tracker: Optional["ProgressTracker"] = None,
        task_id: str = "task",
        use_tqdm: bool = True,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Initialize progress reporter.

        Args:
            tracker: Optional GUI ProgressTracker for Qt progress bar
            task_id: Identifier for this task in the tracker
            use_tqdm: Whether to show terminal progress bar
            cancel_check: Optional callable that returns True if cancelled
        """
        self._tracker = tracker
        self._task_id = task_id
        self._use_tqdm = use_tqdm
        self._cancel_check = cancel_check

        self._tqdm: Optional["tqdm"] = None
        self._total = 0
        self._current = 0
        self._label = ""

    def start(self, total: int, label: str = "") -> None:
        """Start progress tracking with known total."""
        self._total = max(1, total)
        self._current = 0
        self._label = label

        # Start GUI tracker
        if self._tracker:
            self._tracker.start(self._task_id, label, self._total)

        # Start tqdm
        if self._use_tqdm:
            try:
                from tqdm import tqdm
                self._tqdm = tqdm(
                    total=self._total,
                    desc=label,
                    unit="img",
                    leave=False,
                    ncols=100,
                )
            except ImportError:
                self._tqdm = None

    def advance(self, n: int = 1, suffix: str = "") -> None:
        """Advance progress by n steps."""
        self._current = min(self._current + n, self._total)

        # Update GUI tracker
        if self._tracker:
            self._tracker.update(self._task_id, self._current)

        # Update tqdm
        if self._tqdm:
            self._tqdm.update(n)
            if suffix:
                self._tqdm.set_postfix_str(suffix)

    def set_description(self, desc: str) -> None:
        """Update the description/label."""
        self._label = desc
        if self._tqdm:
            self._tqdm.set_description(desc)

    def is_cancelled(self) -> bool:
        """Check if operation should be cancelled."""
        if self._cancel_check:
            return self._cancel_check()
        return False

    def finish(self) -> None:
        """Finish progress tracking."""
        # Close tqdm
        if self._tqdm:
            self._tqdm.close()
            self._tqdm = None

        # Finish GUI tracker
        if self._tracker:
            self._tracker.finish(self._task_id)

        self._current = 0
        self._total = 0

    def __enter__(self) -> "ProgressReporter":
        return self

    def __exit__(self, *args) -> None:
        self.finish()


@contextmanager
def progress_context(
    total: int,
    label: str = "",
    *,
    tracker: Optional["ProgressTracker"] = None,
    task_id: str = "task",
    use_tqdm: bool = True,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Iterator[ProgressReporter]:
    """Context manager for progress reporting.

    Usage:
        with progress_context(100, "Processing", tracker=self.tracker) as progress:
            for item in items:
                if progress.is_cancelled():
                    break
                process(item)
                progress.advance()
    """
    reporter = ProgressReporter(
        tracker=tracker,
        task_id=task_id,
        use_tqdm=use_tqdm,
        cancel_check=cancel_check,
    )
    reporter.start(total, label)
    try:
        yield reporter
    finally:
        reporter.finish()


class NullReporter(ProgressReporter):
    """No-op reporter for when progress reporting is not needed.

    Useful for testing or when called without a reporter.
    """

    def __init__(self, cancel_check: Optional[Callable[[], bool]] = None) -> None:
        super().__init__(use_tqdm=False, cancel_check=cancel_check)

    def start(self, total: int, label: str = "") -> None:
        pass

    def advance(self, n: int = 1, suffix: str = "") -> None:
        pass

    def set_description(self, desc: str) -> None:
        pass

    def finish(self) -> None:
        pass

