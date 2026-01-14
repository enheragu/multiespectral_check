"""Navigation controller for browsing dataset images with filtering support."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from PyQt6.QtCore import QObject, pyqtSignal

from common.log_utils import log_debug

if TYPE_CHECKING:
    from backend.services.dataset_session import DatasetSession


class NavigationController(QObject):
    """Manages image navigation (prev/next/jump) with filter-aware movement."""

    indexChanged = pyqtSignal(int)  # Emitted when current index changes
    navigationBlocked = pyqtSignal(str)  # Emitted when navigation is blocked with reason message

    def __init__(
        self,
        session: DatasetSession,
        get_filter_mode: Callable[[], int],
        filter_accepts: Callable[[Optional[str]], bool],
        parent: Optional[QObject] = None,
    ) -> None:
        """Initialize navigation controller.

        Args:
            session: Dataset session providing image list and filtering
            get_filter_mode: Callable that returns current filter mode
            filter_accepts: Callable that checks if a base passes current filter
            parent: Optional parent QObject
        """
        super().__init__(parent)
        self.session = session
        self._get_filter_mode = get_filter_mode
        self._filter_accepts = filter_accepts
        self._current_index = 0

    @property
    def current_index(self) -> int:
        """Get current image index."""
        return self._current_index

    @current_index.setter
    def current_index(self, value: int) -> None:
        """Set current image index and emit signal if changed."""
        if value != self._current_index:
            log_debug(f"current_index changed: {self._current_index} -> {value}, emitting indexChanged", "NAV")
            self._current_index = value
            self.indexChanged.emit(value)

    def prev(self, message_label: str = "filtered images") -> bool:
        """Navigate to previous image respecting current filter.

        Args:
            message_label: Label to use in status message if navigation blocked

        Returns:
            True if navigation succeeded, False if blocked
        """
        if not self.session.has_images():
            return False
        if not self._navigate(-1):
            filter_mode = self._get_filter_mode()
            # Import here to avoid circular dependency
            from backend.services.filter_modes import FILTER_ALL
            if filter_mode != FILTER_ALL:
                self.navigationBlocked.emit(f"No previous {message_label}.")
            return False
        return True

    def next(self, message_label: str = "filtered images") -> bool:
        """Navigate to next image respecting current filter.

        Args:
            message_label: Label to use in status message if navigation blocked

        Returns:
            True if navigation succeeded, False if blocked
        """
        if not self.session.has_images():
            return False
        if not self._navigate(1):
            filter_mode = self._get_filter_mode()
            # Import here to avoid circular dependency
            from backend.services.filter_modes import FILTER_ALL
            if filter_mode != FILTER_ALL:
                self.navigationBlocked.emit(f"No more {message_label}.")
            return False
        return True

    def jump_to(self, target_index: int) -> bool:
        """Jump directly to specific index.

        Args:
            target_index: Index to jump to

        Returns:
            True if jump succeeded, False if index invalid
        """
        if not self.session.has_images():
            return False
        count = self.session.total_pairs()
        if 0 <= target_index < count:
            self.current_index = target_index
            return True
        return False

    def reset(self) -> None:
        """Reset navigation to first image."""
        self.current_index = 0

    def _navigate(self, direction: int) -> bool:
        """Internal navigation with filter support.

        Args:
            direction: 1 for next, -1 for previous

        Returns:
            True if navigation succeeded
        """
        if not self.session.has_images():
            return False

        # Use total_pairs() to support both datasets and collections
        total = self.session.total_pairs()
        if total <= 0:
            return False

        filter_mode = self._get_filter_mode()
        from backend.services.filter_modes import FILTER_ALL

        # No filter - simple circular navigation
        if filter_mode == FILTER_ALL:
            self.current_index = (self.current_index + direction) % total
            return True

        # Filter active - search for next matching image
        # Use internal _current_index to avoid emitting signals during search
        start = self._current_index
        candidate = start
        for _ in range(total):
            candidate = (candidate + direction) % total
            base = self.session.get_base(candidate)
            if base and self._filter_accepts(base):
                self.current_index = candidate  # Only emit once we found the target
                return True

        # No match found - index stays unchanged
        return False

    def filtered_position(self) -> tuple[int, int]:
        """Get current position in filtered list.

        Returns:
            Tuple of (current_index_in_filtered, total_filtered_count)
        """
        filter_mode = self._get_filter_mode()
        from backend.services.filter_modes import FILTER_ALL

        total = self.session.total_pairs()
        if filter_mode == FILTER_ALL or total == 0:
            return 0, 0

        filtered_total = 0
        filtered_index = 0
        for idx in range(total):
            base = self.session.get_base(idx)
            if not base or not self._filter_accepts(base):
                continue
            filtered_total += 1
            if idx == self.current_index:
                filtered_index = filtered_total

        return filtered_index, filtered_total
