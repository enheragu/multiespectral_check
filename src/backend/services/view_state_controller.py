"""View state controller managing display toggles and overlay invalidation.

Manages view state flags (grid, labels, rectified) and coordinates overlay
cache invalidation when these settings change.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from PyQt6.QtCore import QObject, pyqtSignal

if TYPE_CHECKING:
    from backend.services.dataset_session import DatasetSession


class ViewStateController(QObject):
    """Manages view state flags and coordinates overlay cache invalidation."""

    viewStateChanged = pyqtSignal()
    statusMessage = pyqtSignal(str, int)

    def __init__(
        self,
        session: "DatasetSession",
        parent: Optional[QObject] = None,
        # Callbacks
        invalidate_overlay_cache: Optional[Callable[[], None]] = None,
        load_current: Optional[Callable[[], None]] = None,
        has_calibration_data: Optional[Callable[[], bool]] = None,
        persist_preferences: Optional[Callable[..., None]] = None,
    ):
        """Initialize view state controller.

        Args:
            session: Dataset session
            parent: Optional parent QObject
            invalidate_overlay_cache: Callback to invalidate overlay cache
            load_current: Callback to reload current image
            has_calibration_data: Callback to check if calibration data exists
            persist_preferences: Callback to persist preference changes
        """
        super().__init__(parent)
        self.session = session

        self._invalidate_overlay_cache = invalidate_overlay_cache or (lambda: None)
        self._load_current = load_current or (lambda: None)
        self._has_calibration_data = has_calibration_data or (lambda: False)
        self._persist_preferences = persist_preferences or (lambda **kwargs: None)

        self._grid_mode = "off"  # "off", "thirds", "detailed"
        self._show_labels = False
        self._view_rectified = False

    @property
    def grid_mode(self) -> str:
        """Get grid display mode: 'off', 'thirds', or 'detailed'."""
        return self._grid_mode

    @grid_mode.setter
    def grid_mode(self, value: str) -> None:
        """Set grid display mode."""
        if value not in ("off", "thirds", "detailed"):
            value = "off"
        if self._grid_mode != value:
            self._grid_mode = value
            self._persist_preferences(grid_mode=value)
            self._invalidate_overlay_cache()
            if self.session.has_images():
                self._load_current()
            self.viewStateChanged.emit()

    @property
    def show_grid(self) -> bool:
        """Get grid display state (backwards compatibility)."""
        return self._grid_mode != "off"

    @show_grid.setter
    def show_grid(self, value: bool) -> None:
        """Set grid display state (backwards compatibility)."""
        self.grid_mode = "thirds" if value else "off"

    @property
    def show_labels(self) -> bool:
        """Get label display state."""
        return self._show_labels

    @show_labels.setter
    def show_labels(self, value: bool) -> None:
        """Set label display state."""
        if self._show_labels != value:
            self._show_labels = value
            self._persist_preferences(show_labels=value)
            self._invalidate_overlay_cache()
            if self.session.has_images():
                self._load_current()
            self.viewStateChanged.emit()

    @property
    def view_rectified(self) -> bool:
        """Get rectified view state."""
        return self._view_rectified

    @view_rectified.setter
    def view_rectified(self, value: bool) -> None:
        """Set rectified view state."""
        if self._view_rectified != value:
            self._view_rectified = value
            self._invalidate_overlay_cache()
            if self.session.has_images():
                self._load_current()
            msg = "Rectified view enabled" if value else "Rectified view disabled"
            self.statusMessage.emit(msg, 3000)
            self.viewStateChanged.emit()

    def toggle_grid(self, enabled: bool) -> None:
        """Toggle grid display.

        Args:
            enabled: True to show grid
        """
        self.show_grid = enabled

    def toggle_labels(self, enabled: bool) -> None:
        """Toggle label display.

        Args:
            enabled: True to show labels
        """
        self.show_labels = enabled

    def toggle_rectified(self, enabled: bool) -> bool:
        """Toggle rectified view with validation.

        Args:
            enabled: True to enable rectified view

        Returns:
            True if toggle was successful, False if validation failed
        """
        if enabled and not self._has_calibration_data():
            self.statusMessage.emit(
                "Import calibration data before enabling rectified rendering.",
                4000
            )
            return False
        self.view_rectified = enabled
        return True

    def load_preferences(
        self,
        show_grid: bool = False,
        show_labels: bool = False,
        grid_mode: str = "",
    ) -> None:
        """Load view preferences without triggering cache invalidation.

        Args:
            show_grid: Initial grid display state (legacy, overridden by grid_mode)
            show_labels: Initial label display state
            grid_mode: Grid display mode ("off", "thirds", "detailed")
        """
        # Use grid_mode if provided, otherwise fall back to legacy show_grid
        if grid_mode in ("off", "thirds", "detailed"):
            self._grid_mode = grid_mode
        else:
            self._grid_mode = "thirds" if show_grid else "off"
        self._show_labels = show_labels
