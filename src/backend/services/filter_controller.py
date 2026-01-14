"""Filter controller for managing dataset filtering logic.

Encapsulates filter mode management and filter UI state synchronization
previously handled by FilterWorkflowMixin.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QActionGroup

from backend.services.filter_modes import (
    FILTER_ACTION_NAMES,
    FILTER_ALL,
    FILTER_CAL_ANY,
    FILTER_CAL_BOTH,
    FILTER_CAL_MISSING,
    FILTER_CAL_PARTIAL,
    FILTER_DELETE,
    FILTER_DELETE_BLURRY,
    FILTER_DELETE_DUP,
    FILTER_DELETE_MANUAL,
    FILTER_DELETE_MISSING,
    FILTER_DELETE_MOTION,
    FILTER_DELETE_SYNC,
    FILTER_STATUS_TITLES,
)
from common.reasons import (
    REASON_BLURRY,
    REASON_DUPLICATE,
    REASON_MISSING_PAIR,
    REASON_MOTION,
    REASON_SYNC,
    REASON_USER,
)

if TYPE_CHECKING:
    from backend.services.dataset_session import DatasetSession
    from backend.services.viewer_state import ViewerState


class FilterController(QObject):
    """Manages dataset filtering and filter UI state."""

    filterModeChanged = pyqtSignal(str)  # Emitted when filter mode changes

    def __init__(
        self,
        session: "DatasetSession",
        state: "ViewerState",
        ui: Any,
        on_filter_changed: Callable[[], None],
        parent: Optional[QObject] = None,
    ) -> None:
        """Initialize filter controller.

        Args:
            session: Dataset session
            state: Viewer state
            ui: UI object with filter actions
            on_filter_changed: Callback when filter changes (reload current image)
            parent: Optional parent QObject
        """
        super().__init__(parent)
        self.session = session
        self.state = state
        self.ui = ui
        self._on_filter_changed = on_filter_changed
        self._filter_mode = FILTER_ALL
        self._filter_actions: Dict[str, Any] = {}
        self._filter_group: Optional[QActionGroup] = None

    @property
    def filter_mode(self) -> str:
        """Get current filter mode."""
        return self._filter_mode

    @filter_mode.setter
    def filter_mode(self, mode: str) -> None:
        """Set filter mode (controller owns this state)."""
        if mode != self._filter_mode:
            self._filter_mode = mode
            self.filterModeChanged.emit(mode)

    def setup_filter_actions(self) -> None:
        """Setup filter menu actions with exclusive group."""
        self._filter_group = QActionGroup(self)
        self._filter_group.setExclusive(True)

        for mode, action_name in FILTER_ACTION_NAMES.items():
            action = getattr(self.ui, action_name, None)
            if not action:
                continue

            action.setCheckable(True)
            action.setActionGroup(self._filter_group)
            action.triggered.connect(lambda checked, m=mode: self._handle_filter_change(m))
            self._filter_actions[mode] = action

            if mode == self.filter_mode:
                action.setChecked(True)

    def _handle_filter_change(self, mode: str) -> None:
        """Handle filter mode change from UI action.

        Args:
            mode: New filter mode
        """
        if mode == self.filter_mode:
            return

        self.filter_mode = mode
        self._on_filter_changed()
        self.update_filter_checks()

    def update_filter_checks(self) -> None:
        """Update filter action check states."""
        for mode, action in self._filter_actions.items():
            if action:
                action.blockSignals(True)
                action.setChecked(mode == self.filter_mode)
                action.blockSignals(False)

    def reconcile_filter_state(self, show_warning: bool = False) -> None:
        """Reconcile filter state when marks change.

        Ensures filter mode is still valid after marking/unmarking images.

        Args:
            show_warning: Whether to show warning dialog if filter has no matches
        """
        if self.filter_mode == FILTER_ALL:
            return

        # Check if current filter has any matches
        if not self.session.loader:
            return

        has_matches = any(self.filter_accepts_base(base) for base in self.session.loader.image_bases)
        if not has_matches:
            # No matches - switch to FILTER_ALL
            self.filter_mode = FILTER_ALL
            self.update_filter_checks()

            if show_warning:
                from PyQt6.QtWidgets import QMessageBox
                title = FILTER_STATUS_TITLES.get(self.filter_mode, "Filter")
                QMessageBox.information(
                    self.parent(),
                    title,
                    f"No images match filter '{title}'. Switching to 'All images'.",
                )

    def filter_accepts_base(self, base: str) -> bool:
        """Check if base passes current filter.

        Args:
            base: Image base name

        Returns:
            True if base passes filter
        """
        mode = self.filter_mode

        if mode == FILTER_ALL:
            return True
        elif mode == FILTER_CAL_MISSING:
            # Missing: not marked OR marked but no detections
            if base not in self.state.calibration_marked:
                return False
            results = self.state.calibration_results.get(base, {})
            detected = sum(1 for ch in ("lwir", "visible") if results.get(ch) is True)
            return bool(detected == 0)
        elif mode == FILTER_CAL_PARTIAL:
            if base not in self.state.calibration_marked:
                return False
            results = self.state.calibration_results.get(base, {})
            detected = sum(1 for ch in ("lwir", "visible") if results.get(ch) is True)
            return bool(detected == 1)
        elif mode == FILTER_CAL_ANY:
            return base in self.state.calibration_marked
        elif mode == FILTER_CAL_BOTH:
            if base not in self.state.calibration_marked:
                return False
            results = self.state.calibration_results.get(base, {})
            detected = sum(1 for ch in ("lwir", "visible") if results.get(ch) is True)
            return bool(detected == 2)
        elif mode == FILTER_DELETE:
            return base in self.state.cache_data["marks"]
        elif mode == FILTER_DELETE_MANUAL:
            reason = self.state.cache_data["marks"].get(base)
            return bool(reason == REASON_USER)
        elif mode == FILTER_DELETE_BLURRY:
            reason = self.state.cache_data["marks"].get(base)
            return bool(reason == REASON_BLURRY)
        elif mode == FILTER_DELETE_MOTION:
            reason = self.state.cache_data["marks"].get(base)
            return bool(reason == REASON_MOTION)
        elif mode == FILTER_DELETE_SYNC:
            reason = self.state.cache_data["marks"].get(base)
            return bool(reason == REASON_SYNC)
        elif mode == FILTER_DELETE_DUP:
            reason = self.state.cache_data["marks"].get(base)
            return bool(reason == REASON_DUPLICATE)
        elif mode == FILTER_DELETE_MISSING:
            reason = self.state.cache_data["marks"].get(base)
            return bool(reason == REASON_MISSING_PAIR)
        else:
            return True
