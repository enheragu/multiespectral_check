"""Reusable logic for dataset filtering interactions in the image viewer."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

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


class FilterWorkflowMixin:
    """Encapsulates the filter menu plumbing used by the viewer."""

    # Attributes provided by host class (MainWindow)
    filter_mode: str
    session: Any
    state: Any
    ui: Any
    _filter_actions: Dict[str, Any]
    _filter_group: Optional[QActionGroup]
    _current_base: Optional[str]

    def _persist_preferences(self, **kwargs: Any) -> None:
        """Host must provide persistence method."""
        ...

    def load_current(self) -> None:
        """Host must provide image loading method."""
        ...

    def statusBar(self) -> Any:
        """Host must provide statusBar() method."""
        ...

    def _setup_filter_actions(self) -> None:
        self._filter_group = QActionGroup(self)  # type: ignore[arg-type]
        self._filter_group.setExclusive(True)
        for mode, action_name in FILTER_ACTION_NAMES.items():
            action = getattr(self.ui, action_name, None)
            if not action:
                continue
            self._filter_group.addAction(action)
            action.setData(mode)
            action.triggered.connect(lambda checked, m=mode: self._handle_filter_action_triggered(m, checked))
            self._filter_actions[mode] = action
        self._update_filter_checks()

    def _handle_filter_action_triggered(self, mode: str, checked: bool) -> None:
        if not checked:
            return
        self._apply_filter_mode(mode, user_triggered=True)

    def _update_filter_checks(self) -> None:
        for mode, action in self._filter_actions.items():
            action.blockSignals(True)
            action.setChecked(mode == self.filter_mode)
            action.blockSignals(False)

    def _apply_filter_mode(
        self,
        mode: str,
        *,
        persist: bool = True,
        refresh: bool = True,
        user_triggered: bool = False,
    ) -> None:
        if mode not in FILTER_ACTION_NAMES:
            mode = FILTER_ALL
        if self.filter_mode == mode:
            if refresh and self.session.has_images():
                self._reconcile_filter_state(show_warning=user_triggered)
            return
        self.filter_mode = mode
        if persist:
            self._persist_preferences(
                filter_mode=mode,
                filter_calibration_only=(mode == FILTER_CAL_ANY),
            )
        self._update_filter_checks()
        if refresh and self.session.has_images():
            self._reconcile_filter_state(show_warning=user_triggered)

    def _reconcile_filter_state(self, *, show_warning: bool) -> None:
        if not self.session.has_images():
            return
        if self.filter_mode == FILTER_ALL:
            self.load_current()
            return
        first_match = self._first_filtered_index()
        if first_match is None:
            if show_warning:
                label = FILTER_STATUS_TITLES.get(self.filter_mode, "Filter")
                status_bar = self.statusBar()  # type: ignore[misc]
                if status_bar:
                    status_bar.showMessage(f"{label} filter disabled: no matching images.", 5000)
            self._apply_filter_mode(FILTER_ALL, refresh=False)
            self.load_current()
            return
        current_base = self._current_base()
        if not current_base or not self._filter_accepts(current_base):
            self.current_index = first_match
        self.load_current()

    def _first_filtered_index(self) -> Optional[int]:
        if not self.session.loader:
            return None
        for idx, base in enumerate(self.session.loader.image_bases):
            if self._filter_accepts(base):
                return idx
        return None

    def _filter_accepts(self, base: Optional[str]) -> bool:
        if not base:
            return False
        if self.filter_mode == FILTER_ALL:
            return True
        if self.filter_mode == FILTER_DELETE:
            return base in self.state.cache_data["marks"]
        if self.filter_mode in (
            FILTER_DELETE_MANUAL,
            FILTER_DELETE_DUP,
            FILTER_DELETE_BLURRY,
            FILTER_DELETE_MOTION,
            FILTER_DELETE_SYNC,
            FILTER_DELETE_MISSING,
        ):
            reason = self.state.cache_data["marks"].get(base)
            if self.filter_mode == FILTER_DELETE_MANUAL:
                return bool(reason == REASON_USER)
            if self.filter_mode == FILTER_DELETE_DUP:
                return bool(reason == REASON_DUPLICATE)
            if self.filter_mode == FILTER_DELETE_BLURRY:
                return bool(reason == REASON_BLURRY)
            if self.filter_mode == FILTER_DELETE_MOTION:
                return bool(reason == REASON_MOTION)
            if self.filter_mode == FILTER_DELETE_SYNC:
                return bool(reason == REASON_SYNC)
            if self.filter_mode == FILTER_DELETE_MISSING:
                return bool(reason == REASON_MISSING_PAIR)
            return False
        if base not in self.state.calibration_marked:
            return False
        if self.filter_mode == FILTER_CAL_ANY:
            return True
        results = self.state.calibration_results.get(base, {})
        detected = sum(1 for channel in ("lwir", "visible") if results.get(channel) is True)
        if self.filter_mode == FILTER_CAL_BOTH:
            return bool(detected == 2)
        if self.filter_mode == FILTER_CAL_PARTIAL:
            return bool(detected == 1)
        if self.filter_mode == FILTER_CAL_MISSING:
            return bool(detected == 0)
        return True

    def _filtered_position(self) -> Tuple[int, int]:
        if self.filter_mode == FILTER_ALL or not self.session.loader:
            return 0, 0
        filtered_total = 0
        filtered_index = 0
        for idx, base in enumerate(self.session.loader.image_bases):
            if not self._filter_accepts(base):
                continue
            filtered_total += 1
            if idx == self.current_index:
                filtered_index = filtered_total
        return filtered_index, filtered_total
