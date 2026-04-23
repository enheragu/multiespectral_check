"""Controls marking interactions and context menus for dataset items."""

from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import QApplication, QMenu

from common.reasons import REASON_CHOICES, REASON_SHORTCUTS, REASON_USER
from backend.services.dataset_actions import DatasetActions


class MarkingController:
    """Encapsulates mark toggling and image context-menu behavior."""

    def __init__(
        self,
        *,
        parent,
        state,
        dataset_actions: DatasetActions,
        get_current_base: Callable[[], Optional[str]],
        has_images: Callable[[], bool],
        prev_image: Callable[[], None],
        next_image: Callable[[], None],
        invalidate_overlay_cache: Callable[[Optional[str]], None],
        load_image_pair: Callable[[str], None],
        update_stats: Callable[[], None],
        update_delete_button: Callable[[], None],
        mark_cache_dirty: Callable[[], None],
        status_message: Callable[[str, int], None],
        schedule_calibration_job: Callable[[str, bool], bool],
        reconcile_filter_state: Callable[[], None],
        calibration_shortcut: str,
        get_image_path: Optional[Callable[[str, str], Optional[str]]] = None,
        refresh_workspace: Optional[Callable[[], None]] = None,
        enter_labelling_mode: Optional[Callable[[], None]] = None,
        enter_auto_labelling_mode: Optional[Callable[[], None]] = None,
    ) -> None:
        self.parent = parent
        self.state = state
        self.dataset_actions = dataset_actions
        self.get_current_base = get_current_base
        self.has_images = has_images
        self.prev_image = prev_image
        self.next_image = next_image
        self.invalidate_overlay_cache = invalidate_overlay_cache
        self.load_image_pair = load_image_pair
        self.update_stats = update_stats
        self.update_delete_button = update_delete_button
        self.mark_cache_dirty = mark_cache_dirty
        self.status_message = status_message
        self.schedule_calibration_job = schedule_calibration_job
        self.reconcile_filter_state = reconcile_filter_state
        self.calibration_shortcut = calibration_shortcut
        self.get_image_path = get_image_path
        self.refresh_workspace = refresh_workspace
        self._enter_labelling_mode = enter_labelling_mode
        self._enter_auto_labelling_mode = enter_auto_labelling_mode

    def enter_labelling_mode(self) -> None:
        """Enter manual labelling mode via the configured callback."""
        if self._enter_labelling_mode:
            self._enter_labelling_mode()

    def enter_auto_labelling_mode(self) -> None:
        """Enter auto labelling mode via the configured callback."""
        if self._enter_auto_labelling_mode:
            self._enter_auto_labelling_mode()

    # Context menu -------------------------------------------------
    def show_context_menu(self, global_pos) -> None:
        if not self.has_images():
            return
        base = self.get_current_base()
        if not base:
            return
        menu = QMenu(self.parent)
        nav_prev = menu.addAction("Previous image")
        if nav_prev is not None:
            nav_prev.setShortcut(QKeySequence("Left Arrow"))
            nav_prev.setShortcutVisibleInContextMenu(True)
        nav_next = menu.addAction("Next image")
        if nav_next is not None:
            nav_next.setShortcut(QKeySequence("Right Arrow"))
            nav_next.setShortcutVisibleInContextMenu(True)
        menu.addSeparator()
        current_reason = self.state.cache_data["marks"].get(base)
        reason_actions = []
        for reason, label in REASON_CHOICES:
            action = menu.addAction(label)
            if action is not None:
                action.setCheckable(True)
                action.setChecked(current_reason == reason)
                shortcut = REASON_SHORTCUTS.get(reason)
                if shortcut:
                    action.setShortcut(QKeySequence(shortcut))
                    action.setShortcutVisibleInContextMenu(True)
            reason_actions.append((action, reason))
        menu.addSeparator()
        calibration_action = menu.addAction("Mark as calibration candidate")
        if calibration_action is not None:
            calibration_action.setShortcut(QKeySequence(self.calibration_shortcut))
            calibration_action.setShortcutVisibleInContextMenu(True)
            calibration_action.setCheckable(True)
            calibration_action.setChecked(base in self.state.calibration_marked)
        reanalyze_action = None
        if base in self.state.calibration_marked:
            reanalyze_action = menu.addAction("Re-run calibration detection")
            if reanalyze_action is not None:
                reanalyze_action.setShortcut(QKeySequence("Ctrl+Shift+R"))
                reanalyze_action.setShortcutVisibleInContextMenu(True)
        menu.addSeparator()
        labelling_action = menu.addAction("Enter manual labelling mode")
        if labelling_action is not None:
            labelling_action.setShortcut(QKeySequence("Ctrl+L"))
            labelling_action.setShortcutVisibleInContextMenu(True)
        auto_labelling_action = menu.addAction("Enter auto labelling mode")
        if auto_labelling_action is not None:
            auto_labelling_action.setShortcut(QKeySequence("Ctrl+Shift+L"))
            auto_labelling_action.setShortcutVisibleInContextMenu(True)
        # Copy path submenu
        copy_lwir_action = None
        copy_vis_action = None
        if self.get_image_path:
            menu.addSeparator()
            copy_menu = menu.addMenu("Copy image path")
            if copy_menu:
                copy_lwir_action = copy_menu.addAction("LWIR path")
                copy_vis_action = copy_menu.addAction("Visible path")
        chosen = menu.exec(global_pos)
        if chosen is None:
            return
        if chosen is nav_prev:
            self.prev_image()
            return
        if chosen is nav_next:
            self.next_image()
            return
        for action, reason in reason_actions:
            if chosen is action:
                if action.isChecked():
                    self.apply_mark_reason(base, reason)
                else:
                    self.apply_mark_reason(base, None)
                return
        if chosen is calibration_action:
            if self.dataset_actions.set_calibration_mark(base, calibration_action.isChecked()):
                self.invalidate_overlay_cache(base)
                self.load_image_pair(base)
                self.update_stats()
                self.mark_cache_dirty()
                self.reconcile_filter_state()
            return
        if reanalyze_action and chosen is reanalyze_action:
            if self.schedule_calibration_job(base, True):
                self.status_message(f"Re-running calibration analysis for {base}", 4000)
            return
        if labelling_action and chosen is labelling_action:
            self.enter_labelling_mode()
            return
        if auto_labelling_action and chosen is auto_labelling_action:
            self.enter_auto_labelling_mode()
            return
        if copy_lwir_action and chosen is copy_lwir_action:
            self._copy_image_path(base, "lwir")
            return
        if copy_vis_action and chosen is copy_vis_action:
            self._copy_image_path(base, "visible")
            return

    def _copy_image_path(self, base: str, channel: str) -> None:
        """Copy absolute image path to clipboard."""
        if not self.get_image_path:
            return
        path = self.get_image_path(base, channel)
        if path:
            clipboard = QApplication.clipboard()
            if clipboard:
                clipboard.setText(str(path))
                self.status_message(f"Copied {channel.upper()} path to clipboard", 3000)

    # Marking actions ----------------------------------------------
    def apply_mark_reason(self, base: str, reason: Optional[str]) -> None:
        if not self.state.set_mark_reason(base, reason, REASON_USER):
            return
        self.invalidate_overlay_cache(base)
        self.update_delete_button()
        self.load_image_pair(base)
        self.update_stats()
        self.mark_cache_dirty()

    def handle_reason_shortcut(self, reason: str) -> None:
        base = self.get_current_base()
        if not base:
            return
        mark_entry = self.state.cache_data["marks"].get(base)
        current_reason = mark_entry.get("reason") if isinstance(mark_entry, dict) else None
        if current_reason == reason:
            # Toggle off: same reason pressed again
            self.apply_mark_reason(base, None)
        else:
            self.apply_mark_reason(base, reason)

    def toggle_mark_current(self) -> None:
        base = self.get_current_base()
        if not base:
            return
        if not self.state.toggle_manual_mark(base, REASON_USER):
            return
        self.invalidate_overlay_cache(base)
        self.update_delete_button()
        self.load_image_pair(base)
        self.update_stats()
        self.mark_cache_dirty()
        # Refresh workspace table if available
        if self.refresh_workspace:
            self.refresh_workspace()
