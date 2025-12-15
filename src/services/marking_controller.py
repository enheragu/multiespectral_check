"""Controls marking interactions and context menus for dataset items."""

from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import QMenu

from utils.reasons import REASON_CHOICES, REASON_SHORTCUTS, REASON_USER
from services.dataset_actions import DatasetActions


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

    # Context menu -------------------------------------------------
    def show_context_menu(self, global_pos) -> None:
        if not self.has_images():
            return
        base = self.get_current_base()
        if not base:
            return
        menu = QMenu(self.parent)
        nav_prev = menu.addAction("Previous image")
        nav_prev.setShortcut(QKeySequence("Left Arrow"))
        nav_prev.setShortcutVisibleInContextMenu(True)
        nav_next = menu.addAction("Next image")
        nav_next.setShortcut(QKeySequence("Right Arrow"))
        nav_next.setShortcutVisibleInContextMenu(True)
        menu.addSeparator()
        current_reason = self.state.marked_for_delete.get(base)
        reason_actions = []
        for reason, label in REASON_CHOICES:
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(current_reason == reason)
            shortcut = REASON_SHORTCUTS.get(reason)
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
                action.setShortcutVisibleInContextMenu(True)
            reason_actions.append((action, reason))
        menu.addSeparator()
        calibration_action = menu.addAction("Mark as calibration candidate")
        calibration_action.setShortcut(QKeySequence(self.calibration_shortcut))
        calibration_action.setShortcutVisibleInContextMenu(True)
        calibration_action.setCheckable(True)
        calibration_action.setChecked(base in self.state.calibration_marked)
        reanalyze_action = None
        if base in self.state.calibration_marked:
            reanalyze_action = menu.addAction("Re-run calibration detection")
            reanalyze_action.setShortcut(QKeySequence("Ctrl+Shift+R"))
            reanalyze_action.setShortcutVisibleInContextMenu(True)
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
        current = self.state.marked_for_delete.get(base)
        if current == reason:
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
