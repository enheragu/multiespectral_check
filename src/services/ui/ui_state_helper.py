"""Keeps UI elements aligned with session state (metadata, stats, delete/restore toggles)."""

from __future__ import annotations

from typing import Optional

from utils.ui_messages import DELETE_BUTTON_TEXT, RESTORE_ACTION_TEXT


class UiStateHelper:
    """Updates UI fragments (metadata, stats, delete/restore buttons)."""

    def __init__(self, ui, stats_panel, session, state) -> None:
        self.ui = ui
        self.stats_panel = stats_panel
        self.session = session
        self.state = state

    def update_metadata_panel(self, base: str, type_dir: str, widget) -> None:
        widget.clear()
        widget.setPlainText(self.session.get_metadata_text(base, type_dir))

    def update_delete_button(self) -> None:
        count = len(self.state.marked_for_delete)
        base_text = DELETE_BUTTON_TEXT
        if count:
            self.ui.btn_delete_marked.setText(f"{base_text} ({count})")
        else:
            self.ui.btn_delete_marked.setText(base_text)
        self.ui.btn_delete_marked.setEnabled(count > 0)
        delete_action = getattr(self.ui, "action_delete_selected", None)
        if delete_action:
            delete_action.setEnabled(count > 0)

    def update_restore_menu(self) -> None:
        base_text = RESTORE_ACTION_TEXT
        if not hasattr(self.ui, "action_restore_images"):
            return
        count = self.session.count_trash_pairs()
        if count:
            self.ui.action_restore_images.setText(f"{base_text} ({count})")
        else:
            self.ui.action_restore_images.setText(base_text)
        self.ui.action_restore_images.setEnabled(self.session.loader is not None)

    def update_stats_panel(self) -> None:
        if not hasattr(self, "stats_panel") or self.stats_panel is None:
            return
        missing_counts = {"lwir": 0, "visible": 0}
        if self.session.loader:
            missing_counts = self.session.loader.missing_channel_counts()
        self.stats_panel.update_from_state(
            self.state,
            self.session.total_pairs(),
            missing_counts,
            self.state.calibration_reproj_errors,
            self.state.extrinsic_pair_errors,
        )
