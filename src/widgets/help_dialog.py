"""Reusable dialog that renders the inline help sections."""
from __future__ import annotations

from typing import Sequence, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from utils.ui_messages import HELP_MENU_SECTIONS, HELP_OVERVIEW, HELP_SHORTCUTS, SUPPORT_EMAIL
from widgets import style


SECTION_HEADER_STYLE = style.heading_style()
GROUP_BOX_STYLE = style.group_box_style()


class HelpDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Help")
        self.resize(720, 540)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        card = QWidget(self)
        card.setObjectName("help_card")
        card.setStyleSheet(style.card_style("help_card"))

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 14, 14, 14)
        card_layout.setSpacing(10)

        scroll = QScrollArea(card)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        content = QWidget(scroll)
        content.setStyleSheet("background: transparent;")
        content_layout = QVBoxLayout(content)
        overview = QLabel(HELP_OVERVIEW, content)
        overview.setWordWrap(True)
        overview.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        content_layout.addWidget(overview)
        for title, entries in HELP_MENU_SECTIONS:
            content_layout.addWidget(self._section_header(f"{title} Menu:"))
            content_layout.addWidget(self._build_menu_group(entries))
        content_layout.addWidget(self._section_header("Shortcuts:"))
        content_layout.addWidget(self._build_shortcuts_group())
        content_layout.addWidget(self._section_header("Contact:"))
        contact_box = QWidget(content)
        contact_box.setObjectName("contact_box")
        contact_box.setStyleSheet(style.panel_body_style("contact_box"))
        contact_layout = QVBoxLayout(contact_box)
        contact_layout.setContentsMargins(10, 8, 10, 8)
        contact = QLabel(f'Email: <a href="mailto:{SUPPORT_EMAIL}">{SUPPORT_EMAIL}</a>', contact_box)
        contact.setTextFormat(Qt.TextFormat.RichText)
        contact.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        contact.setOpenExternalLinks(True)
        contact_layout.addWidget(contact)
        content_layout.addWidget(contact_box)
        content_layout.addStretch(1)

        scroll.setWidget(content)
        card_layout.addWidget(scroll)

        layout.addWidget(card)

    def _build_menu_group(self, entries: Sequence[Tuple[str, str]]) -> QGroupBox:
        group = QGroupBox(self)
        group.setStyleSheet(GROUP_BOX_STYLE)
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        for label, description in entries:
            label_widget = QLabel(label, group)
            label_widget.setStyleSheet("font-weight: 600;")
            label_widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            description_widget = QLabel(description, group)
            description_widget.setWordWrap(True)
            description_widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            form.addRow(label_widget, description_widget)
        return group

    def _build_shortcuts_group(self) -> QGroupBox:
        group = QGroupBox(self)
        group.setStyleSheet(GROUP_BOX_STYLE)
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        for combo, description in HELP_SHORTCUTS:
            combo_label = QLabel(combo, group)
            combo_label.setStyleSheet("font-family: monospace;")
            combo_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            description_widget = QLabel(description, group)
            description_widget.setWordWrap(True)
            description_widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            form.addRow(combo_label, description_widget)
        return group

    def _section_header(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(SECTION_HEADER_STYLE + "margin-top: 12px;")
        return label
