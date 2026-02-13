"""Reusable dialog that renders the inline help sections."""
from __future__ import annotations

from typing import Sequence, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QGroupBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from frontend.utils.ui_messages import (
    HELP_CONTEXT_MENUS,
    HELP_DATASET_VIEW,
    HELP_MENU_SECTIONS,
    HELP_OVERVIEW,
    HELP_SHORTCUTS,
    HELP_WORKSPACE_PANEL,
    SUPPORT_EMAIL,
)
from frontend.widgets import style


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

        # Overview
        overview = QLabel(HELP_OVERVIEW, content)
        overview.setWordWrap(True)
        overview.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        content_layout.addWidget(overview)

        # ============ PANELS SECTION ============
        content_layout.addWidget(self._major_section_header("Panels"))

        content_layout.addWidget(self._section_header("Workspace Panel"))
        content_layout.addWidget(self._build_entry_group(HELP_WORKSPACE_PANEL))

        content_layout.addWidget(self._section_header("Dataset / Collection View"))
        content_layout.addWidget(self._build_entry_group(HELP_DATASET_VIEW))

        # ============ CONTEXT MENUS SECTION ============
        content_layout.addWidget(self._major_section_header("Context Menus (Right-Click)"))

        for title, entries in HELP_CONTEXT_MENUS:
            content_layout.addWidget(self._section_header(title))
            content_layout.addWidget(self._build_entry_group(entries))

        # ============ MENUS SECTION ============
        content_layout.addWidget(self._major_section_header("Menus"))

        # Menus in menubar order: File, View, Workspace, Dataset, Calibration, Labelling, Help
        menu_order = ["File", "View", "Workspace", "Dataset", "Calibration", "Labelling", "Help"]
        menu_dict = {title: entries for title, entries in HELP_MENU_SECTIONS}
        for menu_name in menu_order:
            if menu_name in menu_dict:
                content_layout.addWidget(self._section_header(f"{menu_name} Menu"))
                content_layout.addWidget(self._build_entry_group(menu_dict[menu_name]))

        # ============ SHORTCUTS SECTION ============
        content_layout.addWidget(self._major_section_header("Keyboard Shortcuts"))
        content_layout.addWidget(self._build_shortcuts_group())

        # Contact
        content_layout.addWidget(self._section_header("Contact"))
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

    def _build_entry_group(self, entries: Sequence[Tuple[str, str]]) -> QGroupBox:
        """Build a group with vertical layout: each entry is label (bold) + description below."""
        group = QGroupBox(self)
        group.setStyleSheet(GROUP_BOX_STYLE)
        vbox = QVBoxLayout(group)
        vbox.setSpacing(8)
        for label_text, description in entries:
            # Label in bold
            label_widget = QLabel(f"<b>{label_text}</b>", group)
            label_widget.setTextFormat(Qt.TextFormat.RichText)
            label_widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            vbox.addWidget(label_widget)
            # Description below, indented slightly
            desc_widget = QLabel(description, group)
            desc_widget.setWordWrap(True)
            desc_widget.setContentsMargins(12, 0, 0, 4)
            desc_widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            vbox.addWidget(desc_widget)
        return group

    def _build_shortcuts_group(self) -> QGroupBox:
        """Build shortcuts group with monospace keys."""
        group = QGroupBox(self)
        group.setStyleSheet(GROUP_BOX_STYLE)
        vbox = QVBoxLayout(group)
        vbox.setSpacing(6)
        for combo, description in HELP_SHORTCUTS:
            # Key combo in monospace
            label_widget = QLabel(f"<code style='font-family: monospace; font-weight: bold;'>{combo}</code>", group)
            label_widget.setTextFormat(Qt.TextFormat.RichText)
            label_widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            vbox.addWidget(label_widget)
            # Description below
            desc_widget = QLabel(description, group)
            desc_widget.setWordWrap(True)
            desc_widget.setContentsMargins(12, 0, 0, 4)
            desc_widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            vbox.addWidget(desc_widget)
        return group

    def _section_header(self, text: str) -> QLabel:
        """Subsection header (medium size)."""
        label = QLabel(text)
        label.setStyleSheet(SECTION_HEADER_STYLE + "margin-top: 8px;")
        return label

    def _major_section_header(self, text: str) -> QLabel:
        """Major section header (larger, with separator line effect)."""
        label = QLabel(f"━━━  {text}  ━━━")
        label.setStyleSheet(
            f"font-size: 16px; font-weight: bold; color: {style.TEXT_TITLE}; "
            "margin-top: 20px; margin-bottom: 4px;"
        )
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label
