"""Reusable dialog that renders the inline help sections."""
from __future__ import annotations

from typing import Sequence, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from frontend.resources import icon_path
from config import APP_DESCRIPTION, APP_NAME, APP_VERSION, ISSUES_URL, REPO_URL, SUPPORT_EMAIL
from frontend.utils.ui_messages import (
    HELP_CONTEXT_MENUS,
    HELP_DATASET_VIEW,
    HELP_MENU_SECTIONS,
    HELP_OVERVIEW,
    HELP_SHORTCUTS,
    HELP_WORKSPACE_PANEL,
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

        banner = self._build_banner(card)
        if banner is not None:
            card_layout.addWidget(banner)

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
        contact = QLabel(
            f'Maintainer: <a href="mailto:{SUPPORT_EMAIL}">{SUPPORT_EMAIL}</a><br>'
            'Repository: <a href="https://github.com/enheragu/multiespectral_check">'
            'github.com/enheragu/multiespectral_check</a><br>'
            'Issues: <a href="https://github.com/enheragu/multiespectral_check/issues">'
            'Report a problem or request a feature</a>',
            contact_box,
        )
        contact.setTextFormat(Qt.TextFormat.RichText)
        contact.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        contact.setOpenExternalLinks(True)
        contact.setWordWrap(True)
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

    def _build_banner(self, parent: QWidget) -> QLabel | None:
        """Return a centered, scaled banner label, or None if the asset is missing."""
        pix = QPixmap(icon_path("banner.png"))
        if pix.isNull():
            return None
        # Scale to a reasonable header height while preserving aspect ratio.
        target_height = 120
        scaled = pix.scaledToHeight(
            target_height, Qt.TransformationMode.SmoothTransformation
        )
        label = QLabel(parent)
        label.setPixmap(scaled)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

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


class AboutDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About")
        self.resize(520, 340)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        card = QWidget(self)
        card.setObjectName("about_card")
        card.setStyleSheet(style.card_style("about_card"))
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 14, 14, 14)
        card_layout.setSpacing(10)

        banner = self._build_banner(card)
        if banner is not None:
            card_layout.addWidget(banner)

        # title = QLabel(APP_NAME, card)
        # title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # title.setStyleSheet("font-size: 18px; font-weight: bold; color: " + style.TEXT_TITLE + ";")
        # card_layout.addWidget(title)

        description = QLabel(
            APP_DESCRIPTION,
            card,
        )
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(description)

        links = QLabel(
            f'Repository: <a href="{REPO_URL}">GitHub repository</a><br>'
            f'Issues: <a href="{ISSUES_URL}">GitHub issues</a><br>'
            f'Contact: <a href="mailto:{SUPPORT_EMAIL}">{SUPPORT_EMAIL}</a>',
            card,
        )
        links.setTextFormat(Qt.TextFormat.RichText)
        links.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        links.setOpenExternalLinks(True)
        links.setWordWrap(True)
        links.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(links)

        version = QLabel(f"Version: {APP_VERSION}", card)
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(version)
        card_layout.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        card_layout.addWidget(buttons)

        layout.addWidget(card)

    def _build_banner(self, parent: QWidget) -> QLabel | None:
        pix = QPixmap(icon_path("banner.png"))
        if pix.isNull():
            return None
        scaled = pix.scaledToHeight(120, Qt.TransformationMode.SmoothTransformation)
        label = QLabel(parent)
        label.setPixmap(scaled)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label
