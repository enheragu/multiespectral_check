"""Shared UI style helpers for dialogs and panels."""

APP_BG = "#eceff3"  # light gray app background
CARD_BG = "#f9fafc"  # almost-white panels
GROUP_BG = CARD_BG
GROUP_BORDER = "#dfe3e8"
TEXT_PRIMARY = "#2b3035"  # dark gray body text
TEXT_TITLE = "#0f1115"
TEXT_LIGHT = "#1c2230"
BODY_FONT_SIZE = "14.5px"
HEADING_FONT_SIZE = 16.0
MENU_BG = "#2d333d"
MENU_BG_HOVER = "#3a414c"
MENU_FG = "#f5f7fb"
TABLE_BG = "#ffffff"
TABLE_ALT_BG = "#f6f7f9"
TABLE_SELECT_BG = "#dbe7ff"
TABLE_SELECT_FG = "#1c2230"
BUTTON_BG = "#f9fafc"
BUTTON_BG_HOVER = "#edf1f7"
BUTTON_BORDER = GROUP_BORDER
BUTTON_BORDER_STRONG = "#c5ced9"
BUTTON_BG_GRADIENT = "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #fdfefe, stop:1 #edf2f8)"
SECTION_TITLE_STYLE = (
    f"font-size: 12.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.55px;"
    f" color: {TEXT_TITLE};"
)
MONO_TEXT_STYLE = "font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 12.5px;"
TITLE_STYLE_BASE = f"font-weight: 700; color: {TEXT_TITLE}; background: transparent;"


def card_style(object_name: str) -> str:
    return (
        f"#{object_name} {{ background: {CARD_BG}; border-radius: 8px; }}"
        f"#{object_name} QLabel {{ color: {TEXT_PRIMARY}; font-size: {BODY_FONT_SIZE}; background: transparent; }}"
    )


def panel_body_style(object_name: str) -> str:
    return (
        f"#{object_name} {{ background: {CARD_BG}; border: 1px solid {GROUP_BORDER};"
        f" border-radius: 8px; padding: 10px 12px; font-size: {BODY_FONT_SIZE}; }}"
        f"#{object_name} QLabel {{ background: transparent; }}"
    )


def group_box_style() -> str:
    return (
        f"QGroupBox {{ border: 1px solid {GROUP_BORDER}; border-radius: 7px;"
        f" background: {GROUP_BG}; color: {TEXT_PRIMARY}; font-size: {BODY_FONT_SIZE}; }}"
        # margin-top: 15px; padding-top: 15px;
        f"QGroupBox::title {{ font-weight: 700; font-size: 13px; padding: 15 5px; color: {TEXT_TITLE};"
        f" subcontrol-origin: margin; subcontrol-position: top left; margin-left: 6px; }}"
        f"QLabel {{ color: {TEXT_PRIMARY}; }}"
    )


def group_box_with_title_style(title_size: float = 13.0) -> str:
    base = group_box_style()
    return base.replace("font-size: 13px", f"font-size: {title_size}px")


def heading_style(size: float = HEADING_FONT_SIZE) -> str:
    """Shared heading style for panel titles."""
    return f"font-size: {size}px; {TITLE_STYLE_BASE}"


def table_widget_style() -> str:
    return (
        f"QTableWidget {{ background: {TABLE_BG}; alternate-background-color: {TABLE_ALT_BG};"
        f" color: {TEXT_PRIMARY}; selection-background-color: {TABLE_SELECT_BG};"
        f" selection-color: {TABLE_SELECT_FG}; gridline-color: {GROUP_BORDER}; }}"
    )


def app_stylesheet() -> str:
    """Application-wide stylesheet with consistent background and text colors."""
    return (
        f"QMainWindow {{ background: {APP_BG}; color: {TEXT_PRIMARY}; font-size: {BODY_FONT_SIZE}; }}"
        f"QDialog {{ background: {APP_BG}; color: {TEXT_PRIMARY}; font-size: {BODY_FONT_SIZE}; }}"
        f"QMessageBox {{ background: {APP_BG}; color: {TEXT_PRIMARY}; font-size: {BODY_FONT_SIZE}; }}"
        f"QWidget {{ color: {TEXT_PRIMARY}; background: {APP_BG}; font-size: {BODY_FONT_SIZE}; }}"
        f"QLabel {{ color: {TEXT_PRIMARY}; background: transparent; font-size: {BODY_FONT_SIZE}; }}"
        f"QGroupBox {{ color: {TEXT_PRIMARY}; font-size: {BODY_FONT_SIZE}; }}"
        f"QMenuBar {{ background: {MENU_BG}; color: {MENU_FG}; }}"
        f"QMenuBar::item:selected {{ background: {MENU_BG_HOVER}; color: {MENU_FG}; }}"
        f"QMenu {{ background: {MENU_BG}; color: {MENU_FG}; }}"
        f"QMenu::item:selected {{ background: {MENU_BG_HOVER}; color: {MENU_FG}; }}"
        f"QToolTip {{ color: {TEXT_PRIMARY}; background-color: {GROUP_BG}; border: 1px solid {GROUP_BORDER}; }}"
        f"QPushButton {{ color: {TEXT_PRIMARY}; background: {BUTTON_BG_GRADIENT};"
        f" border: 1px solid {BUTTON_BORDER_STRONG}; border-radius: 6px; padding: 5px 10px;"
        f" min-height: 24px; font-size: {BODY_FONT_SIZE}; }}"
        f"QPushButton:hover {{ background: {BUTTON_BG_HOVER}; border-color: {TEXT_TITLE}; }}"
        f"QPushButton:pressed {{ background: #dfe6f1; border-color: {TEXT_PRIMARY}; }}"
        f"QPushButton:focus {{ outline: none; border-color: {TEXT_TITLE}; }}"
        f"QPushButton:disabled {{ color: #8d95a3; background: #f3f4f6; border-color: {BUTTON_BORDER}; }}"
        f"QDialogButtonBox QPushButton {{ color: {TEXT_PRIMARY}; background: {BUTTON_BG_GRADIENT};"
        f" border: 1px solid {BUTTON_BORDER_STRONG}; border-radius: 6px; padding: 5px 10px; min-height: 24px;"
        f" font-size: {BODY_FONT_SIZE}; }}"
        f"QDialogButtonBox QPushButton:hover {{ background: {BUTTON_BG_HOVER}; border-color: {BUTTON_BORDER_STRONG}; }}"
        f"QDialogButtonBox QPushButton:pressed {{ background: #dfe6f1; border-color: {TEXT_PRIMARY}; }}"
        f"QDialogButtonBox QPushButton:disabled {{ color: #8d95a3; background: #f3f4f6; border-color: {BUTTON_BORDER}; }}"
        f"QLineEdit, QTextEdit, QPlainTextEdit {{ background: {CARD_BG}; color: {TEXT_PRIMARY};"
        f" border: 1px solid {GROUP_BORDER}; border-radius: 6px; padding: 6px; font-size: {BODY_FONT_SIZE};"
        f" selection-background-color: {TABLE_SELECT_BG}; selection-color: {TEXT_PRIMARY}; }}"
    )


def section_title_style() -> str:
    return SECTION_TITLE_STYLE


def monospace_text_style() -> str:
    return MONO_TEXT_STYLE.replace("12.5px", BODY_FONT_SIZE)


def apply_app_style(app) -> None:
    """Apply the shared stylesheet to the given QApplication instance."""
    app.setStyleSheet(app_stylesheet())
