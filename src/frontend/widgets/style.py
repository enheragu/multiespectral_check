"""Shared UI style helpers for dialogs and panels."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

_MEDIA_DIR = Path(__file__).parent.parent / "resources" / "media"
_CHECKMARK_SVG = str(_MEDIA_DIR / "checkmark_white.svg")

APP_BG = "#eceff3"  # light gray app background
CARD_BG = "#f9fafc"  # almost-white panels
GROUP_BG = CARD_BG
GROUP_BORDER = "#dfe3e8"
TEXT_PRIMARY = "#2b3035"  # dark gray body text
TEXT_TITLE = "#0f1115"
TEXT_LIGHT = "#1c2230"
TEXT_SECONDARY = "#444444"  # muted / helper text
TEXT_CAPTION = "#666666"    # small captions and hints
BODY_FONT_SIZE = "14.5px"
HEADING_FONT_SIZE = 16.0
MENU_BG = "#2d333d"
MENU_BG_HOVER = "#3a414c"
MENU_FG = "#f5f7fb"
TABLE_BG = "#ffffff"
TABLE_ALT_BG = "#f6f7f9"
TABLE_SELECT_BG = "#dbe7ff"
TABLE_SELECT_FG = "#1c2230"
ACCENT = "#4476c4"        # primary action / interactive accent (blue family of TABLE_SELECT_BG)
ACCENT_HOVER = "#3a68b8"  # darker on hover
ACCENT_LIGHT = "#b8c8e0"  # muted accent for disabled/indeterminate states
BUTTON_BG = "#f9fafc"
BUTTON_BG_HOVER = "#edf1f7"
BUTTON_BORDER = GROUP_BORDER
BUTTON_BORDER_STRONG = "#c5ced9"
BUTTON_BG_GRADIENT = "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #fefefe, stop:1 #e1e6ef)"
BUTTON_BG_DISABLED = BUTTON_BG_GRADIENT
BUTTON_HEIGHT = 18

# Shared chart rendering constants (used by calibration and label report charts)
CHART_DPR = 2
CHART_AXIS_COLOR = "#444444"
CHART_FONT_SIZE = 9
CHART_TITLE_FONT_SIZE = 10
CHART_MARGINS = (40, 28, 22, 10)  # (left, bottom, top, right)
CHART_PLOT_BG = "white"

# Chart pixmap sizes (logical pixels, DPR applied separately)
CHART_W_PAIR = 430   # width for charts shown two side-by-side (calibration coverage/distortion)
CHART_H_PAIR = 310   # height for standard paired charts
CHART_W_GRID = 380   # width for charts in a 2×2 grid (label report)
CHART_H_GRID = 280   # height for grid charts (also pose-diversity rows)

# Recommended minimum dialog width for report/analysis dialogs (2×CHART_W_PAIR + all margins)
DIALOG_REPORT_MIN_W = 960

# Pre-built QColor objects for chart rendering (QColor is safe without QApplication)
CHART_AXIS_QCOLOR = QColor(CHART_AXIS_COLOR)
CHART_PLOT_BG_QCOLOR = QColor(CHART_PLOT_BG)
CHART_BG_QCOLOR = QColor(CARD_BG)
CHART_PLACEHOLDER_QCOLOR = QColor("#999999")  # "no data" borders and placeholder outlines

SECTION_TITLE_STYLE = (
    f"font-size: 12.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.55px;"
    f" color: {TEXT_TITLE};"
)
MONO_TEXT_STYLE = "font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 12.5px;"
TITLE_STYLE_BASE = f"font-weight: 700; color: {TEXT_TITLE}; background: transparent;"


def card_style(object_name: str) -> str:
    return (
        f"#{object_name} {{ background: {TABLE_BG}; border-radius: 8px; }}"
        f"#{object_name} QLabel {{ color: {TEXT_PRIMARY}; font-size: {BODY_FONT_SIZE}; background: transparent; }}"
    )


def panel_body_style(object_name: str) -> str:
    return (
        f"#{object_name} {{ background: {TABLE_BG}; border: 1px solid {GROUP_BORDER};"
        f" border-radius: 8px; padding: 10px 12px; font-size: {BODY_FONT_SIZE}; }}"
        f"#{object_name} QLabel {{ background: transparent; }}"
    )


def group_box_style() -> str:
    """Style for untitled QGroupBoxes (no margin-top)."""
    return (
        f"QGroupBox {{ border: 1px solid {GROUP_BORDER}; border-radius: 7px;"
        f" background: {TABLE_BG}; color: {TEXT_PRIMARY}; font-size: {BODY_FONT_SIZE}; }}"
        f"QLabel {{ background: transparent; color: {TEXT_PRIMARY}; }}"
    )


def titled_group_box_style(title_size: float = 13.0) -> str:
    """Style for QGroupBoxes WITH a visible title (includes margin-top for the title area)."""
    return (
        f"QGroupBox {{ border: 1px solid {GROUP_BORDER}; border-radius: 7px; margin-top: 16px;"
        f" background: {TABLE_BG}; color: {TEXT_PRIMARY}; font-size: {BODY_FONT_SIZE}; }}"
        f"QGroupBox::title {{ font-weight: 700; font-size: {title_size}px; padding: 0 5px;"
        f" color: {TEXT_TITLE}; subcontrol-origin: margin; subcontrol-position: top left;"
        f" margin-left: 6px; }}"
        f"QLabel {{ background: transparent; color: {TEXT_PRIMARY}; }}"
    )


def heading_style(size: float = HEADING_FONT_SIZE) -> str:
    """Shared heading style for panel titles."""
    return f"font-size: {size}px; {TITLE_STYLE_BASE}"

def button_style() -> str:
    """Inline style for QPushButton to guarantee consistent color/height."""
    return (
        f"color: {TEXT_PRIMARY}; background: {BUTTON_BG_GRADIENT};"
        f" border: 1px solid {BUTTON_BORDER_STRONG}; border-radius: 6px; padding: 5px 10px;"
        f" min-height: {BUTTON_HEIGHT}px; max-height: {BUTTON_HEIGHT}px; font-size: {BODY_FONT_SIZE};"
    )


def scoped_button_style(object_name: str) -> str:
    """Per-button stylesheet including hover/pressed/disabled states."""
    return (
        f"#{object_name} {{ color: {TEXT_PRIMARY}; background: {BUTTON_BG_GRADIENT};"
        f" border: 1px solid {BUTTON_BORDER_STRONG}; border-radius: 6px; padding: 5px 10px;"
        f" min-height: {BUTTON_HEIGHT}px; max-height: {BUTTON_HEIGHT}px; font-size: {BODY_FONT_SIZE}; }}"
        f"#{object_name}:hover {{ background: {BUTTON_BG_HOVER}; border-color: {TEXT_TITLE}; }}"
        f"#{object_name}:pressed {{ background: #dfe6f1; border-color: {TEXT_PRIMARY}; }}"
        f"#{object_name}:focus {{ outline: none; border-color: {TEXT_TITLE}; }}"
        f"#{object_name}:disabled {{ color: {TEXT_PRIMARY}; background: {BUTTON_BG_DISABLED}; border-color: {BUTTON_BORDER_STRONG}; }}"
    )

def table_widget_style() -> str:
    return ""  # covered by app_stylesheet() globally


def app_stylesheet() -> str:
    """Application-wide stylesheet with consistent background and text colors."""
    return (
        f"QMainWindow {{ background: {APP_BG}; color: {TEXT_PRIMARY}; font-size: {BODY_FONT_SIZE}; }}"
        f"QWidget#centralwidget {{ background: {APP_BG}; }}"
        f"QDialog {{ background: {APP_BG}; color: {TEXT_PRIMARY}; font-size: {BODY_FONT_SIZE}; }}"
        f"QMessageBox {{ background: {APP_BG}; color: {TEXT_PRIMARY}; font-size: {BODY_FONT_SIZE}; }}"
        f"QWidget {{ color: {TEXT_PRIMARY}; font-size: {BODY_FONT_SIZE}; }}"
        f"QScrollArea {{ background: {APP_BG}; }}"
        f"QLabel {{ color: {TEXT_PRIMARY}; background: transparent; font-size: {BODY_FONT_SIZE}; }}"
        f"QGroupBox {{ color: {TEXT_PRIMARY}; font-size: {BODY_FONT_SIZE}; }}"
        f"QMenuBar {{ background: {MENU_BG}; color: {MENU_FG}; }}"
        f"QMenuBar::item:selected {{ background: {MENU_BG_HOVER}; color: {MENU_FG}; }}"
        f"QMenu {{ background: {MENU_BG}; color: {MENU_FG}; }}"
        f"QMenu::item:selected {{ background: {MENU_BG_HOVER}; color: {MENU_FG}; }}"
        f"QToolTip {{ color: {TEXT_PRIMARY}; background-color: {GROUP_BG}; border: 1px solid {GROUP_BORDER}; }}"
        f"QPushButton {{ color: {TEXT_PRIMARY}; background: {BUTTON_BG_GRADIENT};"
        f" border: 1px solid {BUTTON_BORDER_STRONG}; border-radius: 6px; padding: 5px 10px;"
        f" min-height: {BUTTON_HEIGHT}px; max-height: {BUTTON_HEIGHT}px; font-size: {BODY_FONT_SIZE}; }}"
        f"QPushButton:hover {{ background: {BUTTON_BG_HOVER}; border-color: {TEXT_TITLE}; }}"
        f"QPushButton:pressed {{ background: #dfe6f1; border-color: {TEXT_PRIMARY}; }}"
        f"QPushButton:focus {{ outline: none; border-color: {TEXT_TITLE}; }}"
        f"QPushButton:disabled {{ color: #8d95a3; background: {BUTTON_BG_DISABLED}; border-color: {BUTTON_BORDER_STRONG}; }}"
        f"QDialogButtonBox QPushButton {{ color: {TEXT_PRIMARY}; background: {BUTTON_BG_GRADIENT};"
        f" border: 1px solid {BUTTON_BORDER_STRONG}; border-radius: 6px; padding: 5px 10px;"
        f" min-height: {BUTTON_HEIGHT}px; max-height: {BUTTON_HEIGHT}px; font-size: {BODY_FONT_SIZE}; }}"
        f"QDialogButtonBox QPushButton:hover {{ background: {BUTTON_BG_HOVER}; border-color: {BUTTON_BORDER_STRONG}; }}"
        f"QDialogButtonBox QPushButton:pressed {{ background: #dfe6f1; border-color: {TEXT_PRIMARY}; }}"
        f"QDialogButtonBox QPushButton:disabled {{ color: #8d95a3; background: {BUTTON_BG_DISABLED}; border-color: {BUTTON_BORDER_STRONG}; }}"
        f"QTreeView, QTreeWidget, QTableView, QTableWidget {{"
        f" background: {TABLE_BG}; alternate-background-color: {TABLE_ALT_BG};"
        f" color: {TEXT_PRIMARY}; selection-background-color: {TABLE_SELECT_BG};"
        f" selection-color: {TABLE_SELECT_FG}; gridline-color: {GROUP_BORDER};"
        f" outline: none; border: 1px solid {GROUP_BORDER}; border-radius: 6px; }}"
        f"QTreeView::item, QTreeWidget::item, QTableView::item, QTableWidget::item {{ padding: 2px 0; }}"
        f"QTreeView::item:hover, QTreeWidget::item:hover, QTableView::item:hover, QTableWidget::item:hover {{ background: {BUTTON_BG_HOVER}; }}"
        f"QTreeView::item:selected, QTreeWidget::item:selected, QTableView::item:selected, QTableWidget::item:selected {{ background: {TABLE_SELECT_BG}; color: {TABLE_SELECT_FG}; }}"
        f"QTreeView::item:selected:active, QTreeWidget::item:selected:active, QTableView::item:selected:active, QTableWidget::item:selected:active {{ background: {TABLE_SELECT_BG}; color: {TABLE_SELECT_FG}; }}"
        f"QHeaderView::section {{"
        f" background: {APP_BG}; color: {TEXT_PRIMARY}; border: none;"
        f" border-bottom: 1px solid {GROUP_BORDER}; border-right: 1px solid {GROUP_BORDER};"
        f" padding: 4px 6px; font-size: {BODY_FONT_SIZE}; font-weight: 600; }}"
        f"QProgressBar {{ background: {CARD_BG}; border: 1px solid {GROUP_BORDER}; border-radius: 6px;"
        f" min-height: {BUTTON_HEIGHT}px; max-height: {BUTTON_HEIGHT}px; padding: 5px 10px;"
        f" font-size: {BODY_FONT_SIZE}; text-align: center; color: {TEXT_PRIMARY}; }}"
        f"QProgressBar::chunk {{ background: {TABLE_SELECT_BG}; border-radius: 5px; }}"
        f"QWidget#tab_workspace, QWidget#tab_dataset {{ background: {CARD_BG}; }}"
        f"QTabWidget::pane {{ border: 1px solid {BUTTON_BORDER_STRONG}; background: {CARD_BG}; top: -1px; }}"
        f"QTabBar {{ background: transparent; }}"
        f"QTabBar::tab {{ background: {APP_BG}; color: #5a6270; font-size: {BODY_FONT_SIZE};"
        f" border: 1px solid {BUTTON_BORDER_STRONG}; border-bottom: none;"
        f" border-top-left-radius: 6px; border-top-right-radius: 6px;"
        f" padding: 5px 16px; min-width: 80px; }}"
        f"QTabBar::tab:selected {{ background: {CARD_BG}; color: {TEXT_TITLE}; font-weight: 600;"
        f" border-color: {BUTTON_BORDER_STRONG}; margin-bottom: -1px; padding-bottom: 6px; }}"
        f"QTabBar::tab:hover:!selected {{ background: {BUTTON_BG_HOVER}; color: {TEXT_PRIMARY}; }}"
        f"QLineEdit, QTextEdit, QPlainTextEdit {{ background: {TABLE_BG}; color: {TEXT_PRIMARY};"
        f" border: 1px solid {GROUP_BORDER}; border-radius: 6px; padding: 6px; font-size: {BODY_FONT_SIZE};"
        f" selection-background-color: {TABLE_SELECT_BG}; selection-color: {TEXT_PRIMARY}; }}"
        f"QCheckBox {{ color: {TEXT_PRIMARY}; font-size: {BODY_FONT_SIZE}; spacing: 6px; }}"
        f"QCheckBox::indicator {{ width: 15px; height: 15px; border: 1px solid {BUTTON_BORDER_STRONG};"
        f" border-radius: 3px; background: {CARD_BG}; }}"
        f"QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT};"
        f" image: url({_CHECKMARK_SVG}); }}"
        f"QCheckBox::indicator:disabled {{ background: {BUTTON_BG_DISABLED}; border-color: {BUTTON_BORDER_STRONG}; }}"
        f"QCheckBox:disabled {{ color: #8d95a3; }}"
    )


def section_title_style() -> str:
    return SECTION_TITLE_STYLE


def monospace_text_style() -> str:
    return MONO_TEXT_STYLE.replace("12.5px", BODY_FONT_SIZE)


def light_palette() -> QPalette:
    """Fixed light QPalette so the app renders identically under any OS theme.

    The stylesheet (app_stylesheet) explicitly colors most widgets, but widgets it does
    NOT target (QSpinBox, QComboBox, QAbstractSpinBox, …) otherwise inherit the SYSTEM
    palette — under a dark OS theme that gives them a dark background while the stylesheet's
    global dark text stays dark, i.e. unreadable black-on-black. Forcing a light palette
    closes that gap at the root, without per-widget QSS that can break spin-box arrows.
    """
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(APP_BG))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(TEXT_PRIMARY))
    pal.setColor(QPalette.ColorRole.Base, QColor(TABLE_BG))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(TABLE_ALT_BG))
    pal.setColor(QPalette.ColorRole.Text, QColor(TEXT_PRIMARY))
    pal.setColor(QPalette.ColorRole.Button, QColor(BUTTON_BG))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT_PRIMARY))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(GROUP_BG))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT_PRIMARY))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(TABLE_SELECT_BG))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(TABLE_SELECT_FG))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(TEXT_CAPTION))
    disabled = QColor("#8d95a3")
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, disabled)
    return pal


def apply_app_style(app) -> None:
    """Apply the shared stylesheet to the given QApplication instance."""
    app.setStyleSheet(app_stylesheet())


def make_panel(
    object_name: str,
    margins: Tuple[int, int, int, int] = (6, 6, 6, 6),
    spacing: int = 6,
) -> Tuple[QWidget, QVBoxLayout]:
    """Create a styled panel widget (panel_body_style) with a QVBoxLayout.

    Returns the panel widget and its layout so callers can add content directly.
    Inner sub-layouts (QFormLayout, QGridLayout) should set their own margins to (0,0,0,0).
    """
    panel = QWidget()
    panel.setObjectName(object_name)
    panel.setStyleSheet(panel_body_style(object_name))
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(*margins)
    layout.setSpacing(spacing)
    return panel, layout


def section_heading_label(text: str, size: float = HEADING_FONT_SIZE) -> QLabel:
    """Create a styled section heading QLabel."""
    label = QLabel(text)
    label.setStyleSheet(heading_style(size))
    return label
