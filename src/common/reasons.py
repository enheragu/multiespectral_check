"""Centralized reason constants shared across widgets."""
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

REASON_USER = "user_marked"
REASON_DUPLICATE = "duplicate"
REASON_BLURRY = "blurry"
REASON_MOTION = "motion"
REASON_SYNC = "sync_error"
REASON_MISSING_PAIR = "missing_pair"
REASON_PATTERN = "pattern"

# Auto-detected reasons (never manually marked)
AUTO_REASONS = [
    REASON_MISSING_PAIR,
    REASON_DUPLICATE,
    REASON_BLURRY,
    REASON_MOTION,
    REASON_PATTERN,
]

REASON_CHOICES = [
    (REASON_USER, "Manual delete"),
    (REASON_DUPLICATE, "Duplicate"),
    (REASON_BLURRY, "Blurry"),
    (REASON_MOTION, "Motion blur"),
    (REASON_SYNC, "Sync mismatch"),
    (REASON_MISSING_PAIR, "Missing pair"),
    (REASON_PATTERN, "Pattern match"),
]
REASON_STYLES = {
    REASON_USER: {"color": QColor("#d32f2f"), "text": "Manual delete"},
    REASON_DUPLICATE: {"color": QColor("#1976d2"), "text": "Duplicate"},
    REASON_PATTERN: {"color": QColor("#7b1fa2"), "text": "Pattern match"},
    REASON_BLURRY: {"color": QColor("#ff6004"), "text": "Blurry"},
    REASON_MOTION: {"color": QColor("#8e44ad"), "text": "Motion blur"},
    REASON_SYNC: {"color": QColor("#2ecc71"), "text": "Sync mismatch"},
    REASON_MISSING_PAIR: {"color": QColor("#455a64"), "text": "Missing pair"},
}
REASON_SHORTCUTS = {
    REASON_USER: "Del",
    REASON_DUPLICATE: "Ctrl+Shift+D",
    REASON_BLURRY: "Ctrl+Shift+B",
    REASON_MOTION: "Ctrl+Shift+M",
    REASON_SYNC: "Ctrl+Shift+S",
    REASON_PATTERN: "Ctrl+Shift+P",
    REASON_MISSING_PAIR: None,
}
REASON_KEY_MAP = {
    Qt.Key.Key_D: REASON_DUPLICATE,
    Qt.Key.Key_B: REASON_BLURRY,
    Qt.Key.Key_M: REASON_MOTION,
    Qt.Key.Key_S: REASON_SYNC,
    Qt.Key.Key_P: REASON_PATTERN,
}


def reason_text(reason: str) -> str:
    style = REASON_STYLES.get(reason, {}) if isinstance(REASON_STYLES, dict) else {}
    text = style.get("text") if isinstance(style, dict) else None
    return text if isinstance(text, str) else reason


def format_reason_label(reason: str, *, auto: bool) -> str:
    prefix = "Auto" if auto else "Manual"
    return f"{prefix}: {reason_text(reason)}"
