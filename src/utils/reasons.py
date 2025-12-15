"""Centralized reason constants shared across widgets."""
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

REASON_USER = "user_marked"
REASON_DUPLICATE = "duplicate"
REASON_BLURRY = "blurry"
REASON_MOTION = "motion"
REASON_SYNC = "sync_error"
REASON_MISSING_PAIR = "missing_pair"
REASON_CHOICES = [
    (REASON_USER, "Manual deletion candidate"),
    (REASON_DUPLICATE, "Set as duplicate"),
    (REASON_BLURRY, "Set as blurry"),
    (REASON_MOTION, "Set as motion blur"),
    (REASON_SYNC, "Set as sync mismatch"),
    (REASON_MISSING_PAIR, "Missing pair"),
]
REASON_STYLES = {
    REASON_USER: {"color": QColor("#d32f2f"), "text": "Marked by user"},
    REASON_DUPLICATE: {"color": QColor("#1976d2"), "text": "Duplicate candidate"},
    REASON_BLURRY: {"color": QColor("#ff6004"), "text": "Blurry image"},
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
    REASON_MISSING_PAIR: None,
}
REASON_KEY_MAP = {
    Qt.Key.Key_D: REASON_DUPLICATE,
    Qt.Key.Key_B: REASON_BLURRY,
    Qt.Key.Key_M: REASON_MOTION,
    Qt.Key.Key_S: REASON_SYNC,
}
