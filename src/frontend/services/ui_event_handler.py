"""UI event handler for keyboard shortcuts and menu actions.

Centralizes keyboard shortcut registration and menu action connections
to reduce clutter in the main ImageViewer class.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List

from PyQt6.QtCore import Qt, QObject
from PyQt6.QtGui import QKeySequence, QShortcut

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget


class UIEventHandler(QObject):
    """Manages keyboard shortcuts and menu action connections."""

    def __init__(self, parent: QWidget) -> None:
        """Initialize event handler.

        Args:
            parent: Parent widget (typically ImageViewer)
        """
        super().__init__(parent)
        self.parent_widget = parent
        self._shortcuts: List[QShortcut] = []

    def register_navigation_shortcuts(
        self,
        prev_handler: Callable[[], None],
        next_handler: Callable[[], None],
    ) -> None:
        """Register keyboard shortcuts for image navigation.

        Args:
            prev_handler: Callback for previous image
            next_handler: Callback for next image
        """
        shortcuts = [
            (Qt.Key.Key_Left, prev_handler),
            (Qt.Key.Key_Right, next_handler),
            (Qt.Key.Key_Space, next_handler),
        ]

        for key, handler in shortcuts:
            shortcut = QShortcut(key, self.parent_widget)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(handler)
            self._shortcuts.append(shortcut)

    def register_marking_shortcuts(
        self,
        toggle_mark: Callable[[], None],
        toggle_calibration: Callable[[], None],
        calibration_sequence: str,
        reason_shortcuts: Dict[str, str],
        reason_handler: Callable[[str], None],
    ) -> None:
        """Register keyboard shortcuts for marking operations.

        Args:
            toggle_mark: Callback for toggling delete mark
            toggle_calibration: Callback for toggling calibration mark
            calibration_sequence: Key sequence for calibration toggle (e.g., "Ctrl+Shift+C")
            reason_shortcuts: Map of reason -> key sequence
            reason_handler: Callback that receives reason string
        """
        delete_shortcut = QShortcut(Qt.Key.Key_Delete, self.parent_widget)
        delete_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        delete_shortcut.activated.connect(toggle_mark)
        self._shortcuts.append(delete_shortcut)

        c_shortcut = QShortcut(Qt.Key.Key_C, self.parent_widget)
        c_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        c_shortcut.activated.connect(toggle_calibration)
        self._shortcuts.append(c_shortcut)

        calib_shortcut = QShortcut(QKeySequence(calibration_sequence), self.parent_widget)
        calib_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        calib_shortcut.activated.connect(toggle_calibration)
        self._shortcuts.append(calib_shortcut)

        for reason, sequence in reason_shortcuts.items():
            if not sequence or sequence.lower() == "del":
                continue
            reason_shortcut = QShortcut(QKeySequence(sequence), self.parent_widget)
            reason_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            reason_shortcut.activated.connect(lambda r=reason: reason_handler(r))
            self._shortcuts.append(reason_shortcut)

    def clear_shortcuts(self) -> None:
        """Clear all registered shortcuts."""
        for shortcut in self._shortcuts:
            shortcut.setParent(None)
            shortcut.deleteLater()
        self._shortcuts.clear()

    def shortcut_count(self) -> int:
        """Get number of registered shortcuts.

        Returns:
            Count of active shortcuts
        """
        return len(self._shortcuts)
