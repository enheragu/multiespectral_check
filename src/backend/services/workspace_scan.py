"""Background runnable for scanning workspace datasets on a thread pool."""
from __future__ import annotations

from pathlib import Path
from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

from backend.services.workspace_inspector import scan_workspace


class WorkspaceScanSignals(QObject):
    finished = pyqtSignal(int)


class WorkspaceScanRunnable(QRunnable):
    def __init__(self, workspace_dir: Path) -> None:
        super().__init__()
        self.workspace_dir = workspace_dir
        self.signals = WorkspaceScanSignals()

    def run(self) -> None:
        datasets = scan_workspace(self.workspace_dir)
        try:
            self.signals.finished.emit(len(datasets))
        except RuntimeError:
            # UI may be gone; ignore emit failures
            pass

