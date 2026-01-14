"""Background runnable and signals for pattern sweep across dataset images."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

from backend.services.patterns.pattern_matcher import PatternMatcher


class PatternSweepSignals(QObject):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(dict, int)  # {base: pattern_name}, scanned
    failed = pyqtSignal(str)


class PatternSweepRunnable(QRunnable):
    def __init__(
        self,
        items: List[Tuple[str, Optional[Path], Optional[Path]]],
        *,
        patterns_dir: Path,
        threshold: float,
    ) -> None:
        super().__init__()
        self._items = items
        self._patterns_dir = patterns_dir
        self._threshold = float(threshold)
        self.signals = PatternSweepSignals()

    def run(self) -> None:
        try:
            from tqdm import tqdm

            matcher = PatternMatcher(self._patterns_dir, threshold=self._threshold)
            matcher.load()
            total = len(self._items)
            if total == 0 or not matcher.has_patterns:
                try:
                    self.signals.finished.emit({}, total)
                except RuntimeError:
                    pass
                return
            matched: Dict[str, str] = {}

            with tqdm(total=total, desc="Pattern sweep", unit="img", leave=False) as pbar:
                for idx, (base, vis_path, lwir_path) in enumerate(self._items, start=1):
                    pattern_name = matcher.match_any_paths_detailed([vis_path, lwir_path])
                    if pattern_name:
                        matched[base] = pattern_name
                        pbar.set_postfix({"matched": len(matched)})

                    pbar.update(1)

                    # Emit progress signal periodically for GUI
                    if idx == 1 or idx % 10 == 0 or idx == total:
                        try:
                            self.signals.progress.emit(idx, total, base)
                        except RuntimeError:
                            return

            try:
                self.signals.finished.emit(matched, total)
            except RuntimeError:
                pass
        except Exception as exc:  # noqa: BLE001
            try:
                self.signals.failed.emit(str(exc))
            except RuntimeError:
                pass

