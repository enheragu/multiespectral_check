"""Coordinates overlay prefetching around the current image index."""
from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Optional, Set

from PyQt6.QtCore import QObject, QTimer


class OverlayPrefetcher(QObject):
    def __init__(
        self,
        parent,
        radius: int,
        delay_ms: int,
        ensure_cached: Callable[[str], None],
        is_cached: Callable[[str], bool],
    ) -> None:
        super().__init__(parent)
        self.radius = radius
        self.delay_ms = delay_ms
        self.ensure_cached = ensure_cached
        self.is_cached = is_cached
        self.queue: Deque[str] = deque()
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._drain)

    def clear(self) -> None:
        self.queue.clear()
        if self.timer.isActive():
            self.timer.stop()

    def prepare(
        self,
        *,
        current_index: int,
        total_pairs: int,
        current_base: Optional[str],
        calibration_marked: Set[str],
        get_base: Callable[[int], Optional[str]],
    ) -> None:
        self.queue.clear()
        if not current_base or total_pairs <= 1 or not calibration_marked:
            return
        seen = {current_base}
        for delta in range(1, self.radius + 1):
            for offset in (-delta, delta):
                idx = current_index + offset
                if idx < 0 or idx >= total_pairs:
                    continue
                base = get_base(idx)
                if (
                    not base
                    or base in seen
                    or base not in calibration_marked
                    or self.is_cached(base)
                ):
                    continue
                self.queue.append(base)
                seen.add(base)
        if self.queue:
            self.timer.start(self.delay_ms)

    def _drain(self) -> None:
        if not self.queue:
            return
        base = self.queue.popleft()
        self.ensure_cached(base)
        if self.queue:
            self.timer.start(self.delay_ms)
