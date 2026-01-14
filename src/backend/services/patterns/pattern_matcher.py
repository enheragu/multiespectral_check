"""Pattern matching helpers for automatic marking.

The intent is to flag images that resemble one of a set of template patterns.
Patterns are loaded from a directory (e.g. `config/patterns/`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from common.log_utils import log_info, log_warning, log_debug, is_debug_enabled

_SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PatternTemplate:
    name: str
    image: np.ndarray  # grayscale uint8


def _read_grayscale(path: Path) -> Optional[np.ndarray]:
    try:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None
    if img is None or img.size == 0:
        return None
    if img.dtype != np.uint8:
        img = img.astype(np.uint8, copy=False)
    return img


def _downscale_max(img: np.ndarray, max_dim: int) -> np.ndarray:
    if max_dim <= 0:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img
    scale = max_dim / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


class PatternMatcher:
    def __init__(
        self,
        patterns_dir: Path,
        *,
        threshold: float = 0.85,
        max_image_dim: int = 640,
        max_template_dim: int = 320,
    ) -> None:
        self.patterns_dir = patterns_dir
        self.threshold = float(threshold)
        self.max_image_dim = int(max_image_dim)
        self.max_template_dim = int(max_template_dim)
        self._templates: List[PatternTemplate] = []

    def load(self) -> int:
        self._templates.clear()
        if not self.patterns_dir.exists() or not self.patterns_dir.is_dir():
            log_warning(f"Pattern directory not found: {self.patterns_dir}", "PATTERN")
            return 0

        log_info(f"Loading patterns from: {self.patterns_dir}", "PATTERN")
        loaded_count = 0

        for path in sorted(self.patterns_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in _SUPPORTED_EXTS:
                continue
            img = _read_grayscale(path)
            if img is None:
                log_warning(f"Failed to load pattern: {path.name}", "PATTERN")
                continue
            original_shape = img.shape
            img = _downscale_max(img, self.max_template_dim)
            self._templates.append(PatternTemplate(name=path.name, image=img))
            log_info(f"  Loaded pattern: {path.name} ({original_shape} -> {img.shape})", "PATTERN")
            loaded_count += 1

        log_info(f"Total patterns loaded: {loaded_count}, threshold: {self.threshold}", "PATTERN")
        return len(self._templates)

    @property
    def has_patterns(self) -> bool:
        return bool(self._templates)

    def match_path(self, image_path: Path) -> bool:
        """Check if image matches any pattern (legacy bool interface)."""
        return self.match_path_detailed(image_path) is not None

    def match_path_detailed(self, image_path: Path) -> Optional[str]:
        """Check if image matches any pattern and return pattern name."""
        img = _read_grayscale(image_path)
        if img is None:
            return None
        return self.match_image_detailed(img)

    def match_any_paths(self, image_paths: Sequence[Optional[Path]]) -> bool:
        """Check if any path matches any pattern (legacy bool interface)."""
        return self.match_any_paths_detailed(image_paths) is not None

    def match_any_paths_detailed(self, image_paths: Sequence[Optional[Path]]) -> Optional[str]:
        """Check if any path matches any pattern and return pattern name."""
        for path in image_paths:
            if not path or not path.exists():
                continue
            pattern_name = self.match_path_detailed(path)
            if pattern_name:
                return pattern_name
        return None

    def match_image(self, gray: np.ndarray) -> bool:
        """Check if image matches any pattern (legacy bool interface)."""
        return self.match_image_detailed(gray) is not None

    def match_image_detailed(self, gray: np.ndarray) -> Optional[str]:
        """Check if image matches any pattern and return pattern name.

        Returns:
            Pattern name (e.g. 'noise.png') if match found, else None.
        """

        if not self._templates:
            if is_debug_enabled("pattern"):
                log_debug("No templates loaded for matching", "PATTERN")
            return None
        if gray is None or gray.size == 0:
            return None
        if len(gray.shape) != 2:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray = _downscale_max(gray, self.max_image_dim)

        for template in self._templates:
            tmpl = template.image
            if tmpl is None or tmpl.size == 0:
                continue
            th, tw = tmpl.shape[:2]
            ih, iw = gray.shape[:2]
            if th > ih or tw > iw:
                # Template larger than target; skip.
                continue
            try:
                res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)

                if is_debug_enabled("pattern"):
                    log_debug(f"  Pattern {template.name}: max_val={max_val:.3f} (threshold={self.threshold})", "PATTERN")

                if max_val >= self.threshold:
                    log_info(f"Pattern match: {template.name} (score={max_val:.3f})", "PATTERN")
                    return template.name
            except Exception as e:
                if is_debug_enabled("pattern"):
                    log_debug(f"  Pattern {template.name} match failed: {e}", "PATTERN")
                continue
        return None
