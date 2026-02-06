"""Pattern matching helpers for automatic marking.

The intent is to flag images that resemble one of a set of template patterns.
Patterns are loaded from a directory (e.g. `config/patterns/`).

Uses multiple strategies for robust matching:
1. Histogram comparison (fast, good for overall similarity)
2. SSIM structural similarity (slower, more accurate)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from common.log_utils import log_info, log_warning, log_debug, is_debug_enabled

_SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PatternTemplate:
    name: str
    image: np.ndarray  # grayscale uint8
    histogram: np.ndarray  # normalized histogram for fast comparison


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


def _compute_histogram(img: np.ndarray) -> np.ndarray:
    """Compute normalized histogram for an image."""
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()


def _resize_to_match(img: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resize image to match target shape (height, width)."""
    if img.shape[:2] == target_shape:
        return img
    return cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)


def _compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute structural similarity index (simplified version)."""
    # Ensure same size
    if img1.shape != img2.shape:
        img2 = _resize_to_match(img2, img1.shape[:2])

    # Constants for stability
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Compute means
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Compute variances
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(np.mean(ssim_map))


class PatternMatcher:
    """Matches images against reference patterns using histogram + SSIM comparison.

    Two-stage matching:
    1. Fast histogram correlation (threshold ~0.7) - quick rejection
    2. SSIM structural similarity (threshold ~0.85) - confirmation
    """

    def __init__(
        self,
        patterns_dir: Path,
        *,
        threshold: float = 0.85,
        hist_threshold: float = 0.7,
        comparison_size: int = 256,
    ) -> None:
        self.patterns_dir = patterns_dir
        self.threshold = float(threshold)  # SSIM threshold
        self.hist_threshold = float(hist_threshold)  # Histogram pre-filter
        self.comparison_size = int(comparison_size)  # Downscale for comparison
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
            # Resize for comparison
            img_resized = cv2.resize(img, (self.comparison_size, self.comparison_size),
                                      interpolation=cv2.INTER_AREA)
            hist = _compute_histogram(img_resized)
            self._templates.append(PatternTemplate(name=path.name, image=img_resized, histogram=hist))
            log_info(f"  Loaded pattern: {path.name} ({original_shape} -> {img_resized.shape})", "PATTERN")
            loaded_count += 1

        log_info(f"Total patterns loaded: {loaded_count}, ssim_threshold: {self.threshold}, hist_threshold: {self.hist_threshold}", "PATTERN")
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

        Uses two-stage matching:
        1. Histogram correlation for quick rejection
        2. SSIM for accurate confirmation

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

        # Resize to comparison size
        gray_resized = cv2.resize(gray, (self.comparison_size, self.comparison_size),
                                   interpolation=cv2.INTER_AREA)
        img_hist = _compute_histogram(gray_resized)

        for template in self._templates:
            # Stage 1: Fast histogram comparison
            hist_score = cv2.compareHist(img_hist, template.histogram, cv2.HISTCMP_CORREL)

            if is_debug_enabled("pattern"):
                log_debug(f"  Pattern {template.name}: hist_corr={hist_score:.3f} (threshold={self.hist_threshold})", "PATTERN")

            if hist_score < self.hist_threshold:
                continue  # Quick rejection

            # Stage 2: SSIM for confirmation
            ssim_score = _compute_ssim(gray_resized, template.image)

            if is_debug_enabled("pattern"):
                log_debug(f"  Pattern {template.name}: ssim={ssim_score:.3f} (threshold={self.threshold})", "PATTERN")

            if ssim_score >= self.threshold:
                log_info(f"Pattern match: {template.name} (hist={hist_score:.3f}, ssim={ssim_score:.3f})", "PATTERN")
                return template.name

        return None
