"""Helpers for duplicate detection based on pixmap perceptual signatures."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt

SignatureCache = Dict[str, Dict[str, Optional[bytes]]]
PixmapProvider = Callable[[str, str], Optional[QPixmap]]

SIGNATURE_SIZE = 64
SIGNATURE_THRESHOLD = 0.005


def compute_signature(pixmap: Optional[QPixmap], size: int = SIGNATURE_SIZE) -> Optional[bytes]:
    """Return a compact grayscale signature for the given pixmap."""
    if not pixmap or pixmap.isNull():
        return None
    scaled = pixmap.scaled(
        size,
        size,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    image = scaled.toImage().convertToFormat(QImage.Format.Format_Grayscale8)
    buffer = image.constBits()
    buffer.setsize(image.width() * image.height())
    return bytes(buffer)


def compute_signature_from_path(path: Optional[Path], size: int = SIGNATURE_SIZE) -> Optional[bytes]:
    """Return a signature directly from an image file path."""
    if not path:
        return None
    image = QImage(str(path))
    if image.isNull():
        return None
    scaled = image.scaled(
        size,
        size,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    gray = scaled.convertToFormat(QImage.Format.Format_Grayscale8)
    buffer = gray.constBits()
    buffer.setsize(gray.width() * gray.height())
    return bytes(buffer)


def signature_distance(sig_a: bytes, sig_b: bytes) -> float:
    """Return a normalized distance in [0,1] between two signatures."""
    if len(sig_a) != len(sig_b) or not sig_a:
        return 1.0
    total = sum(abs(a - b) for a, b in zip(sig_a, sig_b))
    return total / (len(sig_a) * 255)


def store_signature(cache: SignatureCache, base: str, type_dir: str, signature: Optional[bytes]) -> None:
    """Persist the signature for a given base/type pair inside the cache."""
    cache.setdefault(base, {})[type_dir] = signature


def get_signature(cache: SignatureCache, base: str, type_dir: str) -> Optional[bytes]:
    """Fetch a signature from the cache, if present."""
    return cache.get(base, {}).get(type_dir)


def ensure_signature(
    cache: SignatureCache,
    base: str,
    type_dir: str,
    provider: PixmapProvider,
    size: int = SIGNATURE_SIZE,
) -> Optional[bytes]:
    """Retrieve a cached signature or compute one using the provided pixmap supplier."""
    signature = get_signature(cache, base, type_dir)
    if signature is not None:
        return signature
    pixmap = provider(base, type_dir)
    signature = compute_signature(pixmap, size)
    store_signature(cache, base, type_dir, signature)
    return signature
