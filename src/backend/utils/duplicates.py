"""Helpers for duplicate detection based on pixmap perceptual signatures."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, cast

from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt

from common.dict_helpers import get_dict_path
from config import get_config

SignatureCache = Dict[str, Dict[str, Optional[bytes]]]
PixmapProvider = Callable[[str, str], Optional[QPixmap]]


def compute_signature(pixmap: Optional[QPixmap], size: Optional[int] = None) -> Optional[bytes]:
    """Return a compact grayscale signature for the given pixmap."""
    if size is None:
        size = get_config().signature_size
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
    if buffer is None:
        return None
    buffer.setsize(image.width() * image.height())
    # Use memoryview to avoid mypy voidptr error
    return cast(bytes, bytes(memoryview(buffer)))  # type: ignore[arg-type]


def compute_signature_from_path(path: Optional[Path], size: Optional[int] = None) -> Optional[bytes]:
    """Return a signature directly from an image file path."""
    if size is None:
        size = get_config().signature_size
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
    if buffer is None:
        return None
    buffer.setsize(gray.width() * gray.height())
    # Use memoryview to avoid mypy voidptr error
    return cast(bytes, bytes(memoryview(buffer)))  # type: ignore[arg-type]


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
    return get_dict_path(cache, f"{base}.{type_dir}")


def ensure_signature(
    cache: SignatureCache,
    base: str,
    type_dir: str,
    provider: PixmapProvider,
    size: Optional[int] = None,
) -> Optional[bytes]:
    """Retrieve a cached signature or compute one using the provided pixmap supplier."""
    if size is None:
        size = get_config().signature_size
    signature = get_signature(cache, base, type_dir)
    if signature is not None:
        return signature
    pixmap = provider(base, type_dir)
    signature = compute_signature(pixmap, size)
    store_signature(cache, base, type_dir, signature)
    return signature
