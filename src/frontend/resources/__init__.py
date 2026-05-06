"""Frontend asset paths.

Icons live on disk under ``frontend/resources/media/`` and are loaded by
absolute path. We deliberately avoid Qt's qrc/rcc system so:

- the project doesn't need ``pyside6-rcc`` at build time,
- there is no ~26K-line generated file under version control,
- a clean install only needs the dependencies declared in
  ``requirements.txt`` (PyQt6) — no PySide6 import sneaks in.

Use :func:`icon_path` to resolve a media file by name.
"""
from __future__ import annotations

from pathlib import Path

# Directory holding the bundled logo / icon variants.
MEDIA_DIR: Path = Path(__file__).parent / "media"


def icon_path(name: str) -> str:
    """Return the absolute filesystem path to a bundled media file.

    Pass the bare filename (e.g. ``"logo.ico"``). The returned string is
    suitable for ``QIcon(...)`` / ``QPixmap(...)`` constructors.
    """
    return str(MEDIA_DIR / name)


__all__ = ["MEDIA_DIR", "icon_path"]
