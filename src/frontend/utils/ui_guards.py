"""UI validation guards to reduce boilerplate in ImageViewer.

Provides common validation patterns like "load a dataset first" to avoid duplicated QMessageBox calls.
"""
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QMessageBox

if TYPE_CHECKING:  # pragma: no cover
    from frontend.image_viewer import ImageViewer


def require_dataset(viewer: "ImageViewer", title: str = "Action") -> bool:
    """Check if a dataset or collection is loaded. Show warning if not.

    Args:
        viewer: ImageViewer instance
        title: Dialog title for the warning

    Returns:
        True if dataset/collection is loaded, False otherwise
    """
    # Check for dataset (loader) or collection
    if not viewer.session.loader and not viewer.session.collection:
        QMessageBox.information(viewer, title, "Load a dataset first.")
        return False
    return True


def require_images(viewer: "ImageViewer", title: str = "Action") -> bool:
    """Check if dataset has images. Show warning if not.

    Args:
        viewer: ImageViewer instance
        title: Dialog title for the warning

    Returns:
        True if images are available, False otherwise
    """
    if not viewer.session.has_images():
        QMessageBox.information(viewer, title, "Load a dataset first.")
        return False
    return True
