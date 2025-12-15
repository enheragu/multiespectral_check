import os
import sys

from PyQt6.QtWidgets import QApplication, QStyleFactory

from image_viewer import ImageViewer
from widgets import style as ui_style


def _apply_style(app: QApplication) -> None:
    available = [str(s) for s in QStyleFactory.keys()]
    prefer = os.environ.get("QT_STYLE_OVERRIDE") or os.environ.get("QT_STYLE") or "Fusion"
    if prefer not in available and available:
        prefer = available[0] if "Fusion" not in available else "Fusion"
    style = QStyleFactory.create(prefer)
    if style is not None:
        app.setStyle(style)
    ui_style.apply_app_style(app)
    # print(f"Qt styles available: {available}; using: {prefer}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    _apply_style(app)

    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec())
