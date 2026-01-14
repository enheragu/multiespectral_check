"""Application entrypoint for the multispectral viewer.

Bootstraps Qt styling, instantiates the main window, and starts the event loop.
"""

import os
import sys
import time
import atexit

from PyQt6.QtWidgets import QApplication, QStyleFactory
from common.log_utils import log_debug, log_info, log_perf


_START_TIME = None


def _apply_style(app: QApplication) -> None:
    available = [str(s) for s in QStyleFactory.keys()]
    prefer = os.environ.get("QT_STYLE_OVERRIDE") or os.environ.get("QT_STYLE") or "Fusion"
    if prefer not in available and available:
        prefer = available[0] if "Fusion" not in available else "Fusion"
    style = QStyleFactory.create(prefer)
    if style is not None:
        app.setStyle(style)
    from frontend.widgets import style as ui_style

    ui_style.apply_app_style(app)
    # print(f"Qt styles available: {available}; using: {prefer}")


def main() -> int:
    global _START_TIME
    _START_TIME = time.perf_counter()

    # Register shutdown handler
    def _on_shutdown():
        if _START_TIME is not None:
            elapsed = time.perf_counter() - _START_TIME
            log_info(f"Shutting down after {elapsed:.1f}s uptime", "APP")
            log_info("===== SESSION END =====", "APP")
    atexit.register(_on_shutdown)

    log_info("===== STARTUP BEGIN =====", "APP")
    log_info("Initializing Qt application...", "APP")

    start = time.perf_counter()
    app = QApplication(sys.argv)
    after_app = time.perf_counter()
    log_perf(f"QApplication init {after_app - start:.3f}s")

    log_info("Applying Qt style...", "APP")
    _apply_style(app)
    after_style = time.perf_counter()
    log_perf(f"Style applied {after_style - after_app:.3f}s")

    log_info("Creating ImageViewer...", "APP")
    from frontend.image_viewer import ImageViewer

    viewer_start = time.perf_counter()
    viewer = ImageViewer()
    log_perf(f"ImageViewer init {time.perf_counter() - viewer_start:.3f}s")

    log_info("Showing window...", "APP")
    viewer.showMaximized()

    total_startup = time.perf_counter() - start
    log_info(f"Startup complete in {total_startup:.3f}s", "APP")
    log_info("===== ENTERING EVENT LOOP =====", "APP")
    log_perf(f"Startup total {total_startup:.3f}s")

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())