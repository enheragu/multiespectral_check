from collections import OrderedDict, deque
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

from PyQt6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QMenu, QDialog
from PyQt6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QKeySequence,
    QPainter,
    QPixmap,
    QShortcut,
)
from PyQt6.QtCore import Qt, QTimer, QPoint, QThreadPool, QEventLoop

from ui_mainwindow import Ui_MainWindow
from services.calibration_debugger import CalibrationDebugger
from services.calibration_controller import CalibrationController
from services.calibration_mixin import CalibrationWorkflowMixin
from services.calibration_extrinsic_solver import CalibrationExtrinsicSolver
from services.calibration_refiner import CalibrationRefiner
from services.calibration_solver import CALIBRATION_RESULTS_FILENAME, CalibrationSolver
from services.cache_writer import CacheFlushNotifier, CacheFlushRunnable, write_cache_payload
from services.dataset_actions import DatasetActions
from services.signature_controller import SignatureController
from services.dataset_session import DatasetSession
from services.cache_service import CachePersistPayload
from services.progress_tracker import ProgressTracker
from services.filter_modes import (
    FILTER_ACTION_NAMES,
    FILTER_ALL,
    FILTER_CAL_ANY,
    FILTER_CAL_BOTH,
    FILTER_CAL_MISSING,
    FILTER_CAL_PARTIAL,
    FILTER_CAL_SUSPECT,
    FILTER_MESSAGE_LABELS,
    FILTER_STATUS_TITLES,
)
from services.filter_workflow_mixin import FilterWorkflowMixin
from services.lru_index import LRUIndex
from utils.reasons import (
    REASON_CHOICES,
    REASON_KEY_MAP,
    REASON_SHORTCUTS,
    REASON_STYLES,
    REASON_USER,
)
from utils.ui_messages import DELETE_BUTTON_TEXT, RESTORE_ACTION_TEXT, STATUS_NO_IMAGES
from widgets.zoom_pan import ZoomPanView
from widgets.calibration_check_dialog import CalibrationCheckDialog
from widgets.calibration_outliers_dialog import CalibrationOutliersDialog
from widgets.help_dialog import HelpDialog
from widgets.progress_panel import ProgressPanel
from utils.overlays import (
    draw_calibration_overlay,
    draw_overlay_labels,
    draw_reason_overlay,
    paint_rule_of_thirds,
)
from widgets.stats_panel import StatsPanel
from widgets import style


DEFAULT_DATASET_DIR = str(Path.home() / "umh/ros2_ws" / "images_eeha")
CALIBRATION_BORDER_COLOR = QColor("#00ffea")
WARNING_LABEL_COLOR = QColor("#ffb347")
CHESSBOARD_SIZE = (7, 7)
CALIBRATION_PREFETCH_LIMIT = 6
OVERLAY_PREFETCH_RADIUS = 2
OVERLAY_PREFETCH_DELAY_MS = 75
OVERLAY_CACHE_LIMIT = 24
CALIBRATION_TOGGLE_SHORTCUT = "Ctrl+Shift+C"
SIGNATURE_SCAN_MAX_INFLIGHT = 6
SIGNATURE_SCAN_TIMER_INTERVAL_MS = 20
CALIBRATION_DETECT_MAX_WORKERS = 4
PROGRESS_TASK_DETECTION = "calibration-detect"
PROGRESS_TASK_SIGNATURES = "signature-scan"
PROGRESS_TASK_REFINEMENT = "calibration-refine"
PROGRESS_TASK_SOLVER = "calibration-solver"
PROGRESS_TASK_EXTRINSIC = "extrinsic-solver"
PROGRESS_TASK_SAVE = "cache-save"

CANCEL_ACTION_LABELS = {
    PROGRESS_TASK_DETECTION: "Cancelling chessboard detection",
    PROGRESS_TASK_SIGNATURES: "Cancelling duplicate sweep",
    PROGRESS_TASK_REFINEMENT: "Cancelling corner refinement",
    PROGRESS_TASK_SOLVER: "Cancelling calibration solve",
    PROGRESS_TASK_EXTRINSIC: "Cancelling stereo solve",
}


@dataclass
class OverlayCacheEntry:
    signature: Tuple[Any, ...]
    pixmap: QPixmap


class ImageViewer(FilterWorkflowMixin, CalibrationWorkflowMixin, QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        if hasattr(self.ui, "centralwidget"):
            self.ui.centralwidget.setStyleSheet(f"background: {style.APP_BG};")
        # Apply scoped styles so main navigation buttons render the same gradient as dialogs
        for btn in (
            getattr(self.ui, "btn_prev", None),
            getattr(self.ui, "btn_next", None),
            getattr(self.ui, "btn_delete_marked", None),
        ):
            if btn:
                btn.setStyleSheet(style.scoped_button_style(btn.objectName()))
        self.progress_panel: Optional[ProgressPanel] = None
        self._setup_progress_panel()

        self.session = DatasetSession()
        self.state = self.session.state
        self.dataset_actions = DatasetActions(self, DEFAULT_DATASET_DIR)

        preferences = self.session.cache_service.get_preferences()
        self.show_grid = bool(preferences.get("show_grid", True))
        self.view_rectified = bool(preferences.get("view_rectified", False))
        pref_mode = preferences.get("filter_mode")
        legacy_filter = preferences.get("filter_calibration_only", False)
        if isinstance(pref_mode, str) and pref_mode in FILTER_ACTION_NAMES:
            self.filter_mode = pref_mode
        elif legacy_filter:
            self.filter_mode = FILTER_CAL_ANY
        else:
            self.filter_mode = FILTER_ALL

        self.current_index = 0
        self._shortcuts: List[QShortcut] = []
        self._filter_actions: Dict[str, Any] = {}
        self._filter_group: Optional[QActionGroup] = None
        self.progress_tracker = ProgressTracker(self._handle_progress_snapshot)
        self._cancel_handlers: "OrderedDict[str, Callable[[], None]]" = OrderedDict()
        self._cancel_inflight: Set[str] = set()
        self._refine_total = 0
        self._refine_progress = 0
        self._reset_queue_progress_state()

        self.thread_pool = QThreadPool.globalInstance()
        self.calibration_thread_pool = QThreadPool(self)
        self.calibration_thread_pool.setMaxThreadCount(max(1, CALIBRATION_DETECT_MAX_WORKERS))
        self.signature_controller = SignatureController(self.thread_pool)
        self.cache_timer = QTimer(self)
        self.cache_timer.setInterval(2000)
        self.cache_timer.setSingleShot(True)
        self.cache_timer.timeout.connect(self._flush_cache)
        self._cache_flush_notifier = CacheFlushNotifier(self)
        self._cache_flush_notifier.finished.connect(self._handle_cache_flush_finished)
        self._cache_flush_inflight = False
        self._cache_flush_pending: Optional[CachePersistPayload] = None
        self.overlay_prefetch_timer = QTimer(self)
        self.overlay_prefetch_timer.setSingleShot(True)
        self.overlay_prefetch_timer.timeout.connect(self._prefetch_neighbor_overlays)
        self._calibration_mark_timer = QTimer(self)
        self._calibration_mark_timer.setSingleShot(True)
        self._calibration_mark_timer.timeout.connect(self._flush_pending_calibration_marks)

        self.lwir_view = ZoomPanView(self)
        self.vis_view = ZoomPanView(self)
        self.stats_panel = StatsPanel()

        self._setup_calibration_outlier_action()

        self._overlay_cache: Dict[str, Dict[str, OverlayCacheEntry]] = {}
        self._overlay_cache_order = LRUIndex(OVERLAY_CACHE_LIMIT)
        self._overlay_prefetch_queue: Deque[str] = deque()
        self._signature_pending: Set[str] = set()
        self._signature_completed: Set[str] = set()
        self._signature_scan_queue: Deque[int] = deque()
        self._signature_scan_force = False
        self._signature_scan_timer = QTimer(self)
        self._signature_scan_timer.setSingleShot(True)
        self._signature_scan_timer.timeout.connect(self._drain_signature_scan_queue)
        self._pending_calibration_marks: Set[str] = set()
        self._pending_calibration_forces: Set[str] = set()

        self.calibration_debugger = CalibrationDebugger(self.session, CHESSBOARD_SIZE)
        self.calibration_controller = CalibrationController(
            self.session,
            CHESSBOARD_SIZE,
            self.calibration_thread_pool,
            max_workers=CALIBRATION_DETECT_MAX_WORKERS,
        )
        self.calibration_controller.calibrationReady.connect(self._handle_calibration_ready)
        self.calibration_controller.calibrationFailed.connect(self._handle_calibration_failed)
        self.calibration_controller.activityChanged.connect(self._handle_calibration_activity_changed)
        self.calibration_refiner = CalibrationRefiner(
            self.session,
            CHESSBOARD_SIZE,
            self.thread_pool,
        )
        self.calibration_refiner.refinementReady.connect(self._handle_refinement_ready)
        self.calibration_refiner.refinementFailed.connect(self._handle_refinement_failed)
        self.calibration_refiner.batchFinished.connect(self._handle_refinement_batch_finished)
        self.calibration_solver = CalibrationSolver(
            self.session,
            CHESSBOARD_SIZE,
            self.thread_pool,
        )
        self.calibration_solver.calibrationSolved.connect(self._handle_calibration_solved)
        self.calibration_solver.calibrationFailed.connect(self._handle_calibration_solver_failed)
        self.calibration_extrinsic_solver = CalibrationExtrinsicSolver(
            self.session,
            CHESSBOARD_SIZE,
            self.thread_pool,
        )
        self.calibration_extrinsic_solver.extrinsicSolved.connect(self._handle_extrinsic_solved)
        self.calibration_extrinsic_solver.extrinsicFailed.connect(self._handle_extrinsic_failed)
        self.signature_controller.signatureReady.connect(self._handle_signature_ready)
        self.signature_controller.signatureFailed.connect(self._handle_signature_failed)
        self.signature_controller.activityChanged.connect(self._handle_signature_activity_changed)

        self._setup_image_views()
        self._setup_stats_panel()
        self._setup_filter_actions()
        self._sync_action_states()
        self.connect_signals()
        self._register_shortcuts()
        self._show_empty_state()
        self._auto_load_last_dataset()
        self._update_cancel_button()
        self._outlier_dialog = None

    def _sync_action_states(self) -> None:
        toggles = [
            (getattr(self.ui, "action_toggle_grid", None), self.show_grid),
            (getattr(self.ui, "action_toggle_rectified", None), self.view_rectified),
        ]
        for action, state in toggles:
            if not action:
                continue
            action.blockSignals(True)
            action.setChecked(state)
            action.blockSignals(False)
        self._update_filter_checks()

    def _register_shortcuts(self) -> None:
        combos = [
            (Qt.Key.Key_Left, self.prev_image),
            (Qt.Key.Key_Right, self.next_image),
            (Qt.Key.Key_Space, self.next_image),
            (Qt.Key.Key_Delete, self.toggle_mark_current),
            (Qt.Key.Key_C, self.toggle_calibration_current),
        ]
        for key, handler in combos:
            shortcut = QShortcut(key, self)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(handler)
            self._shortcuts.append(shortcut)
        calibration_sequence = QKeySequence(CALIBRATION_TOGGLE_SHORTCUT)
        calibration_shortcut = QShortcut(calibration_sequence, self)
        calibration_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        calibration_shortcut.activated.connect(self.toggle_calibration_current)
        self._shortcuts.append(calibration_shortcut)
        for reason, sequence in REASON_SHORTCUTS.items():
            if not sequence or sequence.lower() == "del":
                continue
            reason_shortcut = QShortcut(QKeySequence(sequence), self)
            reason_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            reason_shortcut.activated.connect(lambda r=reason: self._handle_reason_shortcut(r))
            self._shortcuts.append(reason_shortcut)

    def connect_signals(self) -> None:
        if hasattr(self.ui, "btn_prev"):
            self.ui.btn_prev.clicked.connect(self.prev_image)
        if hasattr(self.ui, "btn_next"):
            self.ui.btn_next.clicked.connect(self.next_image)
        if hasattr(self.ui, "btn_delete_marked"):
            self.ui.btn_delete_marked.clicked.connect(self.delete_marked_images)
        if hasattr(self.ui, "action_load_dataset"):
            self.ui.action_load_dataset.triggered.connect(self.select_directory)
        if hasattr(self.ui, "action_save_status"):
            self.ui.action_save_status.triggered.connect(self._handle_save_status_action)
        if hasattr(self.ui, "action_run_duplicate_scan"):
            self.ui.action_run_duplicate_scan.triggered.connect(self._handle_run_duplicate_scan_action)
        if hasattr(self.ui, "action_delete_selected"):
            self.ui.action_delete_selected.triggered.connect(self.delete_marked_images)
        if hasattr(self.ui, "action_restore_images"):
            self.ui.action_restore_images.triggered.connect(self.restore_images)
        if hasattr(self.ui, "action_import_calibration"):
            self.ui.action_import_calibration.triggered.connect(self.import_calibration_data)
        if hasattr(self.ui, "action_clear_empty_datasets"):
            self.ui.action_clear_empty_datasets.triggered.connect(self.clear_empty_datasets)
        if hasattr(self.ui, "action_toggle_rectified"):
            self.ui.action_toggle_rectified.toggled.connect(self._handle_rectified_toggle)
        if hasattr(self.ui, "action_toggle_grid"):
            self.ui.action_toggle_grid.toggled.connect(self._handle_grid_toggle)
        if hasattr(self.ui, "action_calibration_debug"):
            self.ui.action_calibration_debug.triggered.connect(self.export_calibration_debug)
        if hasattr(self.ui, "action_run_calibration"):
            self.ui.action_run_calibration.triggered.connect(self._handle_run_calibration_action)
        if hasattr(self.ui, "action_calibration_refine"):
            self.ui.action_calibration_refine.triggered.connect(self._handle_refine_calibration_action)
        if hasattr(self.ui, "action_calibration_compute"):
            self.ui.action_calibration_compute.triggered.connect(self._handle_compute_calibration_action)
        if hasattr(self.ui, "action_calibration_extrinsic"):
            self.ui.action_calibration_extrinsic.triggered.connect(self._handle_compute_extrinsic_action)
        if hasattr(self.ui, "action_calibration_check"):
            self.ui.action_calibration_check.triggered.connect(self._handle_show_calibration_dialog)
        if hasattr(self.ui, "action_calibration_outliers"):
            self.ui.action_calibration_outliers.triggered.connect(self._open_calibration_outlier_dialog)
        if hasattr(self.ui, "action_show_help"):
            self.ui.action_show_help.triggered.connect(self.show_help_dialog)
        if hasattr(self.ui, "action_exit"):
            self.ui.action_exit.triggered.connect(self.close)

    def _setup_image_views(self) -> None:
        old_lwir = getattr(self.ui, "label_lwir", None)
        old_vis = getattr(self.ui, "label_vis", None)
        layout = getattr(self.ui, "images_layout", None)
        if not all((old_lwir, old_vis, layout)):
            return
        self.lwir_view.setMinimumSize(old_lwir.minimumSize())
        self.vis_view.setMinimumSize(old_vis.minimumSize())
        self.lwir_view.setSizePolicy(old_lwir.sizePolicy())
        self.vis_view.setSizePolicy(old_vis.sizePolicy())
        self.lwir_view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.vis_view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout.replaceWidget(old_lwir, self.lwir_view)
        layout.replaceWidget(old_vis, self.vis_view)
        old_lwir.deleteLater()
        old_vis.deleteLater()
        self.lwir_view.transformChanged.connect(self.vis_view.apply_external_transform)
        self.vis_view.transformChanged.connect(self.lwir_view.apply_external_transform)
        self.lwir_view.contextRequested.connect(self._show_image_context_menu)
        self.vis_view.contextRequested.connect(self._show_image_context_menu)

    def _setup_progress_panel(self) -> None:
        placeholder = getattr(self.ui, "progress_placeholder", None)
        layout = getattr(self.ui, "btn_layout", None)
        panel = ProgressPanel(self)
        panel.cancelRequested.connect(self._handle_cancel_action)
        self.progress_panel = panel
        if layout is not None:
            if placeholder is not None:
                index = layout.indexOf(placeholder)
                if index >= 0:
                    layout.insertWidget(index, panel)
                    layout.removeWidget(placeholder)
                else:
                    layout.insertWidget(0, panel)
                placeholder.deleteLater()
            else:
                layout.insertWidget(0, panel)
        panel.hide()

    def _setup_stats_panel(self) -> None:
        placeholder = getattr(self.ui, "stats_placeholder", None)
        layout = getattr(self.ui, "verticalLayout", None)
        if placeholder and layout:
            layout.replaceWidget(placeholder, self.stats_panel)
            placeholder.deleteLater()
        elif layout:
            layout.addWidget(self.stats_panel)

    def _setup_calibration_outlier_action(self) -> None:
        action = getattr(self.ui, "action_calibration_outliers", None)
        if action is None:
            action = QAction("Check calibration outliers", self)
            setattr(self.ui, "action_calibration_outliers", action)
        menu = getattr(self.ui, "menu_calibration", None)
        if isinstance(menu, QMenu):
            if action in menu.actions():
                menu.removeAction(action)
            target = getattr(self.ui, "action_calibration_check", None)
            if target and target in menu.actions():
                menu.insertAction(target, action)
            else:
                menu.addAction(action)
        else:
            if action not in self.menuBar().actions():
                self.menuBar().addAction(action)

    def _setup_theme_toggle(self) -> None:
        menu = getattr(self.ui, "menu_view", None)
        action = QAction("Dark mode (Fusion palette)", self)
        action.setCheckable(True)
        action.toggled.connect(self._handle_toggle_dark_mode)
        if isinstance(menu, QMenu):
            menu.addSeparator()
            menu.addAction(action)
        self.action_dark_mode = action

    def _auto_load_last_dataset(self) -> None:
        last_dataset = self.session.last_dataset()
        if not last_dataset:
            return
        dataset_path = Path(last_dataset)
        if dataset_path.exists():
            self._load_dataset(dataset_path)


    def keyPressEvent(self, event):
        """Keyboard shortcuts for navigation and tagging."""
        if not self.session.has_images():
            super().keyPressEvent(event)
            return

        handled = False
        if event.key() == Qt.Key.Key_Left:
            self.prev_image()
            handled = True
        elif event.key() == Qt.Key.Key_Right or event.key() == Qt.Key.Key_Space:
            self.next_image()
            handled = True
        elif event.key() == Qt.Key.Key_Delete:
            self.toggle_mark_current()
            handled = True
        elif event.key() == Qt.Key.Key_C and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            self.toggle_calibration_current()
            handled = True
        elif event.key() in REASON_KEY_MAP and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            self._handle_reason_shortcut(REASON_KEY_MAP[event.key()])
            handled = True

        if handled:
            event.accept()
            return
        super().keyPressEvent(event)

    def select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select dataset directory", DEFAULT_DATASET_DIR)

        if dir_path:
            self._load_dataset(Path(dir_path))

    def _reset_runtime_state(self) -> None:
        self.current_index = 0
        self.session.reset_state()
        self.calibration_controller.cancel_all()
        self.signature_controller.cancel_all()
        self.calibration_refiner.cancel()
        self.calibration_solver.cancel()
        self.calibration_extrinsic_solver.cancel()
        self._bump_signature_epoch()
        self._reset_calibration_jobs()
        self._cancel_signature_scan_jobs()
        self._clear_pending_calibration_marks()
        if self.overlay_prefetch_timer.isActive():
            self.overlay_prefetch_timer.stop()
        self.invalidate_overlay_cache()
        self.progress_tracker.clear()
        self._reset_queue_progress_state()
        self._cancel_handlers.clear()
        self._cancel_inflight.clear()
        self._update_cancel_button()

    def _prime_signature_scan(self, *, force: bool = False) -> None:
        if not self.session.loader:
            QMessageBox.information(self, "Duplicates", "Load a dataset before scanning for duplicates.")
            return
        total = self.session.total_pairs()
        if total <= 0:
            self.statusBar().showMessage("No images available for duplicate scanning.", 4000)
            return
        if force:
            self._bump_signature_epoch()
            self._cancel_signature_scan_jobs()
        if force:
            targets = list(range(total))
        else:
            targets = [
                idx
                for idx, base in enumerate(self.session.loader.image_bases)
                if base and not self._has_cached_signatures(base)
            ]
        self._signature_scan_target = len(targets)
        self._signature_progress_started = False
        if not targets:
            self._update_signature_progress()
            if force:
                self.statusBar().showMessage("Duplicate signatures already cached.", 4000)
            return
        self._signature_scan_queue.extend(targets)
        self._signature_scan_force = force
        self._update_signature_progress()
        self._signature_scan_timer.start(0)
        label = "Re-running duplicate sweep" if force else "Buscando duplicados"
        self.statusBar().showMessage(
            f"{label} en {total} imagen(es)…",
            4000,
        )

    def _drain_signature_scan_queue(self) -> None:
        if not self._signature_scan_queue:
            self._update_signature_progress()
            return
        inflight = len(self._signature_pending)
        available = max(0, SIGNATURE_SCAN_MAX_INFLIGHT - inflight)
        scheduled = 0
        while available > 0 and self._signature_scan_queue:
            index = self._signature_scan_queue.popleft()
            if self._schedule_signature_job(index, force=self._signature_scan_force):
                available -= 1
                scheduled += 1
        self._update_signature_progress()
        if self._signature_scan_queue and scheduled == 0:
            self._signature_scan_timer.start(SIGNATURE_SCAN_TIMER_INTERVAL_MS)

    def _handle_run_duplicate_scan_action(self) -> None:
        self._prime_signature_scan(force=True)

    def _load_dataset(self, dir_path: Path) -> None:
        self.current_index = 0
        if not self.session.load(dir_path):
            self._reset_runtime_state()
            self._show_empty_state()
            self.lwir_view.set_placeholder("No images found in lwir/ or visible/")
            self.vis_view.set_placeholder("No images found in lwir/ or visible/")
            return
        self._bump_signature_epoch()
        self._reset_calibration_jobs()
        self.progress_tracker.clear()
        self._reset_queue_progress_state()
        self._prime_calibration_jobs(limit=CALIBRATION_PREFETCH_LIMIT)
        self._prime_signature_scan()
        self.invalidate_overlay_cache()
        self.load_current()
        self.ui.btn_prev.setEnabled(True)
        self.ui.btn_next.setEnabled(True)
        self.setWindowTitle(f"Image Viewer - {self.session.total_pairs()} images")
        self._update_delete_button()
        self._update_restore_menu()
        self._update_stats_panel()
        self._mark_cache_dirty()
        if self.filter_mode != FILTER_ALL:
            self._reconcile_filter_state(show_warning=False)

    def _current_base(self) -> Optional[str]:
        return self.session.get_base(self.current_index)

    def load_current(self):
        base = self._current_base()
        if not base:
            return
        self.update_status(base)
        self._ensure_calibration_analysis(base)
        self.load_image_pair(base)
        self._update_stats_panel()
        self._schedule_signature_job(self.current_index)

    def update_status(self, base: str):
        total = self.session.total_pairs()
        if not total:
            self.setWindowTitle("Image Viewer")
            return
        title = f"Image Viewer - {base} ({self.current_index+1}/{total})"
        filter_index, filter_total = self._filtered_position()
        if self.filter_mode != FILTER_ALL and filter_total:
            label = FILTER_STATUS_TITLES.get(self.filter_mode, "Filter")
            title += f" · {label} {filter_index}/{filter_total}"
        self.setWindowTitle(title)

    def load_image_pair(self, base: str):
        display_lwir, display_vis = self._render_overlayed_pair(base)
        if display_lwir:
            self.lwir_view.set_pixmap(display_lwir)
        else:
            self.lwir_view.set_placeholder("Missing LWIR image")
        if display_vis:
            self.vis_view.set_pixmap(display_vis)
        else:
            self.vis_view.set_placeholder("Missing visible image")
        self._update_metadata_panel(base, "lwir", self.ui.text_metadata_lwir)
        self._update_metadata_panel(base, "visible", self.ui.text_metadata_vis)
        if self.overlay_prefetch_timer.isActive():
            self.overlay_prefetch_timer.stop()
        self._prepare_overlay_prefetch_queue(base)
        if self._overlay_prefetch_queue:
            self.overlay_prefetch_timer.start(OVERLAY_PREFETCH_DELAY_MS)

    def _render_overlayed_pair(self, base: str) -> Tuple[Optional[QPixmap], Optional[QPixmap]]:
        reason = self.state.marked_for_delete.get(base)
        calibration_flag = base in self.state.calibration_marked
        calib_results = self.state.calibration_results.get(base, {})
        calib_corners = self.state.calibration_corners.get(base, {})
        warning_bucket = self.state.calibration_warnings.get(base, {})
        lwir_warning = warning_bucket.get("lwir") if isinstance(warning_bucket, dict) else None
        vis_warning = warning_bucket.get("visible") if isinstance(warning_bucket, dict) else None
        lwir_signature = self._build_overlay_signature(
            reason,
            calibration_flag,
            calib_results.get("lwir"),
            calib_corners.get("lwir"),
            lwir_warning,
        )
        vis_signature = self._build_overlay_signature(
            reason,
            calibration_flag,
            calib_results.get("visible"),
            calib_corners.get("visible"),
            vis_warning,
        )
        cached_lwir = self._get_cached_overlay(base, "lwir", lwir_signature)
        cached_vis = self._get_cached_overlay(base, "visible", vis_signature)
        if cached_lwir is not None and cached_vis is not None:
            return cached_lwir, cached_vis
        display_lwir, display_vis = self.session.prepare_display_pair(base, self.view_rectified)
        if cached_lwir is None:
            display_lwir = self._draw_overlays(
                base,
                "lwir",
                display_lwir,
                reason,
                calibration_flag,
                calib_results.get("lwir"),
                calib_corners.get("lwir"),
                signature=lwir_signature,
            )
        else:
            display_lwir = cached_lwir
        if cached_vis is None:
            display_vis = self._draw_overlays(
                base,
                "visible",
                display_vis,
                reason,
                calibration_flag,
                calib_results.get("visible"),
                calib_corners.get("visible"),
                signature=vis_signature,
            )
        else:
            display_vis = cached_vis
        return display_lwir, display_vis

    def _navigate(self, direction: int) -> bool:
        total = self.session.total_pairs()
        if total <= 0:
            return False
        if self.filter_mode == FILTER_ALL:
            self.current_index = (self.current_index + direction) % total
            self.load_current()
            return True
        start = self.current_index
        for _ in range(total):
            self.current_index = (self.current_index + direction) % total
            base = self._current_base()
            if base and self._filter_accepts(base):
                self.load_current()
                return True
        self.current_index = start
        return False

    def prev_image(self):
        if not self.session.has_images():
            return
        if not self._navigate(-1) and self.filter_mode != FILTER_ALL:
            noun = FILTER_MESSAGE_LABELS.get(self.filter_mode, "filtered images")
            self.statusBar().showMessage(f"No previous {noun}.", 3000)

    def next_image(self):
        if not self.session.has_images():
            return
        if not self._navigate(1) and self.filter_mode != FILTER_ALL:
            noun = FILTER_MESSAGE_LABELS.get(self.filter_mode, "filtered images")
            self.statusBar().showMessage(f"No more {noun}.", 3000)

    def _draw_overlays(
        self,
        base: str,
        channel: str,
        pixmap: Optional[QPixmap],
        reason: Optional[str],
        calibration: bool,
        calibration_detected: Optional[bool],
        corner_points: Optional[List[Tuple[float, float]]] = None,
        signature: Optional[Tuple[Any, ...]] = None,
    ) -> Optional[QPixmap]:
        if not pixmap or pixmap.isNull():
            return None
        warning_bucket = self.state.calibration_warnings.get(base, {})
        warning_text = warning_bucket.get(channel) if isinstance(warning_bucket, dict) else None
        if signature is None:
            signature = self._build_overlay_signature(
                reason,
                calibration,
                calibration_detected,
                corner_points,
                warning_text,
            )
        bucket = self._overlay_cache.setdefault(base, {})
        entry = bucket.get(channel)
        if entry and entry.signature == signature and entry.pixmap and not entry.pixmap.isNull():
            return entry.pixmap
        base_pix = pixmap.copy()
        base_w, base_h = base_pix.width(), base_pix.height()
        overlay_pen_width = max(2, int(max(base_w, base_h) / 200))
        canvas = base_pix
        painter = QPainter(canvas)
        if self.show_grid:
            paint_rule_of_thirds(painter, base_w, base_h)
        label_entries = []
        if reason:
            style = REASON_STYLES.get(reason, {"color": QColor("red"), "text": reason})
            draw_reason_overlay(painter, base_pix, style["color"], style["text"], overlay_pen_width)
            label_entries.append((style["text"], style["color"]))
        if calibration:
            draw_calibration_overlay(painter, base_pix, CALIBRATION_BORDER_COLOR, max(3, overlay_pen_width + 1))
            if warning_text:
                status = "Chessboard discarded"
            elif calibration_detected is not None:
                status = "Chessboard detected" if calibration_detected else "Chessboard missing"
            else:
                status = "Calibration candidate"
            label_entries.append((status, CALIBRATION_BORDER_COLOR))
        if warning_text:
            trimmed = warning_text if len(warning_text) <= 60 else f"{warning_text[:57]}…"
            label_entries.append((f"Suspect corners: {trimmed}", WARNING_LABEL_COLOR))
        if label_entries:
            draw_overlay_labels(painter, canvas.width(), canvas.height(), label_entries)
        if corner_points:
            dot_color = WARNING_LABEL_COLOR if warning_text else CALIBRATION_BORDER_COLOR
            painter.setPen(dot_color)
            painter.setBrush(dot_color)
            radius = max(3, overlay_pen_width)
            for u, v in corner_points:
                x = int(u * base_w)
                y = int(v * base_h)
                painter.drawEllipse(QPoint(x, y), radius, radius)
        painter.end()
        bucket[channel] = OverlayCacheEntry(signature, canvas)
        self._track_overlay_cache_use(base)
        return canvas

    def _prefetch_neighbor_overlays(self) -> None:
        if not self._overlay_prefetch_queue:
            return
        base = self._overlay_prefetch_queue.popleft()
        self._ensure_overlay_cached(base)
        if self._overlay_prefetch_queue:
            self.overlay_prefetch_timer.start(OVERLAY_PREFETCH_DELAY_MS)

    def _ensure_overlay_cached(self, base: str) -> None:
        if not self.session.loader:
            return
        self._render_overlayed_pair(base)

    def _prepare_overlay_prefetch_queue(self, current_base: Optional[str]) -> None:
        self._overlay_prefetch_queue.clear()
        if not current_base or not self.session.loader or not self.state.calibration_marked:
            return
        total = self.session.total_pairs()
        if total <= 1:
            return
        seen = {current_base}
        for delta in range(1, OVERLAY_PREFETCH_RADIUS + 1):
            for offset in (-delta, delta):
                idx = self.current_index + offset
                if idx < 0 or idx >= total:
                    continue
                base = self.session.get_base(idx)
                if (
                    not base
                    or base in seen
                    or base not in self.state.calibration_marked
                    or self._overlay_is_cached(base)
                ):
                    continue
                self._overlay_prefetch_queue.append(base)
                seen.add(base)

    def _overlay_is_cached(self, base: str) -> bool:
        bucket = self._overlay_cache.get(base)
        if not bucket:
            return False
        for channel in ("lwir", "visible"):
            entry = bucket.get(channel)
            if not entry or not entry.pixmap or entry.pixmap.isNull():
                return False
        return True

    def _track_overlay_cache_use(self, base: Optional[str]) -> None:
        if not base:
            return
        evicted = self._overlay_cache_order.touch(base)
        for key in evicted:
            self._overlay_cache.pop(key, None)
        self._enforce_overlay_cache_limit()

    def _remove_overlay_cache_order(self, base: Optional[str]) -> None:
        if not base:
            return
        self._overlay_cache_order.remove(base)

    def _enforce_overlay_cache_limit(self) -> None:
        while len(self._overlay_cache) > OVERLAY_CACHE_LIMIT:
            evicted = self._overlay_cache_order.pop_oldest()
            if evicted is None:
                break
            self._overlay_cache.pop(evicted, None)

    def _update_metadata_panel(self, base: str, type_dir: str, widget):
        widget.clear()
        widget.setPlainText(self.session.get_metadata_text(base, type_dir))

    def _show_image_context_menu(self, global_pos: QPoint) -> None:
        if not self.session.has_images():
            return
        base = self._current_base()
        if not base:
            return
        menu = QMenu(self)
        nav_prev = menu.addAction("Previous image")
        nav_prev.setShortcut(QKeySequence("Left Arrow"))
        nav_prev.setShortcutVisibleInContextMenu(True)
        nav_next = menu.addAction("Next image")
        nav_next.setShortcut(QKeySequence("Right Arrow"))
        nav_next.setShortcutVisibleInContextMenu(True)
        menu.addSeparator()
        current_reason = self.state.marked_for_delete.get(base)
        reason_actions = []
        for reason, label in REASON_CHOICES:
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(current_reason == reason)
            shortcut = REASON_SHORTCUTS.get(reason)
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
                action.setShortcutVisibleInContextMenu(True)
            reason_actions.append((action, reason))
        menu.addSeparator()
        calibration_action = menu.addAction("Mark as calibration candidate")
        calibration_action.setShortcut(QKeySequence(CALIBRATION_TOGGLE_SHORTCUT))
        calibration_action.setShortcutVisibleInContextMenu(True)
        calibration_action.setCheckable(True)
        calibration_action.setChecked(base in self.state.calibration_marked)
        reanalyze_action = None
        if base in self.state.calibration_marked:
            reanalyze_action = menu.addAction("Re-run calibration detection")
            reanalyze_action.setShortcut(QKeySequence("Ctrl+Shift+R"))
            reanalyze_action.setShortcutVisibleInContextMenu(True)
        chosen = menu.exec(global_pos)
        if chosen is None:
            return
        if chosen is nav_prev:
            self.prev_image()
            return
        if chosen is nav_next:
            self.next_image()
            return
        for action, reason in reason_actions:
            if chosen is action:
                if action.isChecked():
                    self._apply_mark_reason(base, reason)
                else:
                    self._apply_mark_reason(base, None)
                return
        if chosen is calibration_action:
            if self.dataset_actions.set_calibration_mark(base, calibration_action.isChecked()):
                self.invalidate_overlay_cache(base)
                self.load_image_pair(base)
                self._update_stats_panel()
                self._mark_cache_dirty()
                self._reconcile_filter_state(show_warning=True)
            return
        if reanalyze_action and chosen is reanalyze_action:
            if self._schedule_calibration_job(base, force=True, priority=True):
                self.statusBar().showMessage(f"Re-running calibration analysis for {base}", 4000)
            return

    def _apply_mark_reason(self, base: str, reason: Optional[str]) -> None:
        if not self.state.set_mark_reason(base, reason, REASON_USER):
            return
        self.invalidate_overlay_cache(base)
        self._update_delete_button()
        self.load_image_pair(base)
        self._update_stats_panel()
        self._mark_cache_dirty()

    def _handle_reason_shortcut(self, reason: str) -> None:
        base = self._current_base()
        if not base:
            return
        current = self.state.marked_for_delete.get(base)
        if current == reason:
            self._apply_mark_reason(base, None)
        else:
            self._apply_mark_reason(base, reason)

    def _has_calibration_data(self) -> bool:
        return any(data for data in self.state.calibration_matrices.values())

    def _persist_preferences(self, **kwargs: Any) -> None:
        self.session.cache_service.set_preferences(**kwargs)

    def _handle_save_status_action(self) -> None:
        self.progress_tracker.set_busy(PROGRESS_TASK_SAVE, "Saving dataset state…")
        try:
            self._flush_cache(wait=True)
            self.statusBar().showMessage("Dataset status saved.", 3000)
        finally:
            self.progress_tracker.finish(PROGRESS_TASK_SAVE)

    def _mark_cache_dirty(self) -> None:
        self.session.mark_cache_dirty()
        if not self.cache_timer.isActive():
            self.cache_timer.start()

    def _flush_cache(self, *, wait: bool = False) -> None:
        if self.cache_timer.isActive():
            self.cache_timer.stop()
        if wait:
            self._wait_for_cache_flush()
            payload = self.session.snapshot_cache_payload()
            if payload:
                write_cache_payload(payload)
            return
        payload = self.session.snapshot_cache_payload()
        if payload:
            if self._cache_flush_inflight:
                self._cache_flush_pending = payload
            else:
                self._dispatch_cache_flush(payload)

    def _dispatch_cache_flush(self, payload: CachePersistPayload) -> None:
        self._cache_flush_inflight = True
        runnable = CacheFlushRunnable(payload, self._cache_flush_notifier)
        self.thread_pool.start(runnable)

    def _handle_cache_flush_finished(self) -> None:
        if self._cache_flush_pending is not None:
            pending = self._cache_flush_pending
            self._cache_flush_pending = None
            self._dispatch_cache_flush(pending)
            return
        self._cache_flush_inflight = False

    def _wait_for_cache_flush(self) -> None:
        if not self._cache_flush_inflight and self._cache_flush_pending is None:
            return
        loop = QEventLoop(self)

        def _check_state() -> None:
            if not self._cache_flush_inflight and self._cache_flush_pending is None:
                loop.quit()

        self._cache_flush_notifier.finished.connect(_check_state)
        _check_state()
        if self._cache_flush_inflight or self._cache_flush_pending is not None:
            loop.exec()
        self._cache_flush_notifier.finished.disconnect(_check_state)

    def invalidate_overlay_cache(self, base: Optional[str] = None) -> None:
        if base is None:
            self._overlay_cache.clear()
            self._overlay_cache_order.clear()
            return
        self._overlay_cache.pop(base, None)
        self._remove_overlay_cache_order(base)

    def _corner_signature(self, corner_points: Optional[List[Tuple[float, float]]]) -> Optional[Tuple[Tuple[float, float], ...]]:
        if not corner_points:
            return None
        return tuple((round(u, 4), round(v, 4)) for u, v in corner_points)

    def _build_overlay_signature(
        self,
        reason: Optional[str],
        calibration: bool,
        calibration_detected: Optional[bool],
        corner_points: Optional[List[Tuple[float, float]]],
        warning_text: Optional[str],
    ) -> Tuple[Any, ...]:
        return (
            self.view_rectified,
            self.show_grid,
            reason or "",
            calibration,
            calibration_detected,
            self._corner_signature(corner_points),
            (warning_text or "")[:64],
        )

    def _reset_queue_progress_state(self) -> None:
        self._calibration_activity_total = 0
        self._calibration_activity_done = 0
        self._calibration_activity_last = 0
        self._signature_activity_total = 0
        self._signature_activity_done = 0
        self._signature_activity_last = 0
        self._signature_scan_target = 0
        self._signature_progress_started = False

    def _update_queue_progress(
        self,
        *,
        pending: int,
        label: str,
        task_id: str,
        total_attr: str,
        done_attr: str,
        last_attr: str,
        cancel_handler: Optional[Callable[[], None]] = None,
    ) -> None:
        total = getattr(self, total_attr)
        done = getattr(self, done_attr)
        last = getattr(self, last_attr)
        if pending <= 0:
            setattr(self, total_attr, 0)
            setattr(self, done_attr, 0)
            setattr(self, last_attr, 0)
            self._unregister_cancel_handler(task_id)
            self.progress_tracker.finish(task_id)
            return
        if last <= 0 or total <= 0:
            setattr(self, total_attr, pending)
            setattr(self, done_attr, 0)
            setattr(self, last_attr, pending)
            self.progress_tracker.start(task_id, label, pending)
            if cancel_handler:
                self._register_cancel_handler(task_id, cancel_handler)
            return
        remaining = max(0, total - done)
        if pending > remaining:
            total = done + pending
            setattr(self, total_attr, total)
        done = max(0, min(total - pending, total))
        setattr(self, done_attr, done)
        setattr(self, last_attr, pending)
        self.progress_tracker.update(task_id, done, total)

    def _handle_progress_snapshot(self, snapshot) -> None:
        if not self.progress_panel:
            return
        self.progress_panel.set_snapshot(snapshot)

    def _register_cancel_handler(self, task_id: str, handler: Callable[[], None]) -> None:
        self._cancel_handlers.pop(task_id, None)
        self._cancel_handlers[task_id] = handler
        self._cancel_inflight.discard(task_id)
        self._update_cancel_button()

    def _unregister_cancel_handler(self, task_id: str) -> None:
        self._cancel_handlers.pop(task_id, None)
        self._cancel_inflight.discard(task_id)
        self._update_cancel_button()

    def _active_cancellable_task(self) -> Optional[str]:
        if not self._cancel_handlers:
            return None
        for task_id in reversed(list(self._cancel_handlers.keys())):
            if task_id not in self._cancel_inflight:
                return task_id
        return None

    def _update_cancel_button(self) -> None:
        if not self.progress_panel:
            return
        task_id = self._active_cancellable_task()
        if not task_id:
            self.progress_panel.set_cancel_state(False)
            return
        tooltip = CANCEL_ACTION_LABELS.get(task_id, "Cancel current action")
        enabled = task_id not in self._cancel_inflight
        self.progress_panel.set_cancel_state(enabled, tooltip)

    def _handle_cancel_action(self) -> None:
        task_id = self._active_cancellable_task()
        if not task_id:
            return
        if task_id in self._cancel_inflight:
            self.statusBar().showMessage("Cancellation already requested…", 2000)
            return
        handler = self._cancel_handlers.get(task_id)
        if not handler:
            return
        self._cancel_inflight.add(task_id)
        self._update_cancel_button()
        handler()
        self.progress_tracker.finish(task_id)
        self._unregister_cancel_handler(task_id)
        label = CANCEL_ACTION_LABELS.get(task_id, "Cancelling action…")
        self.statusBar().showMessage(label, 4000)

    def _handle_calibration_activity_changed(self, pending: int) -> None:
        self._update_queue_progress(
            pending=pending,
            label="Detecting chessboards",
            task_id=PROGRESS_TASK_DETECTION,
            total_attr="_calibration_activity_total",
            done_attr="_calibration_activity_done",
            last_attr="_calibration_activity_last",
            cancel_handler=self.calibration_controller.cancel_all,
        )

    def _update_signature_progress(self) -> None:
        pending = len(self._signature_pending) + len(self._signature_scan_queue)
        total = self._signature_scan_target
        if pending <= 0 or total <= 0:
            if self._signature_progress_started:
                self.progress_tracker.finish(PROGRESS_TASK_SIGNATURES)
                self._unregister_cancel_handler(PROGRESS_TASK_SIGNATURES)
            self._signature_progress_started = False
            self._signature_scan_target = 0
            self._signature_activity_total = 0
            self._signature_activity_done = 0
            self._signature_activity_last = 0
            return
        done = max(0, min(total - pending, total))
        if not self._signature_progress_started:
            self.progress_tracker.start(
                PROGRESS_TASK_SIGNATURES,
                "Scanning duplicates",
                total,
            )
            self._register_cancel_handler(
                PROGRESS_TASK_SIGNATURES,
                self._cancel_signature_scan_jobs,
            )
            self._signature_progress_started = True
        else:
            self.progress_tracker.update(
                PROGRESS_TASK_SIGNATURES,
                done,
                total,
            )
        self._signature_activity_total = total
        self._signature_activity_done = done
        self._signature_activity_last = pending

    def _handle_signature_activity_changed(self, pending: int) -> None:  # noqa: ARG002
        self._update_signature_progress()

    def _start_refinement_progress(self, total: int) -> None:
        if total <= 0:
            return
        self._refine_total = total
        self._refine_progress = 0
        self.progress_tracker.start(
            PROGRESS_TASK_REFINEMENT,
            "Refining chessboard corners",
            total,
        )
        self._register_cancel_handler(
            PROGRESS_TASK_REFINEMENT,
            self.calibration_refiner.cancel,
        )

    def _advance_refinement_progress(self) -> None:
        if self._refine_total <= 0:
            return
        self._refine_progress = min(self._refine_total, self._refine_progress + 1)
        self.progress_tracker.update(
            PROGRESS_TASK_REFINEMENT,
            self._refine_progress,
            self._refine_total,
        )

    def _finish_refinement_progress(self) -> None:
        if self._refine_total <= 0:
            return
        self._refine_total = 0
        self._refine_progress = 0
        self.progress_tracker.finish(PROGRESS_TASK_REFINEMENT)
        self._unregister_cancel_handler(PROGRESS_TASK_REFINEMENT)

    def _get_cached_overlay(
        self,
        base: str,
        channel: str,
        signature: Tuple[Any, ...],
    ) -> Optional[QPixmap]:
        bucket = self._overlay_cache.get(base)
        if not bucket:
            return None
        entry = bucket.get(channel)
        if not entry:
            return None
        if entry.signature != signature or not entry.pixmap or entry.pixmap.isNull():
            return None
        return entry.pixmap

    def _bump_signature_epoch(self) -> None:
        self.signature_controller.reset()
        self._reset_signature_tracking()

    def _reset_calibration_jobs(self) -> None:
        self.calibration_controller.reset()

    def _reset_signature_tracking(self) -> None:
        self._signature_pending.clear()
        self._signature_completed.clear()

    def _clear_pending_calibration_marks(self) -> None:
        self._pending_calibration_marks.clear()
        self._pending_calibration_forces.clear()
        if self._calibration_mark_timer.isActive():
            self._calibration_mark_timer.stop()

    def _stop_signature_scan(self) -> None:
        self._signature_scan_queue.clear()
        self._signature_scan_force = False
        if self._signature_scan_timer.isActive():
            self._signature_scan_timer.stop()
        self._update_signature_progress()

    def _cancel_signature_scan_jobs(self) -> None:
        self.signature_controller.cancel_all()
        self._signature_scan_queue.clear()
        self._signature_pending.clear()
        self._signature_scan_force = False
        if self._signature_scan_timer.isActive():
            self._signature_scan_timer.stop()
        self._signature_scan_target = 0
        self._update_signature_progress()

    def defer_calibration_analysis(self, base: Optional[str], *, force: bool = False) -> None:
        if (
            not base
            or not self.session.loader
            or base not in self.state.calibration_marked
        ):
            return
        self._pending_calibration_marks.add(base)
        if force:
            self._pending_calibration_forces.add(base)
        if not self._calibration_mark_timer.isActive():
            self._calibration_mark_timer.start(200)

    def cancel_deferred_calibration(self, base: Optional[str]) -> None:
        if not base:
            return
        self._pending_calibration_marks.discard(base)
        self._pending_calibration_forces.discard(base)
        if not self._pending_calibration_marks and self._calibration_mark_timer.isActive():
            self._calibration_mark_timer.stop()

    def _flush_pending_calibration_marks(self) -> None:
        if not self._pending_calibration_marks:
            return
        pending = list(self._pending_calibration_marks)
        self._pending_calibration_marks.clear()
        for base in pending:
            force = base in self._pending_calibration_forces
            self._pending_calibration_forces.discard(base)
            self._schedule_calibration_job(base, force=force, priority=True)

    def _has_cached_signatures(self, base: Optional[str]) -> bool:
        if not base:
            return False
        bucket = self.state.signatures.get(base)
        if not bucket:
            return False
        return bucket.get("lwir") is not None and bucket.get("visible") is not None

    def _schedule_signature_job(self, index: int, *, force: bool = False) -> bool:
        loader = self.session.loader
        base = self.session.get_base(index)
        if not loader or base is None:
            return False
        if not force:
            if base in self._signature_pending or base in self._signature_completed:
                return False
            if self._has_cached_signatures(base):
                self._signature_completed.add(base)
                return False
        if self.signature_controller.schedule(index, base, loader):
            self._signature_pending.add(base)
            return True
        return False

    def _finalize_signature_job(self, base: Optional[str], *, success: bool) -> None:
        if not base:
            return
        self._signature_pending.discard(base)
        if success:
            self._signature_completed.add(base)
        else:
            self._signature_completed.discard(base)
        if self._signature_activity_total > 0:
            self._signature_activity_done = min(
                self._signature_activity_total,
                self._signature_activity_done + 1,
            )
        self._update_signature_progress()
        if self._signature_scan_queue:
            self._signature_scan_timer.start(SIGNATURE_SCAN_TIMER_INTERVAL_MS)

    def _handle_signature_ready(
        self,
        index: int,
        base: str,
        lwir_sig: Optional[bytes],
        vis_sig: Optional[bytes],
    ) -> None:
        self._finalize_signature_job(base, success=True)
        if not self.session.loader:
            return
        total = self.session.total_pairs()
        if index >= total or index < 0:
            return
        cache_changed, reason_changed = self.session.apply_signatures(index, lwir_sig, vis_sig)
        if cache_changed:
            self._mark_cache_dirty()
        if not reason_changed:
            return
        self.invalidate_overlay_cache(base)
        self._update_delete_button()
        current_base = self._current_base()
        if current_base == base:
            self.load_image_pair(base)
            self._update_stats_panel()

    def _handle_signature_failed(self, base: str, message: str) -> None:
        self._finalize_signature_job(base, success=False)
        self.statusBar().showMessage(f"Duplicate analysis failed for {base}: {message}", 4000)

    def _handle_refinement_ready(
        self,
        base: str,
        refined: Dict[str, Optional[List[Tuple[float, float]]]],
    ) -> None:
        self._advance_refinement_progress()
        if base not in self.state.calibration_marked:
            return
        bucket = self.state.calibration_corners.setdefault(base, {})
        updated = False
        for channel, points in refined.items():
            if not points:
                continue
            bucket[channel] = points
            updated = True
        if not updated:
            return
        self.invalidate_overlay_cache(base)
        if self._current_base() == base:
            self.load_image_pair(base)
        self._mark_cache_dirty()

    def _handle_refinement_failed(self, base: str, message: str) -> None:
        self._advance_refinement_progress()
        self.statusBar().showMessage(f"Subpixel refinement failed for {base}: {message}", 5000)

    def _handle_refinement_batch_finished(self, success: int, failed: int) -> None:
        self._finish_refinement_progress()
        if success == 0 and failed == 0:
            self.statusBar().showMessage("Corner refinement cancelled.", 4000)
            return
        summary = f"Corner refinement finished ({success} updated"
        summary += f", {failed} skipped)" if failed else ")"
        self.statusBar().showMessage(summary, 4000)

    def _handle_calibration_solved(self, payload: Dict[str, Any]) -> None:
        self.progress_tracker.finish(PROGRESS_TASK_SOLVER)
        self._unregister_cancel_handler(PROGRESS_TASK_SOLVER)
        channels = payload.get("channels", {}) if isinstance(payload, dict) else {}
        if not channels:
            self.statusBar().showMessage("Calibration solver returned no data.", 4000)
            return
        for channel, data in channels.items():
            if not isinstance(data, dict):
                continue
            self.state.calibration_matrices[channel] = data
            per_view = data.get("per_view_errors") if isinstance(data, dict) else None
            if isinstance(per_view, dict):
                self.state.calibration_reproj_errors[channel] = {
                    base: float(err)
                    for base, err in per_view.items()
                    if isinstance(base, str) and isinstance(err, (int, float))
                }
        self._mark_cache_dirty()
        if self.view_rectified and self.session.has_images():
            self.load_current()
        file_path = payload.get("file_path") if isinstance(payload, dict) else None
        if file_path:
            self.statusBar().showMessage(f"Calibration saved to {Path(file_path).name}", 6000)
        else:
            self.statusBar().showMessage("Calibration matrices updated.", 6000)
        self._update_stats_panel()
        self._refresh_outlier_dialog_rows()

    def _open_calibration_outlier_dialog(self) -> None:
        if not self.session.loader:
            QMessageBox.information(self, "Calibration outliers", "Load a dataset first.")
            return
        bases = set(self.state.calibration_marked) | set(self.state.calibration_outliers)
        if not bases:
            QMessageBox.information(self, "Calibration outliers", "No calibration images to review.")
            return
        rows = self.session.build_outlier_rows(sorted(bases))
        if self._outlier_dialog and self._outlier_dialog.isVisible():
            self._outlier_dialog.update_rows(rows)
            self._outlier_dialog.raise_()
            self._outlier_dialog.activateWindow()
            return
        dialog = CalibrationOutliersDialog(rows, self, refresh_callback=self._refresh_calibration_solutions)
        self._outlier_dialog = dialog
        dialog.finished.connect(lambda _code: setattr(self, "_outlier_dialog", None))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        include, exclude = dialog.selected_bases()
        include_set = {b for b in include if b in self.session.loader.image_bases}
        exclude_set = {b for b in exclude if b in self.session.loader.image_bases}
        # Update state: excluded => outlier and unmarked; included => remove outlier and mark.
        self.state.calibration_outliers.difference_update(include_set)
        self.state.calibration_outliers.update(exclude_set)
        self.state.calibration_marked.difference_update(exclude_set)
        self.state.calibration_marked.update(include_set)
        self.state.rebuild_calibration_summary()
        self._mark_cache_dirty()
        self._update_stats_panel()
        current = self._current_base()
        if current in exclude_set:
            self.invalidate_overlay_cache(current)
            self.load_current()
        removed = ", ".join(sorted(exclude_set)) if exclude_set else "none"
        self.statusBar().showMessage(f"Calibration outliers updated (excluded: {removed})", 6000)

    def _refresh_calibration_solutions(self) -> None:
        self._handle_compute_calibration_action()
        self._handle_compute_extrinsic_action()

    def _refresh_outlier_dialog_rows(self) -> None:
        if not self._outlier_dialog or not self._outlier_dialog.isVisible():
            return
        bases = set(self.state.calibration_marked) | set(self.state.calibration_outliers)
        if not bases:
            return
        rows = self.session.build_outlier_rows(sorted(bases))
        self._outlier_dialog.update_rows(rows)

    def _handle_calibration_solver_failed(self, message: str) -> None:
        self.progress_tracker.finish(PROGRESS_TASK_SOLVER)
        self._unregister_cancel_handler(PROGRESS_TASK_SOLVER)
        self.statusBar().showMessage(f"Calibration solve failed: {message}", 6000)

    def _handle_extrinsic_solved(self, payload: Dict[str, Any]) -> None:
        self.progress_tracker.finish(PROGRESS_TASK_EXTRINSIC)
        self._unregister_cancel_handler(PROGRESS_TASK_EXTRINSIC)
        snapshot = dict(payload)
        file_path = snapshot.pop("file_path", None)
        per_pair = snapshot.pop("per_pair_errors", None)
        if isinstance(per_pair, list):
            self.state.extrinsic_pair_errors = {
                entry.get("base"): float(entry.get("translation_error", 0.0))
                for entry in per_pair
                if isinstance(entry, dict) and isinstance(entry.get("base"), str)
            }
        self.state.calibration_extrinsic = snapshot
        self._mark_cache_dirty()
        message = "Stereo calibration updated"
        if file_path:
            message = f"Stereo calibration saved to {Path(file_path).name}"
        self.statusBar().showMessage(message, 6000)
        self._update_stats_panel()
        self._refresh_outlier_dialog_rows()

    def _handle_extrinsic_failed(self, message: str) -> None:
        self.progress_tracker.finish(PROGRESS_TASK_EXTRINSIC)
        self._unregister_cancel_handler(PROGRESS_TASK_EXTRINSIC)
        self.statusBar().showMessage(f"Stereo calibration failed: {message}", 6000)

    def toggle_mark_current(self):
        base = self._current_base()
        if not base:
            return
        if not self.state.toggle_manual_mark(base, REASON_USER):
            return
        self.invalidate_overlay_cache(base)
        self._update_delete_button()
        self.load_image_pair(base)
        self._update_stats_panel()
        self._mark_cache_dirty()

    def toggle_calibration_current(self):
        base = self._current_base()
        if not base:
            return
        enable = base not in self.state.calibration_marked
        if not self.dataset_actions.set_calibration_mark(base, enable):
            return
        self.invalidate_overlay_cache(base)
        self.load_image_pair(base)
        self._update_stats_panel()
        self._mark_cache_dirty()
        self._reconcile_filter_state(show_warning=True)

    def import_calibration_data(self):
        self.dataset_actions.import_calibration_data()

    def export_calibration_debug(self) -> None:
        base = self._current_base()
        if not base:
            QMessageBox.information(
                self,
                "Calibration debug",
                "Load a dataset and select an image before exporting debug overlays.",
            )
            return
        results = self.state.calibration_results.get(base)
        corners = self.state.calibration_corners.get(base)
        if not results:
            QMessageBox.information(
                self,
                "Calibration debug",
                "No calibration detection data available yet. Run calibration detection first.",
            )
            return
        self._emit_calibration_debug(base, results, corners)

    def closeEvent(self, event):  # type: ignore[override]
        self._flush_cache(wait=True)
        super().closeEvent(event)

    def _collect_refinement_candidates(self) -> List[str]:
        if not self.session.loader:
            return []
        candidates: List[str] = []
        for base in self.session.loader.image_bases:
            if base not in self.state.calibration_marked:
                continue
            bucket = self.state.calibration_corners.get(base, {})
            if any(bucket.get(channel) for channel in ("lwir", "visible")):
                candidates.append(base)
        return candidates

    def _handle_refine_calibration_action(self) -> None:
        if not self.session.loader:
            QMessageBox.information(self, "Calibration", "Load a dataset first.")
            return
        targets = self._collect_refinement_candidates()
        if not targets:
            QMessageBox.information(
                self,
                "Calibration",
                "Tag images for calibration and run detection before refining corners.",
            )
            return
        reply = QMessageBox.question(
            self,
            "Refine corners",
            f"Refine subpixel corners for {len(targets)} image(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        queued = self.calibration_refiner.refine(targets)
        if queued:
            self._start_refinement_progress(queued)
            self.statusBar().showMessage(
                f"Refining chessboard corners for {queued} image(s)…",
                4000,
            )

    def _handle_compute_calibration_action(self) -> None:
        if not self.session.loader:
            QMessageBox.information(self, "Calibration", "Load a dataset first.")
            return
        samples = self._collect_calibration_samples()
        if not samples:
            QMessageBox.information(
                self,
                "Calibration",
                "No calibration samples available. Tag images and ensure detections succeed first.",
            )
            return
        channel_counts: Dict[str, int] = {"lwir": 0, "visible": 0}
        for sample in samples:
            channel_counts[sample.channel] = channel_counts.get(sample.channel, 0) + 1
        if all(count < 3 for count in channel_counts.values()):
            QMessageBox.information(
                self,
                "Calibration",
                "Need at least 3 valid samples per channel to calibrate.",
            )
            return
        reply = QMessageBox.question(
            self,
            "Compute calibration",
            (
                "Compute camera matrices using "
                f"{channel_counts.get('lwir', 0)} LWIR and {channel_counts.get('visible', 0)} visible sample(s)?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        if self.calibration_solver.solve(samples):
            self.progress_tracker.set_busy(
                PROGRESS_TASK_SOLVER,
                "Computing calibration matrices…",
            )
            self._register_cancel_handler(PROGRESS_TASK_SOLVER, self.calibration_solver.cancel)
            self.statusBar().showMessage("Computing calibration matrices…", 4000)

    def _handle_compute_extrinsic_action(self) -> None:
        if not self.session.loader:
            QMessageBox.information(self, "Calibration", "Load a dataset first.")
            return
        samples = self._collect_extrinsic_samples()
        if len(samples) < 3:
            QMessageBox.information(
                self,
                "Calibration",
                "Need at least 3 calibration images with detections on both cameras to solve extrinsics.",
            )
            return
        reply = QMessageBox.question(
            self,
            "Compute extrinsic transform",
            f"Compute LWIR ↔ Visible transform using {len(samples)} paired sample(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        if self.calibration_extrinsic_solver.solve(samples):
            self.progress_tracker.set_busy(
                PROGRESS_TASK_EXTRINSIC,
                "Computing stereo calibration…",
            )
            self._register_cancel_handler(PROGRESS_TASK_EXTRINSIC, self.calibration_extrinsic_solver.cancel)
            self.statusBar().showMessage("Computing stereo calibration…", 4000)

    def _calibration_results_path(self) -> Optional[Path]:
        if not self.session.dataset_path:
            return None
        return self.session.dataset_path / CALIBRATION_RESULTS_FILENAME

    def _handle_show_calibration_dialog(self) -> None:
        if not self.session.loader:
            QMessageBox.information(self, "Calibration", "Load a dataset first.")
            return
        dialog = CalibrationCheckDialog(
            self,
            self.state.calibration_matrices,
            self.state.calibration_extrinsic,
            self._calibration_results_path(),
        )
        dialog.exec()

    def _handle_rectified_toggle(self, enabled: bool) -> None:
        if enabled and not self._has_calibration_data():
            QMessageBox.information(
                self,
                "Rectified view",
                "Import calibration data before enabling rectified rendering.",
            )
            if hasattr(self.ui, "action_toggle_rectified"):
                self.ui.action_toggle_rectified.blockSignals(True)
                self.ui.action_toggle_rectified.setChecked(False)
                self.ui.action_toggle_rectified.blockSignals(False)
            return
        self.view_rectified = enabled
        self.statusBar().showMessage(
            "Rectified view enabled" if enabled else "Rectified view disabled",
            3000,
        )
        self.invalidate_overlay_cache()
        if self.session.has_images():
            self.load_current()

    def _handle_grid_toggle(self, enabled: bool) -> None:
        self.show_grid = enabled
        self._persist_preferences(show_grid=enabled)
        self.invalidate_overlay_cache()
        if self.session.has_images():
            self.load_current()

    def clear_empty_datasets(self) -> None:
        self.dataset_actions.clear_empty_datasets()

    def delete_marked_images(self):
        self.dataset_actions.delete_marked_images()
        self._reconcile_filter_state(show_warning=True)

    def restore_images(self):
        self.dataset_actions.restore_images()
        self._reconcile_filter_state(show_warning=True)

    def show_help_dialog(self):
        dialog = HelpDialog(self)
        dialog.exec()

    def _show_empty_state(self):
        self.lwir_view.set_placeholder(STATUS_NO_IMAGES)
        self.vis_view.set_placeholder(STATUS_NO_IMAGES)
        self.ui.text_metadata_lwir.clear()
        self.ui.text_metadata_vis.clear()
        self.ui.btn_prev.setEnabled(False)
        self.ui.btn_next.setEnabled(False)
        self.setWindowTitle("Image Viewer")
        self.state.signatures = {}
        self.state.calibration_marked.clear()
        self.state.calibration_results.clear()
        self.state.calibration_corners.clear()
        self.state.calibration_warnings.clear()
        self._reset_signature_tracking()
        self._stop_signature_scan()
        self._reset_calibration_jobs()
        self._clear_pending_calibration_marks()
        if self.overlay_prefetch_timer.isActive():
            self.overlay_prefetch_timer.stop()
        self._overlay_prefetch_queue.clear()
        self.invalidate_overlay_cache()
        self.progress_tracker.clear()
        self._reset_queue_progress_state()
        self._update_delete_button()
        self._update_restore_menu()
        self._update_stats_panel()
        self.progress_tracker.clear()

    def _update_delete_button(self):
        count = len(self.state.marked_for_delete)
        base_text = DELETE_BUTTON_TEXT
        if count:
            self.ui.btn_delete_marked.setText(f"{base_text} ({count})")
        else:
            self.ui.btn_delete_marked.setText(base_text)
        self.ui.btn_delete_marked.setEnabled(count > 0)
        delete_action = getattr(self.ui, "action_delete_selected", None)
        if delete_action:
            delete_action.setEnabled(count > 0)

    def _update_restore_menu(self):
        base_text = RESTORE_ACTION_TEXT
        if not hasattr(self.ui, "action_restore_images"):
            return
        count = self.session.count_trash_pairs()
        if count:
            self.ui.action_restore_images.setText(f"{base_text} ({count})")
        else:
            self.ui.action_restore_images.setText(base_text)
        self.ui.action_restore_images.setEnabled(self.session.loader is not None)

    def _update_stats_panel(self) -> None:
        if not hasattr(self, "stats_panel"):
            return
        missing_counts: Dict[str, int] = {"lwir": 0, "visible": 0}
        if self.session.loader:
            missing_counts = self.session.loader.missing_channel_counts()
        self.stats_panel.update_from_state(
            self.state,
            self.session.total_pairs(),
            missing_counts,
            self.state.calibration_reproj_errors,
            self.state.extrinsic_pair_errors,
        )
