"""Main Qt window for browsing multispectral datasets while orchestrating overlays, calibration,
label workflows, duplicate scans, and cache management.

Routes user actions into background services and keeps UI state, progress, and dataset sessions in sync.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QDialog, QInputDialog, QComboBox, QMenu
from PyQt6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QKeySequence,
    QPixmap,
    QShortcut,
)
from PyQt6.QtCore import Qt, QTimer, QPoint, QThreadPool, QEventLoop

from ui_mainwindow import Ui_MainWindow
from services import (
    CALIBRATION_RESULTS_FILENAME,
    CalibrationController,
    CalibrationDebugger,
    CalibrationExtrinsicSolver,
    CalibrationRefiner,
    CalibrationSolver,
    CalibrationWorkflowMixin,
    CancelController,
    DeferredCalibrationQueue,
    LabelWorkflow,
    OverlayPrefetcher,
    OverlayWorkflow,
    ProgressQueueManager,
    SignatureController,
    SignatureScanManager,
    UiStateHelper,
    build_controller,
)
from services.cache_writer import CacheFlushNotifier, CacheFlushRunnable, write_cache_payload
from services.dataset_actions import DatasetActions
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
from utils.reasons import REASON_KEY_MAP, REASON_SHORTCUTS
from utils.ui_messages import STATUS_NO_IMAGES
from widgets.zoom_pan import ZoomPanView
from widgets.calibration_check_dialog import CalibrationCheckDialog
from widgets.calibration_outliers_dialog import CalibrationOutliersDialog
from widgets.help_dialog import HelpDialog
from widgets.progress_panel import ProgressPanel
from widgets.stats_panel import StatsPanel
from widgets import style
from services.marking_controller import MarkingController


DEFAULT_DATASET_DIR = str(Path.home() / "umh/ros2_ws" / "images_eeha")
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
        self._default_label_yaml_path = Path(__file__).resolve().parent.parent / "config" / "labels_coco.yaml"
        self.show_grid = bool(preferences.get("show_grid", True))
        self.view_rectified = bool(preferences.get("view_rectified", False))
        self.show_labels = bool(preferences.get("show_labels", False))
        self.label_model_path = preferences.get("label_model")
        self.label_yaml_path = preferences.get("label_yaml") or (
            str(self._default_label_yaml_path) if self._default_label_yaml_path.exists() else None
        )
        self.label_input_mode = preferences.get("label_input_mode", "visible")
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
        self.cancel_controller = CancelController()
        self.queue_manager = ProgressQueueManager(self.progress_tracker, self.cancel_controller)
        self._refine_total = 0
        self._refine_progress = 0

        self.thread_pool = QThreadPool.globalInstance()
        self.calibration_thread_pool = QThreadPool(self)
        self.calibration_thread_pool.setMaxThreadCount(max(1, CALIBRATION_DETECT_MAX_WORKERS))
        self.signature_controller = SignatureController(self.thread_pool)
        self.signature_manager = SignatureScanManager(
            parent=self,
            session=self.session,
            controller=self.signature_controller,
            progress_tracker=self.progress_tracker,
            cancel_controller=self.cancel_controller,
            task_id=PROGRESS_TASK_SIGNATURES,
            max_inflight=SIGNATURE_SCAN_MAX_INFLIGHT,
            timer_interval_ms=SIGNATURE_SCAN_TIMER_INTERVAL_MS,
            cancel_state_callback=self._update_cancel_button,
        )
        self._reset_queue_progress_state()
        self.cache_timer = QTimer(self)
        self.cache_timer.setInterval(2000)
        self.cache_timer.setSingleShot(True)
        self.cache_timer.timeout.connect(self._flush_cache)
        self._cache_flush_notifier = CacheFlushNotifier(self)
        self._cache_flush_notifier.finished.connect(self._handle_cache_flush_finished)
        self._cache_flush_inflight = False
        self._cache_flush_pending: Optional[CachePersistPayload] = None
        self.overlay_prefetcher = OverlayPrefetcher(
            self,
            OVERLAY_PREFETCH_RADIUS,
            OVERLAY_PREFETCH_DELAY_MS,
            self._ensure_overlay_cached,
            lambda base: self.overlay_workflow.is_cached(base),
        )
        self.calibration_queue = DeferredCalibrationQueue(
            parent=self,
            interval_ms=200,
            validator=lambda base: bool(self.session.loader and base in self.state.calibration_marked),
            scheduler=lambda base, force: self._schedule_calibration_job(base, force=force, priority=True),
        )

        self.overlay_workflow = OverlayWorkflow(OVERLAY_CACHE_LIMIT)

        self.lwir_view = ZoomPanView(self)
        self.vis_view = ZoomPanView(self)
        self.stats_panel = StatsPanel()
        self.ui_helper = UiStateHelper(self.ui, self.stats_panel, self.session, self.state)
        self.marking_controller = MarkingController(
            parent=self,
            state=self.state,
            dataset_actions=self.dataset_actions,
            get_current_base=self._current_base,
            has_images=self.session.has_images,
            prev_image=self.prev_image,
            next_image=self.next_image,
            invalidate_overlay_cache=self.invalidate_overlay_cache,
            load_image_pair=self.load_image_pair,
            update_stats=self._update_stats_panel,
            update_delete_button=self._update_delete_button,
            mark_cache_dirty=self._mark_cache_dirty,
            status_message=lambda msg, dur=4000: self.statusBar().showMessage(msg, dur),
            schedule_calibration_job=lambda base, force=False: self._schedule_calibration_job(
                base,
                force=force,
                priority=True,
            ),
            reconcile_filter_state=lambda: self._reconcile_filter_state(show_warning=True),
            calibration_shortcut=CALIBRATION_TOGGLE_SHORTCUT,
        )

        self._setup_calibration_outlier_action()

        self.labeling_controller = None
        self.label_workflow: Optional[LabelWorkflow] = None
        self._manual_label_mode = False

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
        self.signature_manager.signatureReady.connect(self._handle_signature_ready)
        self.signature_manager.signatureFailed.connect(self._handle_signature_failed)

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
            (getattr(self.ui, "action_show_labels", None), self.show_labels),
            (getattr(self.ui, "action_label_manual_mode", None), self._manual_label_mode),
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
            reason_shortcut.activated.connect(lambda r=reason: self.marking_controller.handle_reason_shortcut(r))
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
        if hasattr(self.ui, "action_show_labels"):
            self.ui.action_show_labels.toggled.connect(self._handle_show_labels_toggle)
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
        if hasattr(self.ui, "action_label_config_model"):
            self.ui.action_label_config_model.triggered.connect(self._handle_configure_label_model)
        if hasattr(self.ui, "action_label_config_labels"):
            self.ui.action_label_config_labels.triggered.connect(self._handle_configure_label_yaml)
        if hasattr(self.ui, "action_label_current"):
            self.ui.action_label_current.triggered.connect(self._handle_label_current)
        if hasattr(self.ui, "action_label_dataset"):
            self.ui.action_label_dataset.triggered.connect(self._handle_label_dataset)
        if hasattr(self.ui, "action_label_clear_current"):
            self.ui.action_label_clear_current.triggered.connect(self._handle_clear_labels_current)
        if hasattr(self.ui, "action_label_manual_mode"):
            self.ui.action_label_manual_mode.toggled.connect(self._handle_manual_label_mode_toggle)
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
        self.lwir_view.contextRequested.connect(self.marking_controller.show_context_menu)
        self.vis_view.contextRequested.connect(self.marking_controller.show_context_menu)
        self.lwir_view.labelBoxDefined.connect(
            lambda l, t, r, b: self._handle_manual_box_defined("lwir", l, t, r, b)
        )
        self.vis_view.labelBoxDefined.connect(
            lambda l, t, r, b: self._handle_manual_box_defined("visible", l, t, r, b)
        )
        self.lwir_view.labelSelectionCanceled.connect(lambda: self._handle_manual_selection_canceled("lwir"))
        self.vis_view.labelSelectionCanceled.connect(lambda: self._handle_manual_selection_canceled("visible"))
        self.lwir_view.labelDeleteRequested.connect(
            lambda x, y, g: self._handle_manual_delete_request("lwir", x, y, g)
        )
        self.vis_view.labelDeleteRequested.connect(
            lambda x, y, g: self._handle_manual_delete_request("visible", x, y, g)
        )

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
            self.marking_controller.toggle_mark_current()
            handled = True
        elif event.key() == Qt.Key.Key_C and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            self.toggle_calibration_current()
            handled = True
        elif event.key() in REASON_KEY_MAP and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            self.marking_controller.handle_reason_shortcut(REASON_KEY_MAP[event.key()])
            handled = True
        elif event.key() == Qt.Key.Key_Escape:
            handled = self._cancel_manual_selection()

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
        self.signature_manager.cancel_all()
        self.calibration_refiner.cancel()
        self.calibration_solver.cancel()
        self.calibration_extrinsic_solver.cancel()
        self.signature_manager.reset_epoch()
        self._reset_calibration_jobs()
        self.signature_manager.reset_progress()
        self._clear_pending_calibration_marks()
        if self.label_workflow:
            self.label_workflow.clear_cache()
        self._manual_label_mode = False
        self._update_labeling_views()
        self._sync_action_states()
        self.overlay_prefetcher.clear()
        self.invalidate_overlay_cache()
        self.progress_tracker.clear()
        self._reset_queue_progress_state()
        self.cancel_controller.clear()
        self._update_cancel_button()

    def _prime_signature_scan(self, *, force: bool = False) -> None:
        result = self.signature_manager.prime(force=force)
        if result.status == "no-dataset":
            QMessageBox.information(self, "Duplicates", "Load a dataset before scanning for duplicates.")
            return
        if result.status == "no-images":
            self.statusBar().showMessage("No images available for duplicate scanning.", 4000)
            return
        if result.status == "cached":
            if force:
                self.statusBar().showMessage("Duplicate signatures already cached.", 4000)
            return
        if result.status == "queued":
            label = "Re-running duplicate sweep" if force else "Buscando duplicados"
            self.statusBar().showMessage(
                f"{label} en {result.total} imagen(es)…",
                4000,
            )
        self._update_cancel_button()

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
        prefs = self.session.cache_service.get_preferences()
        self.label_workflow = LabelWorkflow(dir_path, self._default_label_yaml_path, prefs)
        self.label_yaml_path = str(self.label_workflow.yaml_path) if self.label_workflow and self.label_workflow.yaml_path else None
        if self.label_workflow:
            self.label_workflow.ensure_controller()
            self.labeling_controller = self.label_workflow.controller
            self._on_class_map_updated()
        self.signature_manager.reset_epoch()
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
        self.signature_manager.schedule_index(self.current_index)

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
        total = self.session.total_pairs()
        self.overlay_prefetcher.prepare(
            current_index=self.current_index,
            total_pairs=total,
            current_base=base,
            calibration_marked=self.state.calibration_marked,
            get_base=self.session.get_base,
        )

    def _render_overlayed_pair(self, base: str) -> Tuple[Optional[QPixmap], Optional[QPixmap]]:
        reason = self.state.marked_for_delete.get(base)
        calibration_flag = base in self.state.calibration_marked
        calib_results = self.state.calibration_results.get(base, {})
        calib_corners = self.state.calibration_corners.get(base, {})
        warning_bucket = self.state.calibration_warnings.get(base, {})
        lwir_warning = warning_bucket.get("lwir") if isinstance(warning_bucket, dict) else None
        vis_warning = warning_bucket.get("visible") if isinstance(warning_bucket, dict) else None
        label_boxes_lwir: List[Tuple[str, float, float, float, float, QColor]] = []
        label_boxes_vis: List[Tuple[str, float, float, float, float, QColor]] = []
        label_sig_lwir: Optional[Tuple[Any, ...]] = None
        label_sig_vis: Optional[Tuple[Any, ...]] = None
        if self.show_labels:
            label_boxes_lwir = self._read_label_boxes(base, "lwir")
            label_boxes_vis = self._read_label_boxes(base, "visible")
            label_sig_lwir = self._label_signature(base, "lwir", label_boxes_lwir)
            label_sig_vis = self._label_signature(base, "visible", label_boxes_vis)

        display_lwir, display_vis = self.session.prepare_display_pair(base, self.view_rectified)
        display_lwir = self.overlay_workflow.render(
            base,
            "lwir",
            display_lwir,
            view_rectified=self.view_rectified,
            show_grid=self.show_grid,
            reason=reason,
            calibration=calibration_flag,
            calibration_detected=calib_results.get("lwir"),
            corner_points=calib_corners.get("lwir"),
            warning_text=lwir_warning,
            label_boxes=label_boxes_lwir,
            label_sig=label_sig_lwir,
        )
        display_vis = self.overlay_workflow.render(
            base,
            "visible",
            display_vis,
            view_rectified=self.view_rectified,
            show_grid=self.show_grid,
            reason=reason,
            calibration=calibration_flag,
            calibration_detected=calib_results.get("visible"),
            corner_points=calib_corners.get("visible"),
            warning_text=vis_warning,
            label_boxes=label_boxes_vis,
            label_sig=label_sig_vis,
        )
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

    def _ensure_overlay_cached(self, base: str) -> None:
        if not self.session.loader:
            return
        self._render_overlayed_pair(base)

    def _update_metadata_panel(self, base: str, type_dir: str, widget):
        self.ui_helper.update_metadata_panel(base, type_dir, widget)

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

    def _handle_configure_label_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select YOLO model", DEFAULT_DATASET_DIR, "Model (*.pt *.pth)")
        if not path:
            return
        self.label_model_path = path
        self._persist_preferences(label_model=path)
        if self.label_workflow:
            self.label_workflow.update_prefs(label_model=path)
            self.labeling_controller = self.label_workflow.controller
        self.statusBar().showMessage("Label model configured.", 3000)

    def _handle_configure_label_yaml(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select labels YAML", DEFAULT_DATASET_DIR, "YAML (*.yaml *.yml)")
        if not path:
            return
        self.label_yaml_path = path
        self._persist_preferences(label_yaml=path)
        if not self.label_workflow and self.session.dataset_path:
            self.label_workflow = LabelWorkflow(Path(self.session.dataset_path), self._default_label_yaml_path, self.session.cache_service.get_preferences())
        if self.label_workflow:
            self.label_workflow.set_label_yaml(Path(path))
            self.label_workflow.copy_yaml_to_dataset(Path(path))
            self._on_class_map_updated()
        self.statusBar().showMessage("Label classes loaded.", 3000)

    def _ensure_label_controller(self) -> bool:
        if self.labeling_controller:
            return True
        if not self.label_workflow:
            return False
        if self.label_workflow.ensure_controller():
            self.labeling_controller = self.label_workflow.controller
            return True
        return False

    def _label_channel(self) -> str:
        return "lwir" if self.label_input_mode == "lwir" else "visible"

    def _label_image_path(self, base: str) -> Optional[Path]:
        if not self.session.loader:
            return None
        return self.session.loader.get_image_path(base, self._label_channel())

    def _update_labeling_views(self) -> None:
        self.vis_view.set_labeling_mode(self._manual_label_mode)
        self.lwir_view.set_labeling_mode(self._manual_label_mode)

    def _handle_manual_label_mode_toggle(self, enabled: bool) -> None:
        if enabled and not self.session.has_images():
            QMessageBox.information(self, "Manual labels", "Load a dataset before enabling manual labelling.")
            self._manual_label_mode = False
            self._update_labeling_views()
            self._sync_action_states()
            return
        self._manual_label_mode = enabled
        self._update_labeling_views()
        self._sync_action_states()
        if enabled:
            if not self.show_labels:
                self.show_labels = True
                self._persist_preferences(show_labels=True)
            self.statusBar().showMessage(
                "Manual label mode: click two corners (rubber band), right-click to delete a box, Esc cancels.",
                5000,
            )
        else:
            self.statusBar().showMessage("Manual label mode off.", 2000)

    def _cancel_manual_selection(self) -> bool:
        if not self._manual_label_mode:
            return False
        self.vis_view.cancel_label_selection()
        self.lwir_view.cancel_label_selection()
        self.statusBar().showMessage("Label selection cancelled.", 2000)
        return True

    def _handle_manual_selection_canceled(self, channel: str) -> None:  # noqa: ARG002
        if self._manual_label_mode:
            self.statusBar().showMessage("Label selection cancelled.", 2000)

    def _prompt_label_class(self) -> Optional[str]:
        # Ensure class map is loaded (fallback to default YAML if present)
        if self.label_workflow and not self.label_workflow.class_map and self._default_label_yaml_path.exists():
            self.label_workflow.set_label_yaml(self._default_label_yaml_path)
            self._on_class_map_updated()
        choices = self.label_workflow.class_choices() if self.label_workflow else []
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Label class")
        dialog.setLabelText("Class name or id:")
        dialog.setComboBoxItems(choices)
        dialog.setComboBoxEditable(True)
        combo = dialog.findChild(QComboBox)
        if combo:
            combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            if combo.completer():
                completer = combo.completer()
                completer.setFilterMode(Qt.MatchFlag.MatchContains)
                completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
                completer.setCompletionMode(completer.CompletionMode.PopupCompletion)
        ok = dialog.exec() == QDialog.DialogCode.Accepted
        if not ok:
            return None
        text = dialog.textValue().strip()
        if combo and combo.completer():
            completion = combo.completer().currentCompletion()
            if completion:
                text = completion
        return text or None

    def _handle_manual_box_defined(self, channel: str, left: float, top: float, right: float, bottom: float) -> None:
        if not self._manual_label_mode:
            return
        base = self._current_base()
        if not base or not self.session.dataset_path:
            return
        cls_value = self._prompt_label_class()
        if not cls_value:
            self.statusBar().showMessage("Label entry cancelled.", 2000)
            return
        coords = [max(0.0, min(1.0, c)) for c in (left, top, right, bottom)]
        left, top, right, bottom = coords
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            self.statusBar().showMessage("Ignored zero-area box.", 2000)
            return
        x_center = left + width / 2
        y_center = top + height / 2
        path = self._label_path(base, channel)
        if not path:
            return
        resolved_cls = self.label_workflow.class_id_for_value(cls_value) if self.label_workflow else (cls_value or "")
        if self.label_workflow and resolved_cls is None:
            QMessageBox.warning(self, "Label class", "Select a class from the list or type a valid name/id.")
            return
        cls_id = resolved_cls or ""
        if cls_id == "":
            return
        if self.label_workflow:
            self.label_workflow.append_box(base, channel, cls_id, x_center, y_center, width, height)
        self.invalidate_overlay_cache(base)
        self.load_current()
        self.statusBar().showMessage(f"Added label to {channel}:{base}.", 3000)

    def _handle_manual_delete_request(self, channel: str, x_norm: float, y_norm: float, global_pos) -> None:
        if not self._manual_label_mode:
            return
        base = self._current_base()
        if not base or not self.session.dataset_path or not self.label_workflow:
            return
        target = self.label_workflow.find_label_display_at(base, channel, x_norm, y_norm)
        if not target:
            self.statusBar().showMessage("No label near click to delete.", 2000)
            return
        cls_display = target
        menu = QMenu(self)
        delete_action = menu.addAction(f"Delete label {cls_display}")
        chosen = menu.exec(global_pos)
        if chosen is delete_action:
            removed = self.label_workflow.delete_box_at(base, channel, x_norm, y_norm)
            if removed:
                self.invalidate_overlay_cache(base)
                self.load_current()

    def _handle_label_current(self) -> None:
        if not self.session.has_images():
            QMessageBox.information(self, "Labelling", "Load a dataset first.")
            return
        base = self._current_base()
        if not base:
            return
        if not self._ensure_label_controller():
            QMessageBox.information(self, "Labelling", "Configure a model and labels YAML first.")
            return
        img_path = self._label_image_path(base)
        if not img_path or not img_path.exists():
            QMessageBox.information(self, "Labelling", "No image available for the selected channel.")
            return
        channel = self._label_channel()
        boxes = self.labeling_controller.run_single(base, channel, img_path)
        self.statusBar().showMessage(f"Saved {len(boxes)} labels for {channel}:{base}.", 3000)
        self.invalidate_overlay_cache(base)
        self.load_current()

    def _handle_label_dataset(self) -> None:
        if not self.session.has_images():
            QMessageBox.information(self, "Labelling", "Load a dataset first.")
            return
        if not self._ensure_label_controller():
            QMessageBox.information(self, "Labelling", "Configure a model and labels YAML first.")
            return
        loader = self.session.loader
        channel = self._label_channel()
        total = self.session.total_pairs()
        done = 0
        for base in list(loader.image_bases):
            img_path = loader.get_image_path(base, channel)
            if not img_path or not img_path.exists():
                continue
            boxes = self.labeling_controller.run_single(base, channel, img_path)
            done += 1
            if done % 10 == 0:
                self.statusBar().showMessage(f"Labelled {done}/{total} images…", 2000)
        self.statusBar().showMessage(f"Labelling finished for channel {channel} ({done}/{total}).", 4000)
        self.invalidate_overlay_cache()
        self.load_current()

    def _handle_clear_labels_current(self) -> None:
        base = self._current_base()
        if not base:
            return
        if not self.session.dataset_path:
            return
        if self.label_workflow:
            self.label_workflow.clear_labels(base)
        self.invalidate_overlay_cache(base)
        self.load_current()
        self.statusBar().showMessage("Labels cleared for current image.", 3000)

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
        self.overlay_workflow.invalidate(base)

    def _read_label_boxes(self, base: str, channel: str) -> List[Tuple[str, float, float, float, float, QColor]]:
        if not self.label_workflow:
            return []
        return self.label_workflow.read_overlay_boxes(base, channel)

    def _label_signature(
        self,
        base: str,
        channel: str,
        boxes: List[Tuple[str, float, float, float, float, QColor]],
    ) -> Optional[Tuple[Any, ...]]:
        if not self.label_workflow:
            return None
        return self.label_workflow.label_signature(base, channel, boxes)

    def _label_path(self, base: str, channel: str) -> Optional[Path]:
        if not self.session.dataset_path or not self.label_workflow:
            return None
        return self.label_workflow.label_path(base, channel)

    def _reset_queue_progress_state(self) -> None:
        self.queue_manager.reset()
        self.signature_manager.reset_progress()

    def _handle_progress_snapshot(self, snapshot) -> None:
        if not self.progress_panel:
            return
        self.progress_panel.set_snapshot(snapshot)

    def _update_cancel_button(self) -> None:
        if not self.progress_panel:
            return
        task_id = self.cancel_controller.active_task()
        if not task_id:
            self.progress_panel.set_cancel_state(False)
            return
        tooltip = CANCEL_ACTION_LABELS.get(task_id, "Cancel current action")
        enabled = not self.cancel_controller.is_inflight(task_id)
        self.progress_panel.set_cancel_state(enabled, tooltip)

    def _handle_cancel_action(self) -> None:
        task_id = self.cancel_controller.active_task()
        if not task_id:
            return
        if self.cancel_controller.is_inflight(task_id):
            self.statusBar().showMessage("Cancellation already requested…", 2000)
            return
        handler = self.cancel_controller.handler_for(task_id)
        if not handler:
            return
        self.cancel_controller.mark_inflight(task_id)
        self._update_cancel_button()
        handler()
        self.progress_tracker.finish(task_id)
        self.cancel_controller.unregister(task_id)
        label = CANCEL_ACTION_LABELS.get(task_id, "Cancelling action…")
        self.statusBar().showMessage(label, 4000)
        self._update_cancel_button()

    def _handle_calibration_activity_changed(self, pending: int) -> None:
        self.queue_manager.update(
            pending=pending,
            label="Detecting chessboards",
            task_id=PROGRESS_TASK_DETECTION,
            cancel_handler=self.calibration_controller.cancel_all,
        )
        self._update_cancel_button()

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
        self.cancel_controller.register(
            PROGRESS_TASK_REFINEMENT,
            self.calibration_refiner.cancel,
        )
        self._update_cancel_button()

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
        self.cancel_controller.unregister(PROGRESS_TASK_REFINEMENT)
        self._update_cancel_button()

    def _reset_calibration_jobs(self) -> None:
        self.calibration_controller.reset()

    def _clear_pending_calibration_marks(self) -> None:
        self.calibration_queue.clear()

    def defer_calibration_analysis(self, base: Optional[str], *, force: bool = False) -> None:
        self.calibration_queue.defer(base, force=force)

    def cancel_deferred_calibration(self, base: Optional[str]) -> None:
        self.calibration_queue.cancel(base)

    def _handle_signature_ready(
        self,
        index: int,
        base: str,
        lwir_sig: Optional[bytes],
        vis_sig: Optional[bytes],
    ) -> None:
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
        self.cancel_controller.unregister(PROGRESS_TASK_SOLVER)
        self._update_cancel_button()
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
        self.cancel_controller.unregister(PROGRESS_TASK_SOLVER)
        self._update_cancel_button()
        self.statusBar().showMessage(f"Calibration solve failed: {message}", 6000)

    def _handle_extrinsic_solved(self, payload: Dict[str, Any]) -> None:
        self.progress_tracker.finish(PROGRESS_TASK_EXTRINSIC)
        self.cancel_controller.unregister(PROGRESS_TASK_EXTRINSIC)
        self._update_cancel_button()
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
        self.cancel_controller.unregister(PROGRESS_TASK_EXTRINSIC)
        self._update_cancel_button()
        self.statusBar().showMessage(f"Stereo calibration failed: {message}", 6000)

    def toggle_mark_current(self):
        self.marking_controller.toggle_mark_current()

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
            self.cancel_controller.register(PROGRESS_TASK_SOLVER, self.calibration_solver.cancel)
            self._update_cancel_button()
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
            self.cancel_controller.register(PROGRESS_TASK_EXTRINSIC, self.calibration_extrinsic_solver.cancel)
            self._update_cancel_button()
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

    def _handle_show_labels_toggle(self, enabled: bool) -> None:
        self.show_labels = enabled
        self._persist_preferences(show_labels=enabled)
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
        self.signature_manager.cancel_all()
        self._reset_calibration_jobs()
        self._clear_pending_calibration_marks()
        self.overlay_prefetcher.clear()
        self.invalidate_overlay_cache()
        self.progress_tracker.clear()
        self._reset_queue_progress_state()
        if self.label_workflow:
            self.label_workflow.clear_cache()
        self._manual_label_mode = False
        self._update_labeling_views()
        self._sync_action_states()
        self._update_delete_button()
        self._update_restore_menu()
        self._update_stats_panel()
        self.progress_tracker.clear()

    def _on_class_map_updated(self) -> None:
        if self.label_workflow:
            self.label_workflow.clear_cache()
        self.invalidate_overlay_cache()
        if self.session.has_images():
            self.load_current()

    def _update_delete_button(self):
        self.ui_helper.update_delete_button()

    def _update_restore_menu(self):
        self.ui_helper.update_restore_menu()

    def _update_stats_panel(self) -> None:
        self.ui_helper.update_stats_panel()
