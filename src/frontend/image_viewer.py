"""Main Qt window for browsing multispectral datasets while orchestrating overlays, calibration,
label workflows, duplicate scans, and cache management.

Routes user actions into background services and keeps UI state, progress, and dataset sessions in sync.
"""

import os
import time
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Set, Tuple

from PyQt6.QtCore import QEventLoop, QRunnable, Qt, QTimer
from PyQt6.QtGui import (QAction, QActionGroup, QColor, QIcon, QKeySequence,
                         QPixmap, QShortcut)
from PyQt6.QtWidgets import (QApplication, QComboBox, QCompleter, QDialog,
                             QFileDialog, QInputDialog, QMainWindow, QMenu,
                             QMessageBox, QStyle, QVBoxLayout)
from tqdm import tqdm

# Backend services (no Qt dependencies)
from backend.services import (CalibrationController, CalibrationDebugger,
                              CalibrationExtrinsicSolver, CalibrationRefiner,
                              CalibrationSolver, DeferredCalibrationQueue,
                              OverlayPrefetcher,
                              OverlayWorkflow, SignatureController,
                              SignatureScanManager)
from backend.services.cache_flush_coordinator import CacheFlushCoordinator
from backend.services.cache_service import CachePersistPayload
from backend.services.cache_writer import (CacheFlushNotifier,
                                           CacheFlushRunnable,
                                           write_cache_payload)
from backend.services.calibration_workflow import CalibrationWorkflow
from backend.services.collection import Collection
from backend.services.dataset_actions import DatasetActions
from backend.services.dataset_session import DatasetSession
from backend.services.filter_controller import FilterController
from backend.services.filter_modes import (FILTER_ACTION_NAMES, FILTER_ALL,
                                           FILTER_CAL_ANY,
                                           FILTER_MESSAGE_LABELS,
                                           FILTER_STATUS_TITLES)
from backend.services.handler_registry import get_handler_registry
from backend.services.labels import LabelService
from backend.services.marking_controller import MarkingController
from backend.services.navigation_controller import NavigationController
from backend.services.overlay_orchestrator import OverlayOrchestrator
from backend.services.pattern_sweep import PatternSweepRunnable
from backend.services.progress_tracker import ProgressTracker
from backend.services.quality.quality_controller import QualityController
from backend.services.quality.quality_scan_manager import QualityScanManager
from backend.services.thread_pool_manager import get_thread_pool_manager
from backend.services.view_state_controller import ViewStateController
from backend.services.workspace_cache_coordinator import get_cache_coordinator
from backend.services.workspace_config import get_workspace_config_service
from backend.services.workspace_inspector import scan_workspace
from backend.services.workspace_scan import WorkspaceScanRunnable
from backend.utils.table_writer import (format_outliers_table,
                                        write_table_to_log)
from common.log_utils import (log_debug, log_error, log_info, log_perf,
                              log_warning)
from common.reasons import (REASON_BLURRY, REASON_DUPLICATE, REASON_KEY_MAP,
                            REASON_MISSING_PAIR, REASON_MOTION, REASON_PATTERN,
                            REASON_SHORTCUTS, REASON_SYNC)
from config import get_config
# Frontend services (Qt-dependent)
from frontend.services import (CancelController, ProgressQueueManager,
                               UiStateHelper)
from frontend.services.ui_event_handler import UIEventHandler
from frontend.ui_mainwindow import Ui_MainWindow
from frontend.utils.ui_guards import require_dataset, require_images
from frontend.utils.ui_messages import STATUS_NO_IMAGES
from frontend.widgets import style
from frontend.widgets.calibration_check_dialog import CalibrationCheckDialog
from frontend.widgets.calibration_outliers_dialog import \
    CalibrationOutliersDialog
from frontend.widgets.help_dialog import HelpDialog
from frontend.widgets.label_report_dialog import LabelReportDialog
from frontend.widgets.progress_panel import ProgressPanel
from frontend.widgets.stats_panel import StatsPanel
from frontend.widgets.workspace_dialog import WorkspacePanel
from frontend.widgets.zoom_pan import ZoomPanView

# Load centralized configuration
config = get_config()

class ImageViewer(QMainWindow):
    def __init__(self) -> None:

        super().__init__()

        # Initialize dialog references early to avoid AttributeError in callbacks
        self._outlier_dialog = None

        t0 = time.perf_counter()
        self._init_ui()
        log_perf(f"_init_ui {time.perf_counter() - t0:.3f}s")

        t0 = time.perf_counter()
        self._init_session_and_preferences()
        log_perf(f"_init_session_and_preferences {time.perf_counter() - t0:.3f}s")

        t0 = time.perf_counter()
        self._init_controllers()
        log_perf(f"_init_controllers {time.perf_counter() - t0:.3f}s")

        t0 = time.perf_counter()
        self._init_background_services()
        log_perf(f"_init_background_services {time.perf_counter() - t0:.3f}s")

        t0 = time.perf_counter()
        self._init_ui_components()
        log_perf(f"_init_ui_components {time.perf_counter() - t0:.3f}s")

        t0 = time.perf_counter()
        self._init_calibration_pipeline()
        log_perf(f"_init_calibration_pipeline {time.perf_counter() - t0:.3f}s")

        t0 = time.perf_counter()
        self._finalize_initialization()
        log_perf(f"_finalize_initialization {time.perf_counter() - t0:.3f}s")

    def _safe_status_message(self, msg: str, duration: int = 0) -> None:
        """Safely show a status message, handling None status bar."""
        status = self.statusBar()
        if status is not None:
            status.showMessage(msg, duration)
        else:
            log_warning(f"Status bar not available to show message: {msg}", "UI")

    def _safe_standard_icon(self, icon_type: QStyle.StandardPixmap) -> Optional[QIcon]:
        """Safely get standard icon, handling None style."""
        style = self.style()
        if style is not None:
            return style.standardIcon(icon_type)
        return None

    # ============================================================================
    # INITIALIZATION SECTIONS
    # ============================================================================

    def _init_ui(self) -> None:
        """Initialize UI and apply styling."""
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        if hasattr(self.ui, "centralwidget"):
            self.ui.centralwidget.setStyleSheet(f"background: {style.APP_BG};")

        for btn in (
            getattr(self.ui, "btn_prev", None),
            getattr(self.ui, "btn_next", None),
            getattr(self.ui, "btn_delete_marked", None),
        ):
            if btn:
                btn.setStyleSheet(style.scoped_button_style(btn.objectName()))

        self.progress_panel: Optional[ProgressPanel] = None
        self._setup_progress_panel()

    def _init_session_and_preferences(self) -> None:
        """Initialize session, state, and load user preferences."""

        t = time.perf_counter()
        self.session = DatasetSession()
        self.state = self.session.state
        log_perf(f"DatasetSession init {time.perf_counter() - t:.3f}s")

        t = time.perf_counter()
        preferences = self.session.cache_service.get_preferences()
        log_perf(f"get_preferences {time.perf_counter() - t:.3f}s")

        # Get thread pool limits
        pool_manager = get_thread_pool_manager()
        self._worker_limits = pool_manager.limits

        self.workspace_dir = preferences.get("workspace_dir") or str(config.default_dataset_dir)
        self._workspace_counts: Tuple[int, int] = (0, 0)

        # Validate workspace exists - if not, just use home directory as fallback
        workspace_path = Path(self.workspace_dir)
        if not workspace_path.exists():
            log_warning(f"Configured workspace does not exist: {self.workspace_dir}", "VIEWER")
            # Use home directory as fallback
            fallback = Path.home()
            log_info(f"Using fallback workspace: {fallback}", "VIEWER")
            self.workspace_dir = str(fallback)
            workspace_path = fallback

        try:
            get_cache_coordinator().set_workspace(workspace_path)
        except Exception:
            pass

        # Set workspace root in session's cache service to exclude from recent
        self.session.cache_service.set_workspace_root(Path(self.workspace_dir))

        # Initialize workspace config service
        get_workspace_config_service().set_workspace(Path(self.workspace_dir))

        t = time.perf_counter()
        self.dataset_actions = DatasetActions(self, str(config.default_dataset_dir))
        log_debug(f"DatasetActions init {time.perf_counter() - t:.3f}s", "PERF")

        t = time.perf_counter()
        self.recent_menu: Optional[QMenu] = None
        self._retitle_actions_for_collections()
        self._setup_recent_menu()
        self._update_workspace_label()
        log_debug(f"workspace setup {time.perf_counter() - t:.3f}s", "PERF")
        # Default: use our multiespectral schema (12 classes with attributes)
        self._default_label_yaml_path = Path(__file__).resolve().parent.parent / "config" / "labels_multiespectral_dataset.yaml"

        self.label_model_path = preferences.get("label_model")
        self.label_yaml_path = preferences.get("label_yaml") or (
            str(self._default_label_yaml_path) if self._default_label_yaml_path.exists() else None
        )
        self.label_input_mode = preferences.get("label_input_mode", "visible")

        self.view_rectified = bool(preferences.get("view_rectified", False))

        pref_mode = preferences.get("filter_mode")
        self._initial_filter_mode = FILTER_ALL
        if isinstance(pref_mode, str) and pref_mode in FILTER_ACTION_NAMES:
            self._initial_filter_mode = pref_mode

        self._view_prefs = {
            "show_labels": bool(preferences.get("show_labels", False)),
            "show_overlays": bool(preferences.get("show_overlays", True)),  # Default: show info overlays
            "grid_mode": preferences.get("grid_mode") or "thirds",  # Default to thirds
        }

        # Stereo alignment mode: "disabled", "full", "fov_focus", "max_overlap"
        saved_align_mode = preferences.get("align_mode", "")
        valid_align_modes = ("disabled", "full", "fov_focus", "max_overlap")
        self._align_mode = saved_align_mode if saved_align_mode in valid_align_modes else "disabled"

        # Corner display mode: "original", "subpixel", "both"
        saved_corner_mode = preferences.get("corner_display_mode", "")
        valid_corner_modes = ("original", "subpixel", "both")
        if saved_corner_mode in valid_corner_modes:
            self._corner_display_mode = saved_corner_mode
        else:
            self._corner_display_mode = "subpixel"  # Default to subpixel

    def _init_controllers(self) -> None:
        """Initialize state management controllers (view, filter, navigation, calibration)."""
        self.view_state = ViewStateController(
            session=self.session,
            parent=self,
            invalidate_overlay_cache=self.invalidate_overlay_cache,
            load_current=self.load_current,
            has_calibration_data=self._has_calibration_data,
            persist_preferences=self._persist_preferences,
        )
        self.view_state.statusMessage.connect(self._safe_status_message)
        self.view_state.load_preferences(**self._view_prefs)

        self.filter_controller = FilterController(
            session=self.session,
            state=self.state,
            ui=self.ui,
            on_filter_changed=self._on_filter_changed,
            parent=self,
        )
        self.filter_controller.filter_mode = self._initial_filter_mode
        self.filter_controller.filterModeChanged.connect(self._on_filter_mode_changed)

        self.navigation = NavigationController(
            session=self.session,
            get_filter_mode=lambda: self.filter_controller.filter_mode,
            filter_accepts=lambda base: self.filter_controller.filter_accepts_base(base) if base else False,
            parent=self,
        )
        # Debounced navigation: rapid key-repeat only updates the index;
        # the expensive load_current runs once the user stops pressing.
        self._nav_debounce_timer = QTimer(self)
        self._nav_debounce_timer.setSingleShot(True)
        self._nav_debounce_timer.setInterval(80)  # ms
        self._nav_debounce_timer.timeout.connect(self.load_current)
        self.navigation.indexChanged.connect(self._schedule_deferred_load)
        self.navigation.navigationBlocked.connect(lambda msg: self._safe_status_message(msg, 3000))

        self.event_handler = UIEventHandler(self)
        self._filter_group: Optional[QActionGroup] = None

    def _init_background_services(self) -> None:
        """Initialize background services: thread pools, progress tracking, queues, and managers."""
        self.progress_tracker = ProgressTracker(self._handle_progress_snapshot)
        self.cancel_controller = CancelController()
        self.queue_manager = ProgressQueueManager(self.progress_tracker, self.cancel_controller)
        self._refine_total = 0
        self._refine_progress = 0
        self._refine_tqdm: Optional[Any] = None  # Optional tqdm bar for refinement
        self._background_jobs: List[QRunnable] = []
        self._is_closing = False

        self.pool_manager = get_thread_pool_manager()
        self.thread_pool = self.pool_manager.global_pool()
        self.calibration_thread_pool = self.pool_manager.calibration_pool(parent=self)

        self.signature_controller = SignatureController(self.thread_pool)
        self.signature_manager = SignatureScanManager(
            parent=self,
            session=self.session,
            controller=self.signature_controller,
            progress_tracker=self.progress_tracker,
            cancel_controller=self.cancel_controller,
            task_id=config.progress_task_signatures,
            max_inflight=self._worker_limits.signature_scan_inflight,
            timer_interval_ms=config.signature_scan_timer_interval_ms,
            cancel_state_callback=self._update_cancel_button,
        )

        self.quality_controller = QualityController(self.thread_pool)
        self.quality_manager = QualityScanManager(
            parent=self,
            session=self.session,
            controller=self.quality_controller,
            progress_tracker=self.progress_tracker,
            cancel_controller=self.cancel_controller,
            task_id=config.progress_task_quality,
            max_inflight=self._worker_limits.signature_scan_inflight,
            timer_interval_ms=config.signature_scan_timer_interval_ms,
            on_cancel_state=self._update_cancel_button,
        )
        self.quality_manager.finished.connect(self._handle_quality_finished)
        self._reset_queue_progress_state()

        self.cache_timer = QTimer(self)
        self.cache_timer.setInterval(config.cache_flush_timer_interval_ms)
        self.cache_timer.setSingleShot(True)
        self.cache_timer.timeout.connect(self._flush_cache)
        self._cache_flush_coordinator = CacheFlushCoordinator()
        self._cache_flush_notifier = CacheFlushNotifier(self)
        self._cache_flush_notifier.finished.connect(self._handle_cache_flush_finished)

        self.overlay_workflow = OverlayWorkflow(config.overlay_cache_limit)
        self.overlay_prefetcher = OverlayPrefetcher(
            self,
            2,  # overlay_prefetch_radius
            75,  # overlay_prefetch_delay_ms
            self._ensure_overlay_cached,
            lambda base: self.overlay_workflow.is_cached(base),
        )
        self.overlay_orchestrator = OverlayOrchestrator(
            self.overlay_workflow,
            self.overlay_prefetcher,
            self.session,
            self.state,
            get_label_boxes=self._read_label_boxes,
            get_label_signature=self._label_signature,
            get_error_thresholds=self._calibration_error_thresholds,
            parent=self,
        )

        self.calibration_queue = DeferredCalibrationQueue(
            parent=self,
            interval_ms=config.calibration_queue_interval_ms,
            validator=lambda base: bool(self.session.loader and base in self.state.calibration_marked),
            scheduler=lambda base, force: self.calibration_workflow.schedule_calibration_job(base, force=force, priority=True),
        )

    def _init_ui_components(self) -> None:
        """Initialize UI components: views, panels, and controllers."""
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
            status_message=self._safe_status_message,
            schedule_calibration_job=lambda base, force=False: self.calibration_workflow.schedule_calibration_job(
                base,
                force=force,
                priority=True,
            ),
            reconcile_filter_state=lambda: self.filter_controller.reconcile_filter_state(show_warning=True),
            calibration_shortcut=config.calibration_toggle_shortcut,
            get_image_path=lambda base, channel: str(p) if (p := self.session.get_image_path(base, channel)) else None,
            enter_labelling_mode=lambda: self._toggle_manual_label_mode(True),
            enter_auto_labelling_mode=lambda: self._toggle_auto_label_mode(True),
        )

        self._setup_calibration_outlier_action()

        self.label_service: Optional[LabelService] = None
        self._manual_label_mode = False
        self._auto_label_active = False
        self._auto_detected_bases: set[tuple[str, str]] = set()  # (base, channel) already processed

    def _init_calibration_pipeline(self) -> None:
        """Initialize calibration pipeline: debugger, controllers, solvers, and workflow."""
        self.calibration_debugger = CalibrationDebugger(self.session, config.chessboard_size)

        self.calibration_controller = CalibrationController(
            self.session,
            config.chessboard_size,
            self.calibration_thread_pool,
            max_workers=self._worker_limits.calibration_detect,
        )

        self.calibration_workflow = CalibrationWorkflow(
            session=self.session,
            state=self.state,
            calibration_debugger=self.calibration_debugger,
            calibration_controller=self.calibration_controller,
            parent=self,
            invalidate_overlay_cache=self.invalidate_overlay_cache,
            load_image_pair=self.load_image_pair,
            get_current_base=self._current_base,
            update_stats_panel=self._update_stats_panel,
            mark_cache_dirty=self._mark_cache_dirty,
        )
        self.calibration_workflow.statusMessage.connect(self._safe_status_message)
        self.calibration_workflow.calibrationDetectionCompleted.connect(self._handle_calibration_detection_completed)
        self.calibration_controller.calibrationReady.connect(self.calibration_workflow.handle_calibration_ready)
        self.calibration_controller.calibrationFailed.connect(self.calibration_workflow.handle_calibration_failed)
        self.calibration_controller.activityChanged.connect(self._handle_calibration_activity_changed)

        # Auto-search calibration progress tracking variables
        self._calib_search_task_id: Optional[str] = None
        self._calib_search_total: int = 0
        self._calib_search_completed: int = 0
        self._calib_search_found: int = 0
        self._calib_search_tqdm: Optional[tqdm] = None

        self.calibration_refiner = CalibrationRefiner(
            self.session,
            config.chessboard_size,
            self.thread_pool,
        )
        self.calibration_refiner.refinementReady.connect(self._handle_refinement_ready)
        self.calibration_refiner.refinementFailed.connect(self._handle_refinement_failed)
        self.calibration_refiner.batchFinished.connect(self._handle_refinement_batch_finished)

        self.calibration_solver = CalibrationSolver(
            self.session,
            config.chessboard_size,
            self.thread_pool,
        )
        self.calibration_solver.calibrationSolved.connect(self._handle_calibration_solved)
        self.calibration_solver.calibrationFailed.connect(self._handle_calibration_solver_failed)

        self.calibration_extrinsic_solver = CalibrationExtrinsicSolver(
            self.session,
            config.chessboard_size,
            self.thread_pool,
        )
        self.calibration_extrinsic_solver.extrinsicSolved.connect(self._handle_extrinsic_solved)
        self.calibration_extrinsic_solver.extrinsicFailed.connect(self._handle_extrinsic_failed)

        self.signature_manager.signatureReady.connect(self._handle_signature_ready)
        self.signature_manager.signatureFailed.connect(self._handle_signature_failed)
        self.signature_manager.sweepCompleted.connect(self._handle_signature_sweep_completed)

    def _finalize_initialization(self) -> None:
        """Complete initialization: setup UI, connect signals, load initial state."""
        self._setup_image_views()
        self._setup_stats_panel()
        self.filter_controller.setup_filter_actions()
        self._sync_action_states()
        self._setup_workspace_menu()
        self._setup_fullscreen_toggle()
        self._mount_workspace_panel()
        self.connect_signals()
        self._register_shortcuts()
        self._show_empty_state()

        if hasattr(self.ui, "tab_widget") and self.ui.tab_widget.count() >= 1:
            self.ui.tab_widget.setCurrentIndex(0)
            self._update_workspace_title()

        try:
            self._auto_load_last_dataset()
        except Exception as e:
            log_warning(f"Failed to auto-load last dataset: {e}", "VIEWER")
            self._show_empty_state()
        self._update_cancel_button()

    # ============================================================================\n    # UI SETUP & CONFIGURATION\n    # ============================================================================

    def _sync_action_states(self) -> None:
        # Use flag to prevent recursion when syncing
        if getattr(self, "_syncing_actions", False):
            return
        self._syncing_actions = True

        try:
            # Sync boolean toggle actions
            toggles = [
                (getattr(self.ui, "action_toggle_rectified", None), self.view_rectified),
                (getattr(self.ui, "action_show_labels", None), self.view_state.show_labels),
                (getattr(self.ui, "action_show_overlays", None), self.view_state.show_overlays),
                (getattr(self.ui, "action_label_manual_mode", None), self._manual_label_mode),
                (getattr(self.ui, "action_label_auto_mode", None), self._auto_label_active),
            ]
            for action, state in toggles:
                if not action:
                    continue
                action.blockSignals(True)
                action.setChecked(state)
                action.blockSignals(False)

            # Sync grid action group - trigger the correct action without blocking
            grid_mode = self.view_state.grid_mode
            grid_actions = {
                "off": getattr(self.ui, "action_grid_off", None),
                "thirds": getattr(self.ui, "action_grid_thirds", None),
                "detailed": getattr(self.ui, "action_grid_detailed", None),
            }
            target_grid = grid_actions.get(grid_mode)
            if target_grid and not target_grid.isChecked():
                target_grid.setChecked(True)  # QActionGroup will uncheck others

            # Sync alignment action group
            align_actions = {
                "disabled": getattr(self.ui, "action_align_disabled", None),
                "full": getattr(self.ui, "action_align_full", None),
                "fov_focus": getattr(self.ui, "action_align_fov_focus", None),
                "max_overlap": getattr(self.ui, "action_align_max_overlap", None),
            }
            align_mode = getattr(self, "_align_mode", "disabled")
            target_align = align_actions.get(align_mode)
            if target_align and not target_align.isChecked():
                target_align.setChecked(True)  # QActionGroup will uncheck others

            # Sync corner display action group
            corner_actions = {
                "original": getattr(self.ui, "action_corners_original", None),
                "subpixel": getattr(self.ui, "action_corners_subpixel", None),
                "both": getattr(self.ui, "action_corners_both", None),
            }
            corner_mode = getattr(self, "_corner_display_mode", "subpixel")
            target_corner = corner_actions.get(corner_mode)
            if target_corner and not target_corner.isChecked():
                target_corner.setChecked(True)  # QActionGroup will uncheck others

            # Sync use_subpixel_corners toggle
            use_subpixel_action = getattr(self.ui, "action_use_subpixel_corners", None)
            if use_subpixel_action:
                use_subpixel = self.session.cache_service.get_preference("use_subpixel_corners", False)
                if use_subpixel_action.isChecked() != use_subpixel:
                    use_subpixel_action.blockSignals(True)
                    use_subpixel_action.setChecked(use_subpixel)
                    use_subpixel_action.blockSignals(False)

            self.filter_controller.update_filter_checks()
        finally:
            self._syncing_actions = False

    def _register_shortcuts(self) -> None:
        """Register keyboard shortcuts via UIEventHandler."""
        self.event_handler.register_navigation_shortcuts(
            prev_handler=self.prev_image,
            next_handler=self.next_image,
        )
        self.event_handler.register_marking_shortcuts(
            toggle_mark=self.toggle_mark_current,
            toggle_calibration=self.toggle_calibration_current,
            calibration_sequence=config.calibration_toggle_shortcut,
            reason_shortcuts=REASON_SHORTCUTS,
            reason_handler=self.marking_controller.handle_reason_shortcut,
        )

    def connect_signals(self) -> None:
        if hasattr(self.ui, "btn_prev_fast"):
            self.ui.btn_prev_fast.clicked.connect(lambda: self.prev_image(step=5))
        if hasattr(self.ui, "btn_prev"):
            self.ui.btn_prev.clicked.connect(self.prev_image)
        if hasattr(self.ui, "btn_goto"):
            self.ui.btn_goto.clicked.connect(self._handle_goto_image)
        if hasattr(self.ui, "btn_next"):
            self.ui.btn_next.clicked.connect(self.next_image)
        if hasattr(self.ui, "btn_next_fast"):
            self.ui.btn_next_fast.clicked.connect(lambda: self.next_image(step=5))

        # Window-level shortcuts (bypass QScrollArea key grab)
        QShortcut(QKeySequence("Shift+Left"), self).activated.connect(
            lambda: self.prev_image(step=5))
        QShortcut(QKeySequence("Shift+Right"), self).activated.connect(
            lambda: self.next_image(step=5))
        QShortcut(QKeySequence("Ctrl+G"), self).activated.connect(
            self._handle_goto_image)

        if hasattr(self.ui, "btn_delete_marked"):
            self.ui.btn_delete_marked.clicked.connect(self.delete_marked_images)
        if hasattr(self.ui, "action_load_dataset"):
            self.ui.action_load_dataset.triggered.connect(self._handle_check_datasets)
        if hasattr(self.ui, "action_set_workspace"):
            self.ui.action_set_workspace.triggered.connect(self._handle_set_workspace)
        if hasattr(self.ui, "action_check_datasets"):
            self.ui.action_check_datasets.triggered.connect(self._handle_check_datasets)
        if hasattr(self.ui, "action_save_status"):
            self.ui.action_save_status.triggered.connect(self._handle_save_status_action)
        if hasattr(self.ui, "action_run_duplicate_scan"):
            self.ui.action_run_duplicate_scan.triggered.connect(self._handle_run_duplicate_scan_action)
        if hasattr(self.ui, "action_delete_selected"):
            self.ui.action_delete_selected.triggered.connect(self.delete_marked_images)
        if hasattr(self.ui, "action_delete_blurry"):
            self.ui.action_delete_blurry.triggered.connect(lambda: self.dataset_actions.delete_by_reason(REASON_BLURRY))
        if hasattr(self.ui, "action_delete_motion"):
            self.ui.action_delete_motion.triggered.connect(lambda: self.dataset_actions.delete_by_reason(REASON_MOTION))
        if hasattr(self.ui, "action_delete_sync"):
            self.ui.action_delete_sync.triggered.connect(lambda: self.dataset_actions.delete_by_reason(REASON_SYNC))
        if hasattr(self.ui, "action_delete_duplicates"):
            self.ui.action_delete_duplicates.triggered.connect(lambda: self.dataset_actions.delete_by_reason(REASON_DUPLICATE))
        if hasattr(self.ui, "action_delete_missing_pair"):
            self.ui.action_delete_missing_pair.triggered.connect(lambda: self.dataset_actions.delete_by_reason(REASON_MISSING_PAIR))

        # Untag actions
        if hasattr(self.ui, "action_untag_all"):
            self.ui.action_untag_all.triggered.connect(lambda: self._untag_by_reason(None, "all"))
        if hasattr(self.ui, "action_untag_blurry"):
            self.ui.action_untag_blurry.triggered.connect(lambda: self._untag_by_reason(REASON_BLURRY, "blurry"))
        if hasattr(self.ui, "action_untag_motion"):
            self.ui.action_untag_motion.triggered.connect(lambda: self._untag_by_reason(REASON_MOTION, "motion"))
        if hasattr(self.ui, "action_untag_sync"):
            self.ui.action_untag_sync.triggered.connect(lambda: self._untag_by_reason(REASON_SYNC, "sync"))
        if hasattr(self.ui, "action_untag_missing"):
            self.ui.action_untag_missing.triggered.connect(lambda: self._untag_by_reason(REASON_MISSING_PAIR, "missing-pair"))
        if hasattr(self.ui, "action_untag_duplicates"):
            self.ui.action_untag_duplicates.triggered.connect(lambda: self._untag_by_reason(REASON_DUPLICATE, "duplicate"))

        # Restore actions
        if hasattr(self.ui, "action_restore_all"):
            self.ui.action_restore_all.triggered.connect(self.restore_images)
        if hasattr(self.ui, "action_restore_blurry"):
            self.ui.action_restore_blurry.triggered.connect(lambda: self._restore_by_reason(REASON_BLURRY, "blurry"))
        if hasattr(self.ui, "action_restore_motion"):
            self.ui.action_restore_motion.triggered.connect(lambda: self._restore_by_reason(REASON_MOTION, "motion"))
        if hasattr(self.ui, "action_restore_sync"):
            self.ui.action_restore_sync.triggered.connect(lambda: self._restore_by_reason(REASON_SYNC, "sync"))
        if hasattr(self.ui, "action_restore_missing"):
            self.ui.action_restore_missing.triggered.connect(lambda: self._restore_by_reason(REASON_MISSING_PAIR, "missing-pair"))
        if hasattr(self.ui, "action_restore_duplicates"):
            self.ui.action_restore_duplicates.triggered.connect(lambda: self._restore_by_reason(REASON_DUPLICATE, "duplicate"))

        if hasattr(self.ui, "action_run_quality_scan"):
            self.ui.action_run_quality_scan.triggered.connect(self._handle_run_quality_scan)
        if hasattr(self.ui, "action_run_pattern_scan"):
            self.ui.action_run_pattern_scan.triggered.connect(self._handle_run_pattern_scan_action)
        if hasattr(self.ui, "action_restore_images"):
            self.ui.action_restore_images.triggered.connect(self.restore_images)
        if hasattr(self.ui, "action_reset_dataset"):
            self.ui.action_reset_dataset.triggered.connect(self.dataset_actions.reset_dataset)
        if hasattr(self.ui, "action_import_calibration"):
            self.ui.action_import_calibration.triggered.connect(self.import_calibration_data)
        if hasattr(self.ui, "action_clear_empty_datasets"):
            self.ui.action_clear_empty_datasets.triggered.connect(self.clear_empty_datasets)
        if hasattr(self.ui, "action_toggle_rectified"):
            self.ui.action_toggle_rectified.toggled.connect(self._handle_rectified_toggle)

        # Grid submenu actions - use triggered (QActionGroup handles exclusivity)
        if hasattr(self.ui, "action_grid_off"):
            self.ui.action_grid_off.triggered.connect(lambda: self._set_grid_mode("off"))
        if hasattr(self.ui, "action_grid_thirds"):
            self.ui.action_grid_thirds.triggered.connect(lambda: self._set_grid_mode("thirds"))
        if hasattr(self.ui, "action_grid_detailed"):
            self.ui.action_grid_detailed.triggered.connect(lambda: self._set_grid_mode("detailed"))

        # Stereo alignment submenu actions - use triggered (QActionGroup handles exclusivity)
        if hasattr(self.ui, "action_align_disabled"):
            self.ui.action_align_disabled.triggered.connect(lambda: self._set_align_mode("disabled"))
        if hasattr(self.ui, "action_align_full"):
            self.ui.action_align_full.triggered.connect(lambda: self._set_align_mode("full"))
        if hasattr(self.ui, "action_align_fov_focus"):
            self.ui.action_align_fov_focus.triggered.connect(lambda: self._set_align_mode("fov_focus"))
        if hasattr(self.ui, "action_align_max_overlap"):
            self.ui.action_align_max_overlap.triggered.connect(lambda: self._set_align_mode("max_overlap"))

        # Corner display submenu actions - use triggered (QActionGroup handles exclusivity)
        if hasattr(self.ui, "action_corners_original"):
            self.ui.action_corners_original.triggered.connect(lambda: self._set_corner_display_mode("original"))
        if hasattr(self.ui, "action_corners_subpixel"):
            self.ui.action_corners_subpixel.triggered.connect(lambda: self._set_corner_display_mode("subpixel"))
        if hasattr(self.ui, "action_corners_both"):
            self.ui.action_corners_both.triggered.connect(lambda: self._set_corner_display_mode("both"))

        if hasattr(self.ui, "action_show_labels"):
            self.ui.action_show_labels.toggled.connect(self._handle_show_labels_toggle)
        if hasattr(self.ui, "action_show_overlays"):
            self.ui.action_show_overlays.toggled.connect(self._handle_show_overlays_toggle)
        if hasattr(self.ui, "action_calibration_debug"):
            self.ui.action_calibration_debug.triggered.connect(self.export_calibration_debug)
        if hasattr(self.ui, "action_run_calibration"):
            self.ui.action_run_calibration.triggered.connect(
                lambda: self.calibration_workflow.handle_run_calibration_action(self)
            )
        if hasattr(self.ui, "action_auto_calibration_search"):
            self.ui.action_auto_calibration_search.triggered.connect(self._handle_auto_calibration_search)
        if hasattr(self.ui, "action_calibration_refine"):
            self.ui.action_calibration_refine.triggered.connect(self._handle_refine_calibration_action)
        if hasattr(self.ui, "action_use_subpixel_corners"):
            self.ui.action_use_subpixel_corners.toggled.connect(self._handle_use_subpixel_toggle)
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
        if hasattr(self.ui, "action_label_reload_config"):
            self.ui.action_label_reload_config.triggered.connect(self._handle_reload_label_config)
        if hasattr(self.ui, "action_label_current"):
            self.ui.action_label_current.triggered.connect(self._handle_label_current)
        if hasattr(self.ui, "action_label_dataset"):
            self.ui.action_label_dataset.triggered.connect(self._handle_label_dataset)
        if hasattr(self.ui, "action_label_clear_current"):
            self.ui.action_label_clear_current.triggered.connect(self._handle_clear_labels_current)
        if hasattr(self.ui, "action_label_manual_mode"):
            self.ui.action_label_manual_mode.toggled.connect(self._handle_manual_label_mode_toggle)
        if hasattr(self.ui, "action_label_auto_mode"):
            self.ui.action_label_auto_mode.toggled.connect(self._handle_auto_label_mode_toggle)
        if hasattr(self.ui, "action_label_detection_channel"):
            self.ui.action_label_detection_channel.triggered.connect(self._handle_toggle_detection_channel)
            self._sync_detection_channel_action()
        if hasattr(self.ui, "action_label_report"):
            self.ui.action_label_report.triggered.connect(self._handle_label_report)
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

        # Store attributes safely
        if old_lwir is not None:
            self.lwir_view.setMinimumSize(old_lwir.minimumSize())
            self.lwir_view.setSizePolicy(old_lwir.sizePolicy())
        if old_vis is not None:
            self.vis_view.setMinimumSize(old_vis.minimumSize())
            self.vis_view.setSizePolicy(old_vis.sizePolicy())

        self.lwir_view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.vis_view.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Replace widgets safely
        if layout is not None and old_lwir is not None:
            layout.replaceWidget(old_lwir, self.lwir_view)
            old_lwir.deleteLater()
        if layout is not None and old_vis is not None:
            layout.replaceWidget(old_vis, self.vis_view)
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
            menu_bar = self.menuBar()
            if menu_bar is not None and action not in menu_bar.actions():
                menu_bar.addAction(action)

    def _setup_fullscreen_toggle(self) -> None:
        view_menu = getattr(self.ui, "menu_view", None)
        action = QAction("Toggle full screen", self)
        action.setShortcut(QKeySequence("F11"))
        action.setCheckable(True)
        action.triggered.connect(self._toggle_fullscreen)
        action.setChecked(self.isMaximized())
        if isinstance(view_menu, QMenu):
            view_menu.addSeparator()
            view_menu.addAction(action)
        self.action_toggle_fullscreen = action

    def _retitle_actions_for_collections(self) -> None:
        action = getattr(self.ui, "action_load_dataset", None)
        if action:
            action.setText("Load collection...")
        check_action = getattr(self.ui, "action_check_datasets", None)
        if check_action:
            check_action.setText("Open workspace summary")

    def _setup_recent_menu(self) -> None:
        # Use existing Load recent menu from ui_mainwindow if available
        recent = getattr(self.ui, "action_load_recent", None)
        if isinstance(recent, QMenu):
            recent.aboutToShow.connect(self._rebuild_recent_menu)
            self.recent_menu = recent
            return

        # Fallback: create menu dynamically
        menu = getattr(self.ui, "menu_dataset", None)
        if not isinstance(menu, QMenu):
            return
        recent = QMenu("Load recent", self)
        recent.aboutToShow.connect(self._rebuild_recent_menu)
        # Insert after load_dataset action
        load_action = getattr(self.ui, "action_load_dataset", None)
        if isinstance(load_action, QAction):
            actions = menu.actions()
            idx = actions.index(load_action) if load_action in actions else -1
            if idx >= 0 and idx + 1 < len(actions):
                menu.insertMenu(actions[idx + 1], recent)
            else:
                menu.addMenu(recent)
        else:
            menu.addMenu(recent)
        self.recent_menu = recent

    def _rebuild_recent_menu(self) -> None:
        if not self.recent_menu:
            return
        self.recent_menu.clear()
        entries = self.session.cache_service.recent_datasets()
        log_debug(f"[RECENT] Rebuilding menu with {len(entries)} entries: {entries}", "VIEWER")
        if not entries:
            empty_action = self.recent_menu.addAction("No recent datasets")
            if empty_action is not None:
                empty_action.setEnabled(False)
            return
        for path_str in entries:
            path = Path(path_str)
            label = self._format_recent_label(path)
            log_debug(f"[RECENT] Adding entry: {path} -> label: {label}", "VIEWER")
            action = self.recent_menu.addAction(label)
            if action is not None:
                action.triggered.connect(lambda _=False, p=path: self._load_path(p))

    def _mount_workspace_panel(self) -> None:
        container_layout = getattr(self.ui, "workspace_panel_layout", None)
        if not isinstance(container_layout, (QVBoxLayout,)):
            self.workspace_panel = None
            return
        workspace_path = Path(self._workspace_root())
        self.workspace_panel = WorkspacePanel(
            self,
            workspace_path,
            on_open=self._open_workspace_entry,
            progress_tracker=self.progress_tracker,
            autoload=False,
            cancel_controller=self.cancel_controller,
            on_cancel_state=self._update_cancel_button,
        )
        container_layout.addWidget(self.workspace_panel)
        try:
            if hasattr(self.ui, "tab_widget") and self.ui.tab_widget.currentWidget() == getattr(self.ui, "tab_workspace", None):
                if self.workspace_panel is not None:
                    self.workspace_panel.ensure_loaded()
        except Exception:
            pass
        try:
            self.ui.tab_widget.currentChanged.connect(self._handle_tab_changed)
        except Exception:
            pass
        try:
            self.workspace_panel.refreshed.connect(self._handle_workspace_refreshed)
        except Exception:
            pass

    def _format_recent_label(self, path: Path) -> str:
        kind = self.session.cache_service.dataset_kind(path)
        if kind == "collection":
            return f"Collection: {path.name}"
        if kind == "dataset":
            return f"Dataset: {path.name}"
        if self._is_collection_path(path):
            return f"Collection: {path.name}"
        if self._is_dataset_path(path):
            return f"Dataset: {path.name}"
        return str(path)

    def _setup_workspace_menu(self) -> None:
        menubar = getattr(self.ui, "menubar", None)
        if not menubar:
            return
        view_menu = getattr(self.ui, "menu_view", None)
        anchor = view_menu.menuAction() if isinstance(view_menu, QMenu) else None
        existing = getattr(self, "menu_workspace", None)
        self.menu_workspace = existing or QMenu("Workspace", self)
        if self.menu_workspace.menuAction() in menubar.actions():
            menubar.removeAction(self.menu_workspace.menuAction())
        if anchor:
            menubar.insertMenu(anchor, self.menu_workspace)
        else:
            menubar.addMenu(self.menu_workspace)
        self.menu_workspace.clear()

        # Helper to add action with icon
        def add_action(text: str, icon_type: QStyle.StandardPixmap, handler, menu: Optional[QMenu] = None):
            target_menu = menu if menu else self.menu_workspace
            action = target_menu.addAction(text)
            if action is not None:
                icon = self._safe_standard_icon(icon_type)
                if icon is not None:
                    action.setIcon(icon)
                action.triggered.connect(handler)

        def add_submenu(text: str, icon_type: QStyle.StandardPixmap, parent: QMenu) -> QMenu:
            submenu = QMenu(text, parent)
            icon = self._safe_standard_icon(icon_type)
            if icon is not None:
                submenu.setIcon(icon)
            parent.addMenu(submenu)
            return submenu

        # Build menu - Open/Refresh workspace
        add_action("Open workspace…", QStyle.StandardPixmap.SP_DirOpenIcon, self._handle_set_workspace)
        add_action("Refresh workspace", QStyle.StandardPixmap.SP_BrowserReload, self._handle_workspace_refresh)
        self.menu_workspace.addSeparator()

        # Detect delete candidates submenu (sweeps)
        detect_menu = add_submenu("Detect delete candidates", QStyle.StandardPixmap.SP_MediaPlay, self.menu_workspace)
        add_action("Run Duplicates sweep", QStyle.StandardPixmap.SP_MediaPlay, self._handle_workspace_duplicate_sweep, detect_menu)
        add_action("⚠ Run Blur/motion sweep (experimental)", QStyle.StandardPixmap.SP_MediaPlay, self._handle_workspace_quality_sweep, detect_menu)
        add_action("Run Patterns sweep", QStyle.StandardPixmap.SP_MediaPlay, self._handle_workspace_pattern_sweep, detect_menu)
        detect_menu.addSeparator()
        add_action("Run Missing pairs sweep", QStyle.StandardPixmap.SP_MediaPlay, self._handle_workspace_missing_sweep, detect_menu)
        detect_menu.addSeparator()
        add_action("Run all sweeps…", QStyle.StandardPixmap.SP_MediaPlay, self._handle_workspace_all_sweeps, detect_menu)

        # Untag delete candidates submenu
        untag_menu = add_submenu("Untag delete candidates", QStyle.StandardPixmap.SP_DialogResetButton, self.menu_workspace)
        add_action("Untag all…", QStyle.StandardPixmap.SP_DialogResetButton, lambda: self._run_workspace_untag_by_reason(None, "all"), untag_menu)
        untag_menu.addSeparator()
        add_action("Untag blurry…", QStyle.StandardPixmap.SP_DialogResetButton, lambda: self._run_workspace_untag_by_reason(REASON_BLURRY, "blurry"), untag_menu)
        add_action("Untag motion-blur…", QStyle.StandardPixmap.SP_DialogResetButton, lambda: self._run_workspace_untag_by_reason(REASON_MOTION, "motion"), untag_menu)
        add_action("Untag sync-mismatch…", QStyle.StandardPixmap.SP_DialogResetButton, lambda: self._run_workspace_untag_by_reason(REASON_SYNC, "sync"), untag_menu)
        add_action("Untag missing-pair…", QStyle.StandardPixmap.SP_DialogResetButton, lambda: self._run_workspace_untag_by_reason(REASON_MISSING_PAIR, "missing-pair"), untag_menu)
        add_action("Untag duplicates…", QStyle.StandardPixmap.SP_DialogResetButton, lambda: self._run_workspace_untag_by_reason(REASON_DUPLICATE, "duplicate"), untag_menu)

        # Delete marked submenu
        delete_menu = add_submenu("Delete marked", QStyle.StandardPixmap.SP_TrashIcon, self.menu_workspace)
        add_action("Delete all marked…", QStyle.StandardPixmap.SP_TrashIcon, self._handle_workspace_delete_all, delete_menu)
        delete_menu.addSeparator()
        add_action("Delete blurry…", QStyle.StandardPixmap.SP_TrashIcon, lambda: self._run_workspace_delete_by_reason(REASON_BLURRY, "blurry"), delete_menu)
        add_action("Delete motion-blur…", QStyle.StandardPixmap.SP_TrashIcon, lambda: self._run_workspace_delete_by_reason(REASON_MOTION, "motion"), delete_menu)
        add_action("Delete sync-mismatch…", QStyle.StandardPixmap.SP_TrashIcon, lambda: self._run_workspace_delete_by_reason(REASON_SYNC, "sync"), delete_menu)
        add_action("Delete missing-pair…", QStyle.StandardPixmap.SP_TrashIcon, lambda: self._run_workspace_delete_by_reason(REASON_MISSING_PAIR, "missing-pair"), delete_menu)
        add_action("Delete duplicates…", QStyle.StandardPixmap.SP_TrashIcon, lambda: self._run_workspace_delete_by_reason(REASON_DUPLICATE, "duplicate"), delete_menu)

        # Restore from trash submenu
        restore_menu = add_submenu("Restore from trash", QStyle.StandardPixmap.SP_BrowserReload, self.menu_workspace)
        add_action("Restore all…", QStyle.StandardPixmap.SP_BrowserReload, self._handle_workspace_restore_all, restore_menu)
        restore_menu.addSeparator()
        add_action("Restore blurry…", QStyle.StandardPixmap.SP_BrowserReload, lambda: self._run_workspace_restore_by_reason(REASON_BLURRY, "blurry"), restore_menu)
        add_action("Restore motion-blur…", QStyle.StandardPixmap.SP_BrowserReload, lambda: self._run_workspace_restore_by_reason(REASON_MOTION, "motion"), restore_menu)
        add_action("Restore sync-mismatch…", QStyle.StandardPixmap.SP_BrowserReload, lambda: self._run_workspace_restore_by_reason(REASON_SYNC, "sync"), restore_menu)
        add_action("Restore missing-pair…", QStyle.StandardPixmap.SP_BrowserReload, lambda: self._run_workspace_restore_by_reason(REASON_MISSING_PAIR, "missing-pair"), restore_menu)
        add_action("Restore duplicates…", QStyle.StandardPixmap.SP_BrowserReload, lambda: self._run_workspace_restore_by_reason(REASON_DUPLICATE, "duplicate"), restore_menu)

        self.menu_workspace.addSeparator()
        add_action("Clear empty dataset folders", QStyle.StandardPixmap.SP_TrashIcon, self.clear_empty_datasets)

        # Calibration settings
        self.menu_workspace.addSeparator()
        calib_menu = add_submenu("Default Calibration", QStyle.StandardPixmap.SP_FileDialogDetailedView, self.menu_workspace)
        add_action("Set from current dataset…", QStyle.StandardPixmap.SP_DialogSaveButton, self._handle_set_workspace_calibration, calib_menu)
        add_action("Clear default calibration", QStyle.StandardPixmap.SP_DialogResetButton, self._handle_clear_workspace_calibration, calib_menu)
        add_action("Show calibration info…", QStyle.StandardPixmap.SP_FileDialogInfoView, self._handle_show_workspace_calibration_info, calib_menu)

        self.menu_workspace.addSeparator()
        add_action("Label report…", QStyle.StandardPixmap.SP_FileDialogDetailedView, self._handle_workspace_label_report)

        self.menu_workspace.addSeparator()
        add_action("Reset selected dataset (dangerous)…", QStyle.StandardPixmap.SP_MessageBoxWarning, self._handle_workspace_reset_selected)
        add_action("Reset workspace (dangerous)…", QStyle.StandardPixmap.SP_MessageBoxWarning, self._handle_workspace_reset_all)

    def _auto_load_last_dataset(self) -> None:
        last_dataset = self.session.last_dataset()
        if not last_dataset:
            log_debug("No last dataset to auto-load", "VIEWER")
            return
        dataset_path = Path(last_dataset)

        # Don't auto-load workspace root as collection
        if self.workspace_dir and str(dataset_path) == self.workspace_dir:
            log_debug("Skipping auto-load of workspace root (not a dataset)", "VIEWER")
            return

        if not dataset_path.exists():
            log_warning(f"Last dataset path does not exist: {dataset_path}", "VIEWER")
            return

        # Check if this dataset is part of a collection - if so, load the collection instead
        collection_root = self._collection_root_for(dataset_path)
        if collection_root and collection_root != dataset_path:
            log_info(f"Auto-loading collection {collection_root.name} (contains last dataset {dataset_path.name})", "VIEWER")
            self._load_path(collection_root)
        else:
            log_info(f"Auto-loading last dataset: {dataset_path} ({dataset_path.name})", "VIEWER")
            self._load_path(dataset_path)

    @staticmethod
    def _is_dataset_path(path: Path) -> bool:
        return path.is_dir() and (path / "lwir").is_dir() and (path / "visible").is_dir()

    @staticmethod
    def _is_collection_path(path: Path) -> bool:
        if not path.is_dir():
            return False
        return Collection.is_collection_dir(path)

    def _load_path(self, path: Path) -> None:
        kind = self.session.cache_service.dataset_kind(path)
        log_debug(f"_load_path: path={path}, cached_kind={kind}", "VIEWER")
        if kind == "collection":
            log_debug("Loading as collection (from cache)", "VIEWER")
            self._load_collection(path)
            return
        if kind == "dataset":
            log_debug("Loading as dataset (from cache)", "VIEWER")
            self._load_dataset(path)
            return

        # PRIORITY ORDER:
        # 1. Collection (has child datasets inside) takes priority - loads multiple datasets together
        # 2. Dataset (has lwir/visible) - loads single dataset
        # 3. Neither - show error
        #
        # This ensures that when opening a collection, we load ALL child datasets,
        # not just the first one. A collection can have lwir/visible AND child datasets,
        # in which case it's treated as a collection (aggregate view).
        #
        # EXCEPTION: Workspace root is NEVER loaded as collection (even if it has collection structure)

        is_collection = self._is_collection_path(path)
        is_workspace_root = self.workspace_dir and str(path) == self.workspace_dir
        log_debug(f"is_collection_path={is_collection}, is_workspace_root={is_workspace_root}", "VIEWER")

        if is_collection and not is_workspace_root:
            log_debug("Loading as collection (detected)", "VIEWER")
            self._load_collection(path)
            return

        if is_workspace_root and is_collection:
            log_debug("Path is workspace root - NOT loading as collection", "VIEWER")
            QMessageBox.information(
                self,
                "Workspace",
                f"{path.name} is the workspace root.\n\n"
                "Use the Workspace panel to browse collections and datasets, "
                "or select a specific collection/dataset to open."
            )
            return

        is_dataset = self._is_dataset_path(path)
        log_debug(f"is_dataset_path={is_dataset}", "VIEWER")
        if is_dataset:
            log_debug("Loading as dataset (detected)", "VIEWER")
            self._load_dataset(path)
            return

        # Not a dataset or collection
        QMessageBox.warning(self, "Open", f"{path} is not a dataset or collection.")

    def _collection_root_for(self, path: Path) -> Optional[Path]:
        """Find collection root for path, stopping at workspace root."""
        cur = path
        while True:
            # Stop if we reached workspace root - workspace is NOT a collection
            if self.workspace_dir and str(cur) == self.workspace_dir:
                return None

            # Also stop if this IS the workspace dir (don't check if it's a collection)
            if cur.parent == cur:
                return None

            if self._is_collection_path(cur):
                return cur
            cur = cur.parent


    @property
    def current_index(self) -> int:
        """Delegate current_index to NavigationController."""
        return self.navigation.current_index

    @current_index.setter
    def current_index(self, value: int) -> None:
        """Delegate current_index setter to NavigationController."""
        self.navigation.current_index = value

    def _workspace_root(self) -> str:
        return str(self.workspace_dir) if self.workspace_dir else str(config.default_dataset_dir)

    def _workspace_label_config_path(self) -> Optional[Path]:
        """Return the workspace-level ``labels_config.yaml`` if it exists."""
        if self.workspace_dir:
            from backend.services.labels.label_storage import WORKSPACE_CONFIG_FILENAME
            p = Path(self.workspace_dir) / WORKSPACE_CONFIG_FILENAME
            if p.exists():
                return p
        return None

    def _update_workspace_label(self) -> None:
        panel = getattr(self, "workspace_panel", None)
        if panel:
            panel.path_label.setText(f"Workspace: {self.workspace_dir}")

    def _update_dataset_window_title(self, force: bool = False) -> None:
        if not force and hasattr(self.ui, "tab_widget") and self.ui.tab_widget.currentWidget() == getattr(self.ui, "tab_workspace", None):
            return
        if not self.session.dataset_path:
            self.setWindowTitle("Image Viewer")
            return
        path = Path(self.session.dataset_path)
        total = self.session.total_pairs()
        kind = self.session.loaded_kind or self.session.cache_service.dataset_kind(path)
        if kind == "collection":
            title = f"Image Viewer - {path.name} - {total} images (collection)"
        else:
            title = f"Image Viewer - {path.name} - {total} images"
        log_debug(f"Window title set to: {title}", "VIEWER")
        self.setWindowTitle(title)

    def _update_workspace_title(self) -> None:
        datasets, collections = self._workspace_counts
        label = "Workspace Viewer"
        if datasets or collections:
            label = f"Workspace Viewer - {datasets} dataset(s), {collections} collection(s)"
        self.setWindowTitle(label)

    def _handle_workspace_refreshed(self, datasets: int, collections: int) -> None:
        self._workspace_counts = (datasets, collections)
        if hasattr(self.ui, "tab_widget") and self.ui.tab_widget.currentWidget() == getattr(self.ui, "tab_workspace", None):
            self._update_workspace_title()

    def _handle_tab_changed(self, index: int) -> None:
        """Handle tab change to refresh workspace when switching to it."""
        try:
            current_widget = self.ui.tab_widget.widget(index)
            workspace_tab = getattr(self.ui, "tab_workspace", None)

            if current_widget == workspace_tab and workspace_tab is not None:
                log_debug("Switched to workspace tab, refreshing...", "VIEWER")
                if hasattr(self, 'workspace_panel') and self.workspace_panel is not None:
                    self.workspace_panel.refresh_workspace()
        except Exception as e:
            log_error(f"Failed to handle tab change: {e}", "VIEWER")
        widget = None
        try:
            widget = self.ui.tab_widget.widget(index)
        except Exception:
            pass
        if widget == getattr(self.ui, "tab_workspace", None):
            panel = getattr(self, "workspace_panel", None)
            if panel is not None:
                panel.ensure_loaded()
            self._update_workspace_title()
        else:
            self._update_dataset_window_title(force=True)

    def _toggle_fullscreen(self, checked: Optional[bool] = None) -> None:
        # Use maximized state instead of borderless full screen to keep the title bar visible.
        target_maximized = bool(checked) if checked is not None else not self.isMaximized()
        if target_maximized:
            self.showMaximized()
        else:
            self.showNormal()
        action = getattr(self, "action_toggle_fullscreen", None)
        if isinstance(action, QAction):
            action.blockSignals(True)
            action.setChecked(self.isMaximized())
            action.blockSignals(False)


    def keyPressEvent(self, event):
        """Keyboard shortcuts for navigation and tagging."""
        if event.key() == Qt.Key.Key_F11:
            self._toggle_fullscreen()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape and self.isFullScreen() and not self.session.has_images():
            self._toggle_fullscreen()
            event.accept()
            return

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
            if not handled and self.isFullScreen():
                self._toggle_fullscreen()
                handled = True

        if handled:
            event.accept()
            return

        super().keyPressEvent(event)

    def _update_cancel_button(self) -> None:
        """Sync cancel button state with CancelController."""
        panel = getattr(self, "progress_panel", None)
        if not panel:
            return
        try:
            task_id = self.cancel_controller.active_task()
            if not task_id:
                panel.set_cancel_state(False)
                return
            tooltip = config.cancel_action_labels.get(task_id, "Cancel current action")
            enabled = not self.cancel_controller.is_inflight(task_id)
            panel.set_cancel_state(enabled, tooltip)
        except Exception:
            # Never fail UI render on cancel state.
            pass

    def _handle_cancel_action(self) -> None:
        """Cancel currently active background task (duplicates/quality/workspace sweeps/etc.)."""
        task_id = self.cancel_controller.active_task()
        if not task_id:
            return
        if self.cancel_controller.is_inflight(task_id):
            self._safe_status_message("Cancellation already requested…", 2000)
            return
        handler = self.cancel_controller.handler_for(task_id)
        if not handler:
            return
        # Mark inflight to avoid re-entrant cancels.
        self.cancel_controller.mark_inflight(task_id)
        self._update_cancel_button()
        try:
            handler()
        finally:
            # Ensure progress UI is cleared even if handler raises.
            try:
                self.progress_tracker.finish(task_id)
            except Exception:
                pass
            self.cancel_controller.unregister(task_id)
            self._update_cancel_button()
            label = config.cancel_action_labels.get(task_id, "Cancelling action…")
            self._safe_status_message(label, 4000)

    def _cancel_manual_selection(self) -> bool:
        """Cancel active manual selection and return True if there was one."""
        if hasattr(self, "manual_selection_controller") and self.manual_selection_controller.active_channel:
            self.manual_selection_controller.cancel()
            return True
        return False

    def select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select dataset directory", self._workspace_root())

        if dir_path:
            self._load_dataset(Path(dir_path))

    def _handle_set_workspace(self) -> None:
        start_dir = self._workspace_root()
        dir_path = QFileDialog.getExistingDirectory(self, "Select workspace directory", start_dir)
        if not dir_path:
            return
        self.workspace_dir = dir_path
        try:
            get_cache_coordinator().set_workspace(Path(dir_path))
        except Exception:
            pass
        # Update workspace root in cache service to exclude from recent
        self.session.cache_service.set_workspace_root(Path(dir_path))
        # Update workspace config service
        get_workspace_config_service().set_workspace(Path(dir_path))
        self.dataset_actions.default_dataset_dir = dir_path
        self.session.cache_service.set_preference("workspace_dir", dir_path)
        self._safe_status_message(f"Workspace set to {dir_path}", 5000)
        self._background_workspace_scan(Path(dir_path))
        self._update_workspace_label()
        panel = getattr(self, "workspace_panel", None)
        if panel is not None:
            panel.set_workspace_dir(Path(dir_path), force_refresh=True)

    def _handle_check_datasets(self) -> None:
        workspace_path = self._workspace_path_or_warn()
        panel = getattr(self, "workspace_panel", None)
        if not workspace_path or panel is None:
            return
        panel.set_workspace_dir(workspace_path, force_refresh=True)
        if hasattr(self.ui, "tab_widget"):
            self.ui.tab_widget.setCurrentWidget(self.ui.tab_workspace)

    def _open_workspace_sweep(self) -> None:
        workspace_path = self._workspace_path_or_warn()
        panel = getattr(self, "workspace_panel", None)
        if not workspace_path or panel is None:
            return
        panel.set_workspace_dir(workspace_path, force_refresh=False)
        panel._open_sweep_dialog()
        if hasattr(self.ui, "tab_widget"):
            self.ui.tab_widget.setCurrentWidget(self.ui.tab_workspace)

    def _workspace_path_or_warn(self) -> Optional[Path]:
        workspace_path = Path(self._workspace_root())
        if not workspace_path.exists():
            QMessageBox.warning(self, "Workspace", "Set a workspace directory first.")
            return None
        return workspace_path

    def _run_workspace_sweep_with_flags(
        self,
        *,
        title: str,
        prompt: str,
        run_missing: bool = False,
        run_duplicates: bool = False,
        run_quality: bool = False,
        run_patterns: bool = False,
        delete_marked: bool = False,
        restore_all: bool = False,
    ) -> None:
        workspace_path = self._workspace_path_or_warn()
        if not workspace_path or not getattr(self, "workspace_panel", None):
            return
        reply = QMessageBox.question(
            self,
            title,
            prompt,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        panel = self.workspace_panel
        if panel is None:
            return
        panel.set_workspace_dir(workspace_path, force_refresh=False)
        panel._start_sweep(
            run_missing=run_missing,
            run_duplicates=run_duplicates,
            run_quality=run_quality,
            run_patterns=run_patterns,
            delete_marked=delete_marked,
            restore_all=restore_all,
        )
        if hasattr(self.ui, "tab_widget"):
            self.ui.tab_widget.setCurrentWidget(self.ui.tab_workspace)

    def _handle_workspace_duplicate_sweep(self) -> None:
        self._run_workspace_sweep_with_flags(
            title="Workspace duplicate sweep",
            prompt="Run duplicate detection across every dataset? This recalculates signatures and auto-marks duplicates.",
            run_duplicates=True,
        )

    def _handle_workspace_missing_sweep(self) -> None:
        self._run_workspace_sweep_with_flags(
            title="Workspace missing-pair sweep",
            prompt="Re-scan missing pairs across every dataset? This only marks missing images.",
            run_missing=True,
        )

    def _handle_workspace_quality_sweep(self) -> None:
        self._run_workspace_sweep_with_flags(
            title="Workspace blur/motion sweep",
            prompt="Run blur/motion analysis across every dataset? This can take time.",
            run_quality=True,
        )

    def _handle_workspace_pattern_sweep(self) -> None:
        self._run_workspace_sweep_with_flags(
            title="Workspace pattern sweep",
            prompt="Run pattern matching across every dataset? This auto-marks pattern matches.",
            run_patterns=True,
        )

    def _handle_workspace_all_sweeps(self) -> None:
        self._run_workspace_sweep_with_flags(
            title="Workspace full sweep",
            prompt="Run missing-pair, duplicate, blur/motion, and pattern sweeps across every dataset?",
            run_missing=True,
            run_duplicates=True,
            run_quality=True,
            run_patterns=True,
        )

    def _handle_workspace_delete_all(self) -> None:
        self._run_workspace_sweep_with_flags(
            title="Delete marked across workspace",
            prompt="Delete all images marked for deletion in every dataset? This moves them to to_delete folders.",
            delete_marked=True,
        )


    def _run_workspace_delete_by_reason(self, reason: str, reason_label: str) -> None:
        """Delete images with specific reason across all datasets in workspace."""
        workspace_path = self._workspace_path_or_warn()
        if not workspace_path:
            return

        reply = QMessageBox.question(
            self,
            f"Delete {reason_label} across workspace",
            f"Delete all images marked as '{reason_label}' in every dataset?\n\n"
            "This will move them to to_delete folders.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Run in background
        self._safe_status_message(f"Deleting {reason_label} images across workspace...", 5000)
        infos = scan_workspace(workspace_path)

        total_deleted = 0
        for info in infos:
            if info.is_collection:
                continue
            session = DatasetSession()
            if not session.load(info.path):
                continue
            # Filter marks by reason - only keep matching ones temporarily
            marks = session.state.cache_data.get("marks", {})
            to_delete = {base: r for base, r in marks.items() if r == reason}
            if to_delete:
                # Temporarily replace marks with filtered set, delete, then restore
                original_marks = dict(marks)
                session.state.cache_data["marks"] = to_delete
                outcome = session.delete_marked_entries()
                total_deleted += outcome.moved
                # Save changes
                payload = session.snapshot_cache_payload()
                if payload:
                    write_cache_payload(payload)

        self._safe_status_message(f"Deleted {total_deleted} {reason_label} images across workspace", 5000)
        # Refresh workspace panel
        panel = getattr(self, "workspace_panel", None)
        if panel:
            panel.refresh()

    def _run_workspace_untag_by_reason(self, reason: Optional[str], reason_label: str) -> None:
        """Untag (remove marks for) images with specific reason across all datasets in workspace."""
        workspace_path = self._workspace_path_or_warn()
        if not workspace_path:
            return

        what = f"'{reason_label}'" if reason else "all"
        reply = QMessageBox.question(
            self,
            f"Untag {reason_label} across workspace",
            f"Remove {what} delete tags from images in every dataset?\n\n"
            "This does NOT restore deleted images, only removes tags from non-deleted ones.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._safe_status_message(f"Untagging {reason_label} images across workspace...", 5000)
        infos = scan_workspace(workspace_path)

        total_untagged = 0
        for info in infos:
            if info.is_collection:
                continue
            session = DatasetSession()
            if not session.load(info.path):
                continue
            marks = session.state.cache_data.get("marks", {})
            if reason is None:
                # Untag all
                count = len(marks)
                if count > 0:
                    session.state.cache_data["marks"] = {}
                    session.state.rebuild_reason_counts()
                    session.mark_cache_dirty()
                    total_untagged += count
            else:
                # Untag specific reason - only affects that specific reason (new unified format)
                to_untag = [
                    base for base, entry in marks.items()
                    if isinstance(entry, dict) and entry.get("reason") == reason
                ]
                for base in to_untag:
                    marks.pop(base, None)
                if to_untag:
                    session.state.rebuild_reason_counts()
                    session.mark_cache_dirty()
                    total_untagged += len(to_untag)

            # Save changes
            if total_untagged > 0:
                payload = session.snapshot_cache_payload()
                if payload:
                    write_cache_payload(payload)

        self._safe_status_message(f"Untagged {total_untagged} {reason_label} images across workspace", 5000)
        panel = getattr(self, "workspace_panel", None)
        if panel:
            panel.refresh()

    def _run_workspace_restore_by_reason(self, reason: str, reason_label: str) -> None:
        """Restore trashed images with specific reason across all datasets in workspace.

        Note: Currently restores ALL trashed images since selective restore
        by reason requires additional tracking not yet implemented.
        """
        # Redirect to restore all with explanation
        reply = QMessageBox.question(
            self,
            f"Restore {reason_label} across workspace",
            f"Selective restore by reason is not yet fully implemented.\n\n"
            f"Would you like to restore ALL trashed images across the workspace instead?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._handle_workspace_restore_all()

    def _handle_workspace_restore_all(self) -> None:
        self._run_workspace_sweep_with_flags(
            title="Restore trashed across workspace",
            prompt="Restore all trashed images in every dataset and collection?",
            restore_all=True,
        )

    def _handle_workspace_reset_selected(self) -> None:
        panel = getattr(self, "workspace_panel", None)
        if panel is None:
            return
        if hasattr(self.ui, "tab_widget"):
            self.ui.tab_widget.setCurrentWidget(getattr(self.ui, "tab_workspace", None))
        panel.reset_selected()

    def _handle_workspace_reset_all(self) -> None:
        workspace_path = self._workspace_path_or_warn()
        panel = getattr(self, "workspace_panel", None)
        if not workspace_path or panel is None:
            return
        panel.set_workspace_dir(workspace_path, force_refresh=True)
        panel.reset_workspace()
        if hasattr(self.ui, "tab_widget"):
            self.ui.tab_widget.setCurrentWidget(getattr(self.ui, "tab_workspace", None))

    def _handle_set_workspace_calibration(self) -> None:
        """Set the current dataset's calibration as workspace default."""
        if not self.session.dataset_path:
            self._safe_status_message("No dataset loaded. Load a dataset with calibration first.", 5000)
            return

        config = get_config()

        # Check if current dataset has calibration files
        intrinsic_path = self.session.dataset_path / config.calibration_intrinsic_filename
        extrinsic_path = self.session.dataset_path / config.calibration_extrinsic_filename

        if not intrinsic_path.exists() and not extrinsic_path.exists():
            self._safe_status_message("Current dataset has no calibration files. Run calibration first.", 5000)
            return

        # Confirm with user
        files_info = []
        if intrinsic_path.exists():
            files_info.append(f"• Intrinsic: {config.calibration_intrinsic_filename}")
        if extrinsic_path.exists():
            files_info.append(f"• Extrinsic: {config.calibration_extrinsic_filename}")

        reply = QMessageBox.question(
            self,
            "Set Workspace Default Calibration",
            f"Set calibration from '{self.session.dataset_path.name}' as workspace default?\n\n"
            f"Files:\n{chr(10).join(files_info)}\n\n"
            "This calibration will be used for datasets that don't have their own calibration.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Set workspace default
        ws_config = get_workspace_config_service()
        success = ws_config.set_default_calibration(
            intrinsic_path=intrinsic_path if intrinsic_path.exists() else None,
            extrinsic_path=extrinsic_path if extrinsic_path.exists() else None,
            source_dataset=self.session.dataset_path.name,
        )

        if success:
            self._safe_status_message(
                f"Workspace default calibration set from '{self.session.dataset_path.name}'", 5000
            )
        else:
            self._safe_status_message("Failed to save workspace configuration", 5000)

    def _handle_clear_workspace_calibration(self) -> None:
        """Clear the workspace default calibration."""
        ws_config = get_workspace_config_service()

        if not ws_config.get_default_calibration():
            self._safe_status_message("No workspace default calibration is set", 3000)
            return

        reply = QMessageBox.question(
            self,
            "Clear Workspace Calibration",
            "Remove the workspace default calibration?\n\n"
            "Datasets without their own calibration will no longer inherit this.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        success = ws_config.clear_default_calibration()
        if success:
            self._safe_status_message("Workspace default calibration cleared", 5000)
        else:
            self._safe_status_message("Failed to clear workspace configuration", 5000)

    def _handle_show_workspace_calibration_info(self) -> None:
        """Show information about the workspace default calibration."""
        ws_config = get_workspace_config_service()
        default_calib = ws_config.get_default_calibration()

        if not default_calib:
            QMessageBox.information(
                self,
                "Workspace Calibration",
                "No workspace default calibration is set.\n\n"
                "Use 'Set from current dataset…' to set one.",
            )
            return

        info_lines = [f"Source dataset: {default_calib.source_dataset or 'Unknown'}"]
        if default_calib.intrinsic_path:
            exists = "✓" if default_calib.intrinsic_path.exists() else "✗ (missing)"
            info_lines.append(f"\nIntrinsic: {exists}\n  {default_calib.intrinsic_path}")
        if default_calib.extrinsic_path:
            exists = "✓" if default_calib.extrinsic_path.exists() else "✗ (missing)"
            info_lines.append(f"\nExtrinsic: {exists}\n  {default_calib.extrinsic_path}")

        QMessageBox.information(
            self,
            "Workspace Default Calibration",
            "\n".join(info_lines),
        )

    def _handle_workspace_refresh(self) -> None:
        workspace_path = self._workspace_path_or_warn()
        if not workspace_path or not getattr(self, "workspace_panel", None):
            return
        panel = self.workspace_panel
        if panel is not None:
            panel.set_workspace_dir(workspace_path, force_refresh=True)
            panel.refresh_workspace()
        if hasattr(self.ui, "tab_widget"):
            self.ui.tab_widget.setCurrentWidget(getattr(self.ui, "tab_workspace", None))

    def _open_workspace_entry(self, info) -> None:
        path = Path(info.path)
        is_coll = getattr(info, "is_collection", False)
        log_debug(f"_open_workspace_entry: name={info.name}, path={path}, is_collection={is_coll}", "VIEWER")
        if is_coll:
            self._load_collection(path)
        else:
            self._load_dataset(path)

    def _background_workspace_scan(self, workspace_path: Path) -> None:
        worker = WorkspaceScanRunnable(workspace_path)
        self._background_jobs.append(worker)

        def _on_finished(count: int) -> None:
            self._safe_status_message(f"Workspace scan finished: {count} dataset(s)", 4000)
            try:
                self._background_jobs.remove(worker)
            except ValueError:
                pass
            self.progress_tracker.finish(config.progress_task_workspace_scan)

        worker.signals.finished.connect(_on_finished)
        self.progress_tracker.set_busy(config.progress_task_workspace_scan, "Scanning workspace…")
        self.thread_pool.start(worker)

    def _reset_runtime_state(self) -> None:
        self.current_index = 0
        self.session.reset_state()
        self.calibration_controller.cancel_all()
        self.signature_manager.cancel_all()
        self.quality_manager.cancel_all()
        self.calibration_refiner.cancel()
        self.calibration_solver.cancel()
        self.calibration_extrinsic_solver.cancel()
        self.signature_manager.reset_epoch()
        self._reset_calibration_jobs()
        self.signature_manager.reset_progress()
        self.quality_manager.reset()
        self._clear_pending_calibration_marks()
        if self.label_service:
            self.label_service.clear_cache()
        self._manual_label_mode = False
        self._update_labeling_views()
        self._sync_action_states()
        self.overlay_prefetcher.clear()
        self.invalidate_overlay_cache()
        self.progress_tracker.clear()
        self._update_dataset_label()

    def _bump_signature_epoch(self) -> None:
        # Reset signature cache epoch so duplicate scan reschedules after mutations.
        self.signature_manager.reset_epoch()
        self.signature_manager.reset_progress()
        self._reset_queue_progress_state()
        self.cancel_controller.clear()
        self._update_cancel_button()

    def _update_dataset_label(self) -> None:
        label = getattr(self.ui, "label_dataset_path", None)
        if not label:
            return
        if not self.session.dataset_path:
            label.setText("No dataset loaded")
            return
        path = Path(self.session.dataset_path)
        kind = self.session.loaded_kind or self.session.cache_service.dataset_kind(path)
        if kind == "collection":
            label.setText(f"Collection: {path.name}")
            return
        if kind == "dataset":
            label.setText(f"Dataset: {path.name}")
            return
        if self._is_collection_path(path):
            label.setText(f"Collection: {path.name}")
        elif self._is_dataset_path(path):
            label.setText(f"Dataset: {path.name}")
        else:
            label.setText(str(path))

    def _prime_signature_scan(self, *, force: bool = False) -> None:
        result = self.signature_manager.prime(force=force)
        if result.status == "no-dataset":
            QMessageBox.information(self, "Duplicates", "Load a dataset or collection before scanning for duplicates.")
            return
        if result.status == "no-images":
            self._safe_status_message("No images available for duplicate scanning.", 4000)
            return
        if result.status == "cached":
            if force:
                self._safe_status_message("Duplicate signatures already cached.", 4000)
            return
        if result.status == "queued":
            label = "Re-running duplicate sweep" if force else "Buscando duplicados"
            self._safe_status_message(
                f"{label} en {result.total} imagen(es)…",
                4000,
            )
        self._update_cancel_button()

    def _handle_run_duplicate_scan_action(self) -> None:
        self._prime_signature_scan(force=True)

    def _handle_run_quality_scan(self) -> None:
        result, queued = self.quality_manager.prime(force=True)
        if result == "no-dataset":
            QMessageBox.information(self, "Quality sweep", "Load a dataset or collection first.")
            return
        if result == "no-images":
            self._safe_status_message("No images available for quality sweep.", 4000)
            return
        if queued == 0:
            self._safe_status_message("Quality sweep cached.", 3000)
            return
        self._safe_status_message(f"Scanning blur/motion in {queued} images…", 4000)

    def _handle_run_pattern_scan_action(self) -> None:
        if not require_images(self, "Pattern sweep"):
            return
        if not self.session.has_images() or not self.session.dataset_path:
            return

        patterns_dir = Path(config.patterns_dir)
        threshold = float(getattr(config, "pattern_match_threshold", 0.85))
        if not patterns_dir.exists():
            self._safe_status_message(f"Pattern sweep: no patterns dir ({patterns_dir}).", 6000)
            return

        bases = self.session.get_all_bases()
        if not bases:
            self._safe_status_message("Pattern sweep: no images.", 3000)
            return

        # Build items using session methods (works for both datasets and collections)
        items: List[Tuple[str, Optional[Path], Optional[Path]]] = [
            (base, self.session.get_image_path(base, "visible"), self.session.get_image_path(base, "lwir"))
            for base in bases
        ]

        started_on = str(self.session.dataset_path)
        total = len(items)
        task_id = config.progress_task_patterns
        runnable = PatternSweepRunnable(items, patterns_dir=patterns_dir, threshold=threshold)

        # Start progress bar
        self.progress_tracker.start(task_id, "Pattern sweep", total)

        def _on_progress(idx: int, total: int, base: str) -> None:
            self.progress_tracker.update(task_id, idx, total)

        def _on_failed(message: str) -> None:
            self.progress_tracker.finish(task_id)
            self._safe_status_message(f"Pattern sweep failed: {message}", 6000)

        def _on_finished(matched: Dict[str, str], scanned: int) -> None:
            self.progress_tracker.finish(task_id)
            if str(self.session.dataset_path) != started_on:
                return
            changed = False
            match_count = 0
            pattern_counts: Dict[str, int] = {}
            for base, pattern_name in matched.items():
                pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1
                # User note: use pattern name as auto reason to track which pattern
                reason_label = f"pattern:{pattern_name}"
                if self.state.set_mark_reason(base, reason_label, REASON_PATTERN, auto=True):
                    changed = True
                    match_count += 1
            self.state.rebuild_reason_counts()
            if changed:
                self.invalidate_overlay_cache()
                self._update_delete_button()
                self._update_stats_panel()
                self._mark_cache_dirty()
                self.load_current()

            # Build summary with per-pattern counts
            pattern_summary = ", ".join(f"{name}: {cnt}" for name, cnt in sorted(pattern_counts.items()))
            msg = f"Pattern sweep: {match_count} match(es) (scanned {scanned})"
            if pattern_summary:
                msg += f" — {pattern_summary}"
            self._safe_status_message(msg, 8000)

            # Mark pattern sweep as completed
            self.session.mark_sweep_done('patterns')

        runnable.signals.progress.connect(_on_progress)
        runnable.signals.failed.connect(_on_failed)
        runnable.signals.finished.connect(_on_finished)
        self.thread_pool.start(runnable)

    def _handle_quality_finished(self, blurry: int, motion: int) -> None:
        if blurry or motion:
            self.invalidate_overlay_cache()
            self._update_delete_button()
            self._update_stats_panel()
            self._mark_cache_dirty()
            self._safe_status_message(
                f"Quality sweep: blurry {blurry}, motion {motion}.",
                5000,
            )
        else:
            self._safe_status_message("Quality sweep finished: no candidates found.", 4000)

    # ============================================================================
    # DATASET LOADING & NAVIGATION
    # ============================================================================

    def _load_dataset(self, dir_path: Path) -> None:
        log_debug(f"_load_dataset called: {dir_path}", "VIEWER")

        # CRITICAL: Flush pending cache before switching datasets
        # This ensures outliers and other changes are saved before state.reset() clears them
        if self.session.cache_dirty:
            log_debug("Flushing cache before dataset switch", "VIEWER")
            self._flush_cache(wait=True)

        self.current_index = 0
        if not self.session.load(dir_path):
            log_warning(f"_load_dataset failed for: {dir_path}", "VIEWER")
            self._reset_runtime_state()
            self._show_empty_state()
            self.lwir_view.set_placeholder("No images found in lwir/ or visible/")
            self.vis_view.set_placeholder("No images found in lwir/ or visible/")
            return

        # Connect dataset handler to show save status
        handler = get_handler_registry().get_or_create(dir_path)
        handler.saveStatusChanged.connect(self._on_save_status_changed)

        # Sync sweep states FROM handler summary TO session state (summary is source of truth)
        # NOTE: "missing" is not tracked in SummaryCache — it's always auto-derived
        # on load by DatasetSession._auto_mark_missing_pairs(), so no sync needed.
        if handler.summary:
            self.session.state.cache_data["sweep_flags"]["duplicates"] = handler.summary.get_sweep_duplicates_done()
            self.session.state.cache_data["sweep_flags"]["quality"] = handler.summary.get_sweep_quality_done()
            self.session.state.cache_data["sweep_flags"]["patterns"] = handler.summary.get_sweep_patterns_done()

        # Initialize LabelService for this dataset
        self.label_service = LabelService(dataset_path=dir_path)
        # Config resolution: 1) workspace level, 2) user preference, 3) repo default
        config_path = self._workspace_label_config_path()
        if not config_path and self.label_yaml_path:
            candidate = Path(self.label_yaml_path)
            if candidate.exists():
                config_path = candidate
        if not config_path and self._default_label_yaml_path.exists():
            config_path = self._default_label_yaml_path
        if config_path:
            self.label_service.load_config(config_path)
            self._on_class_map_updated()
        else:
            self.label_yaml_path = None

        self.signature_manager.reset_epoch()
        self._reset_calibration_jobs()
        self.progress_tracker.clear()
        self._reset_queue_progress_state()
        self.calibration_workflow.prime_calibration_jobs(limit=config.calibration_prefetch_limit)
        # Run duplicate sweep only on first load when we have no cached signatures
        # AND sweep hasn't been marked as done
        sweep_done = self.session.state.cache_data["sweep_flags"].get("duplicates", False)
        if not self.session.state.signatures and not sweep_done:
            log_info("Auto-starting duplicate sweep (no signatures and not done before)", "VIEWER")
            self._prime_signature_scan()
        elif sweep_done:
            log_info("Skipping duplicate sweep (already done)", "VIEWER")
        else:
            log_info("Skipping duplicate sweep (signatures already exist)", "VIEWER")
        self.invalidate_overlay_cache()
        self.ui.btn_prev.setEnabled(True)
        self.ui.btn_next.setEnabled(True)
        self.ui.btn_prev_fast.setEnabled(True)
        self.ui.btn_next_fast.setEnabled(True)
        self.ui.btn_goto.setEnabled(True)
        self._update_dataset_window_title(force=True)
        self._update_dataset_label()
        self._update_delete_button()
        self._update_restore_menu()
        self._update_stats_panel()
        self._mark_cache_dirty()

        # Disable rectified view if no calibration available
        self._validate_rectified_after_load()

        # Apply saved filter mode - navigate to first matching image if needed
        if self.filter_controller.filter_mode != FILTER_ALL:
            first_match = self.filter_controller.reconcile_filter_state(show_warning=False)
            if first_match is not None:
                current_base = self.session.get_base(self.current_index)
                if not current_base or not self.filter_controller.filter_accepts_base(current_base):
                    self.current_index = first_match
        self.load_current()

    def _on_save_status_changed(self, status: str) -> None:
        """Show save status in status bar."""
        # Determine if we're showing dataset or collection
        entity_type = "Collection" if self.session.loaded_kind == "collection" else "Dataset"

        messages = {
            "pending": f"⏳ Pending {entity_type.lower()} save...",
            "saving": f"💾 Saving {entity_type.lower()}...",
            "saved": f"✓ {entity_type} saved"
        }
        msg = messages.get(status, status)
        duration = 2000 if status == "saved" else 0  # Permanent for pending/saving
        self._safe_status_message(msg, duration)

    def _load_collection(self, dir_path: Path) -> None:
        log_debug(f"_load_collection called: {dir_path}", "VIEWER")

        # CRITICAL: Flush pending cache before switching to collection
        # This ensures outliers and other changes are saved before state.reset() clears them
        if self.session.cache_dirty:
            log_debug("Flushing cache before collection switch", "VIEWER")
            self._flush_cache(wait=True)

        self.current_index = 0
        if not self.session.load_collection(dir_path):
            log_debug(f"_load_collection failed for: {dir_path}", "VIEWER")
            self._reset_runtime_state()
            self._show_empty_state()
            self.lwir_view.set_placeholder("No datasets found in collection")
            self.vis_view.set_placeholder("No datasets found in collection")
            return
        # Use a safe filter default for collections to avoid empty views if a prior filter was active.
        # NOTE: Collections don't show save status because they don't have a session.
        # Changes in collections are saved by their child datasets, not by the collection itself.

        self.filter_controller.filter_mode = FILTER_ALL
        self.filter_controller.reconcile_filter_state(show_warning=False)

        # Initialize LabelService for collection with default config
        # Collections store labels per-child-dataset; configure collection routing
        self.label_service = LabelService(dataset_path=dir_path)
        if self.session.collection:
            self.label_service.set_collection_children(
                dict(self.session.collection._child_dirs)
            )
        # Config resolution: 1) workspace level, 2) user preference, 3) repo default
        config_path = self._workspace_label_config_path()
        if not config_path and self.label_yaml_path:
            candidate = Path(self.label_yaml_path)
            if candidate.exists():
                config_path = candidate
        if not config_path and self._default_label_yaml_path.exists():
            config_path = self._default_label_yaml_path
        if config_path:
            self.label_service.load_config(config_path)
            log_debug(f"Loaded label config for collection: {config_path}", "VIEWER")
            self._on_class_map_updated()

        self.signature_manager.reset_epoch()
        self._reset_calibration_jobs()
        self.progress_tracker.clear()
        self._reset_queue_progress_state()
        self.calibration_workflow.prime_calibration_jobs(limit=config.calibration_prefetch_limit)
        # Collections: NO automatic sweeps - they are done via workspace sweep dialog
        log_debug("Collection loaded - sweeps must be done via workspace sweep dialog", "VIEWER")
        self.invalidate_overlay_cache()
        # Disable rectified view if no calibration available
        self._validate_rectified_after_load()
        self.load_current()
        self.ui.btn_prev.setEnabled(True)
        self.ui.btn_next.setEnabled(True)
        self.ui.btn_prev_fast.setEnabled(True)
        self.ui.btn_next_fast.setEnabled(True)
        self.ui.btn_goto.setEnabled(True)
        self._update_dataset_window_title(force=True)
        self._update_dataset_label()
        self._update_delete_button()
        self._update_restore_menu()
        self._update_stats_panel()
        self._mark_cache_dirty()
        if self.filter_controller.filter_mode != FILTER_ALL:
            self.filter_controller.reconcile_filter_state(show_warning=False)

    def _current_base(self) -> Optional[str]:
        return self.session.get_base(self.current_index)

    def _on_filter_changed(self) -> None:
        """Called when filter state changes (checkbox toggled). Reloads current image."""
        self.load_current()

    def _on_filter_mode_changed(self) -> None:
        """Called when filter mode changes. Persists preferences."""
        self.session.cache_service.set_preference("filter_mode", self.filter_controller.filter_mode)

    def _schedule_deferred_load(self, _index: int = 0) -> None:
        """Restart the navigation debounce timer.

        Called on every ``indexChanged`` signal.  Instead of loading the
        image immediately (which includes expensive alignment/overlay
        rendering), we (re)start a short timer.  When the user stops
        pressing arrow keys the timer fires and ``load_current`` runs
        exactly once for the final index.
        """
        self._nav_debounce_timer.start()  # restart if already running

    def load_current(self):
        base = self._current_base()
        if not base:
            return
        self.update_status(base)
        self.calibration_workflow.ensure_calibration_analysis(base)
        self.load_image_pair(base)
        self._update_stats_panel()
        self.signature_manager.schedule_index(self.current_index)
        if self._auto_label_active:
            self._run_auto_label_on_current()

    # ============================================================================
    # IMAGE DISPLAY & OVERLAYS
    # ============================================================================

    def update_status(self, base: str):
        total = self.session.total_pairs()
        if not total:
            self.setWindowTitle("Image Viewer")
            return
        dataset_name = Path(self.session.dataset_path).name if self.session.dataset_path else base
        if self.session.loaded_kind == "collection":
            title = f"Image Viewer - {dataset_name} ({self.current_index+1}/{total})"
        else:
            title = f"Image Viewer - {base} ({self.current_index+1}/{total})"
        filter_index, filter_total = self.navigation.filtered_position()
        if self.filter_controller.filter_mode != FILTER_ALL and filter_total:
            label = FILTER_STATUS_TITLES.get(self.filter_controller.filter_mode, "Filter")
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
        """Render overlayed image pair using OverlayOrchestrator."""
        # Sync view settings via setters (they handle cache invalidation if changed)
        self.overlay_orchestrator.set_view_rectified(self.view_rectified)
        self.overlay_orchestrator.set_align_mode(self._align_mode)
        self.overlay_orchestrator.set_grid_mode(self.view_state.grid_mode)
        self.overlay_orchestrator.set_show_labels(self.view_state.show_labels)
        self.overlay_orchestrator.set_show_overlays(self.view_state.show_overlays)
        self.overlay_orchestrator.set_corner_display_mode(self._corner_display_mode)

        return self.overlay_orchestrator.render_pair(base)

    def prev_image(self, step: int = 1):
        log_debug(f"prev_image called (step={step}). has_images={self.session.has_images()}, current_index={self.navigation.current_index}", "NAV")
        message_label = FILTER_MESSAGE_LABELS.get(self.filter_controller.filter_mode, "filtered images")
        result = self.navigation.prev(message_label, step=step)
        log_debug(f"prev_image result={result}, new_index={self.navigation.current_index}", "NAV")

    def next_image(self, step: int = 1):
        log_debug(f"next_image called (step={step}). has_images={self.session.has_images()}, current_index={self.navigation.current_index}", "NAV")
        message_label = FILTER_MESSAGE_LABELS.get(self.filter_controller.filter_mode, "filtered images")
        result = self.navigation.next(message_label, step=step)
        log_debug(f"next_image result={result}, new_index={self.navigation.current_index}", "NAV")

    def _handle_goto_image(self):
        """Open a dialog to jump to a specific image number."""
        total = self.session.total_pairs()
        if total <= 0:
            return
        from PyQt6.QtWidgets import QInputDialog
        current_1based = self.navigation.current_index + 1
        number, ok = QInputDialog.getInt(
            self, "Go to image", f"Image number (1 – {total}):",
            value=current_1based, min=1, max=total, step=1,
        )
        if ok:
            self.navigation.jump_to(number - 1)

    def _ensure_overlay_cached(self, base: str) -> None:
        # Silent check - don't show popup, just return if no images
        if not self.session.has_images():
            return
        self._render_overlayed_pair(base)

    def _update_metadata_panel(self, base: str, type_dir: str, widget):
        self.ui_helper.update_metadata_panel(base, type_dir, widget)

    def _has_calibration_data(self) -> bool:
        matrices = self.state.cache_data.get("_matrices") or {}
        return any(data for data in matrices.values() if data)

    def _validate_rectified_after_load(self) -> None:
        """Disable rectified view if no calibration is available for the loaded dataset."""
        if self.view_rectified and not self._has_calibration_data():
            self.view_rectified = False
            self.overlay_orchestrator.set_view_rectified(False)
            # Sync UI toggle
            action = getattr(self.ui, "action_toggle_rectified", None)
            if action:
                action.blockSignals(True)
                action.setChecked(False)
                action.blockSignals(False)
            self._safe_status_message(
                "Undistort disabled: no calibration data available for this dataset", 4000
            )
            log_info("Disabled rectified view: no calibration data in loaded dataset", "VIEWER")

    def _calibration_error_thresholds(self) -> Dict[str, float]:
        def _mad(values: List[float], center: float) -> float:
            if not values:
                return 0.0
            deviations = [abs(v - center) for v in values]
            return median(deviations) if deviations else 0.0

        def _threshold(values: List[float], floor: float) -> float:
            if not values:
                return 0.0
            center = median(values)
            mad_value = _mad(values, center)
            threshold = center + 2.5 * mad_value if mad_value > 0 else center * 1.5
            return max(threshold, floor)

        lwir_vals = [float(v) for v in self.state.cache_data["reproj_errors"].get("lwir", {}).values() if isinstance(v, (int, float))]
        vis_vals = [float(v) for v in self.state.cache_data["reproj_errors"].get("visible", {}).values() if isinstance(v, (int, float))]
        stereo_vals = [float(v) for v in self.state.cache_data["extrinsic_errors"].values() if isinstance(v, (int, float))]
        return {
            "lwir": _threshold(lwir_vals, 0.5),
            "visible": _threshold(vis_vals, 0.5),
            "stereo": _threshold(stereo_vals, 0.0),
        }

    def _persist_preferences(self, **kwargs: Any) -> None:
        self.session.cache_service.set_preferences(**kwargs)

    def _handle_save_status_action(self) -> None:
        self.progress_tracker.set_busy(config.progress_task_save, "Saving dataset state…")
        try:
            # Flush session cache
            self._flush_cache(wait=True)

            # Flush handler if exists (for summary cache)
            if self.session.dataset_path:
                handler = get_handler_registry().get(self.session.dataset_path)
                if handler:
                    handler.force_flush()

            self._safe_status_message("Dataset status saved.", 3000)
        finally:
            self.progress_tracker.finish(config.progress_task_save)

    def _handle_configure_label_model(self) -> None:
        """Let the user pick a detection model from the factory registry."""
        from backend.services.labels.detection.detector_factory import (
            list_available_models,
        )

        specs = list_available_models()
        display_names = [s.display_name for s in specs]

        choice, ok = QInputDialog.getItem(
            self, "Select Detection Model", "Model:", display_names, 0, False,
        )
        if not ok:
            return

        spec = specs[display_names.index(choice)]

        # If the model needs a weights file, ask for it
        file_path: str | None = None
        if spec.needs_file:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select model weights", self._workspace_root(),
                "Model (*.pt *.pth)",
            )
            if not path:
                return
            file_path = path

        # Must have label_service ready (and config for models that need it)
        if not self.label_service:
            QMessageBox.information(
                self, "Detection Model",
                "Load a dataset and configure a labels YAML first.",
            )
            return
        if spec.needs_config and not self.label_service.config:
            QMessageBox.information(
                self, "Detection Model",
                "This model requires a label schema.\n\n"
                "Configure a labels YAML first.",
            )
            return

        # Get image dimensions from the currently loaded dataset so the
        # detector can process at full resolution instead of downscaling.
        image_size: tuple[int, int] | None = None
        base = self._current_base()
        if base:
            ch = self._label_channel()
            image_size = self.session.get_original_image_size(base, ch)

        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self.label_service.load_detector(
                spec.key, file_path=file_path, image_size=image_size,
            )
            self.label_model_path = file_path or spec.key
            self._persist_preferences(label_model=self.label_model_path)
            self._safe_status_message(f"Model loaded: {spec.display_name}", 3000)
        except Exception as e:
            QMessageBox.warning(
                self, "Model Error", f"Failed to load model:\n{e}",
            )
        finally:
            QApplication.restoreOverrideCursor()

    def _handle_configure_label_yaml(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select labels YAML", self._workspace_root(), "YAML (*.yaml *.yml)")
        if not path:
            return
        self.label_yaml_path = path
        self._persist_preferences(label_yaml=path)
        # Copy config to workspace level (single authoritative location)
        if self.workspace_dir:
            import shutil
            from backend.services.labels.label_storage import WORKSPACE_CONFIG_FILENAME
            ws_config = Path(self.workspace_dir) / WORKSPACE_CONFIG_FILENAME
            shutil.copy2(path, str(ws_config))
        if not self.label_service and self.session.dataset_path:
            self.label_service = LabelService(dataset_path=Path(self.session.dataset_path))
        if self.label_service:
            self.label_service.load_config(Path(path))
            self._on_class_map_updated()
        self._safe_status_message("Label classes loaded and saved to workspace.", 3000)

    def _handle_reload_label_config(self) -> None:
        """Reload label config from source, updating workspace copy."""
        if not self.label_yaml_path:
            QMessageBox.information(
                self, "Reload Config",
                "No source label config configured.\n\n"
                "Use 'Configure labels YAML…' first to set a source config."
            )
            return

        source_path = Path(self.label_yaml_path)
        if not source_path.exists():
            QMessageBox.warning(
                self, "Reload Config",
                f"Source config not found:\n{source_path}\n\n"
                "Use 'Configure labels YAML…' to select a new source."
            )
            return

        if not self.label_service:
            QMessageBox.information(self, "Reload Config", "No dataset loaded.")
            return

        # Confirm overwrite of workspace config
        ws_config = self._workspace_label_config_path()
        if ws_config:
            reply = QMessageBox.question(
                self, "Reload Config",
                f"This will reload the workspace label config from:\n{source_path}\n\n"
                "Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Force reload and copy to workspace
        self.label_service.load_config(source_path)
        if self.workspace_dir:
            import shutil
            from backend.services.labels.label_storage import WORKSPACE_CONFIG_FILENAME
            ws_dst = Path(self.workspace_dir) / WORKSPACE_CONFIG_FILENAME
            shutil.copy2(str(source_path), str(ws_dst))
        self._on_class_map_updated()
        self._safe_status_message("Label config reloaded from source.", 3000)

    def _ensure_label_service(self) -> bool:
        """Ensure label service is ready for detection operations."""
        if not self.label_service:
            return False
        if not self.label_service.is_ready:
            return False
        return True

    def _label_channel(self) -> str:
        return "lwir" if self.label_input_mode == "lwir" else "visible"

    def _label_image_path(self, base: str) -> Optional[Path]:
        return self.session.get_image_path(base, self._label_channel())

    def _handle_toggle_detection_channel(self) -> None:
        """Toggle detection input between visible and lwir."""
        if self.label_input_mode == "lwir":
            self.label_input_mode = "visible"
        else:
            self.label_input_mode = "lwir"
        self._persist_preferences(label_input_mode=self.label_input_mode)
        self._sync_detection_channel_action()
        self._safe_status_message(
            f"Detection channel set to: {self.label_input_mode}", 3000,
        )

    def _sync_detection_channel_action(self) -> None:
        """Update menu action text to reflect current channel."""
        if hasattr(self.ui, "action_label_detection_channel"):
            ch = self.label_input_mode.upper()
            self.ui.action_label_detection_channel.setText(
                f"Detection channel: {ch}"
            )

    def _update_labeling_views(self) -> None:
        self.vis_view.set_labeling_mode(self._manual_label_mode)
        self.lwir_view.set_labeling_mode(self._manual_label_mode)
        self.vis_view.set_auto_label_mode(self._auto_label_active)
        self.lwir_view.set_auto_label_mode(self._auto_label_active)

    def _handle_manual_label_mode_toggle(self, enabled: bool) -> None:
        if enabled and not require_images(self, "Manual labels"):
            self._manual_label_mode = False
            self._update_labeling_views()
            self._sync_action_states()
            return
        if enabled and self._auto_label_active:
            self._toggle_auto_label_mode(False)
        self._manual_label_mode = enabled
        self._update_labeling_views()
        self._sync_action_states()
        if enabled:
            if not self.view_state.show_labels:
                self.view_state.show_labels = True
                self._persist_preferences(show_labels=True)
            self._safe_status_message(
                "Manual label mode: click two corners (rubber band), right-click to delete a box, Esc cancels.",
                5000,
            )
        else:
            self._safe_status_message("Manual label mode off.", 2000)

    def _handle_auto_label_mode_toggle(self, enabled: bool) -> None:
        """Handle auto-labelling mode toggle from menu / context menu."""
        if enabled and not require_images(self, "Auto labelling"):
            self._auto_label_active = False
            self._sync_action_states()
            return
        if enabled:
            # Ensure detector is configured; if not, open model picker directly
            if not self._ensure_label_service() or not self.label_service:
                QMessageBox.information(
                    self, "Auto Labelling",
                    "Configure a labels YAML first.",
                )
                self._auto_label_active = False
                self._sync_action_states()
                return
            if not self.label_service.has_detector:
                self._auto_label_active = False
                self._sync_action_states()
                self._handle_configure_label_model()
                # Re-check after model selection
                if not self.label_service or not self.label_service.has_detector:
                    return
                self._auto_label_active = True
                self._sync_action_states()
            # Deactivate manual mode if it was active (mutual exclusion)
            if self._manual_label_mode:
                self._toggle_manual_label_mode(False)
        self._auto_label_active = enabled
        self._update_labeling_views()
        self._sync_action_states()
        if enabled:
            self._auto_detected_bases.clear()
            if not self.view_state.show_labels:
                self.view_state.show_labels = True
                self._persist_preferences(show_labels=True)
            self._safe_status_message(
                "Auto label mode: detection runs on every navigation. Right-click labels to edit/delete.",
                5000,
            )
            # Run detection on the current image immediately
            self._run_auto_label_on_current()
        else:
            self._safe_status_message("Auto label mode off.", 2000)

    def _toggle_auto_label_mode(self, enabled: bool) -> None:
        """Toggle auto-labelling mode programmatically."""
        if hasattr(self.ui, "action_label_auto_mode"):
            self.ui.action_label_auto_mode.setChecked(enabled)

    def _run_auto_label_on_current(self) -> None:
        """Run detection on the current image (silent, no popups).

        Called automatically in auto-label mode after every navigation.
        Each (base, channel) is processed at most once per auto-label
        session so that user deletions are not overwritten on re-visit.
        """
        if not self._auto_label_active:
            return
        if not self._ensure_label_service() or not self.label_service:
            return
        if not self.label_service.has_detector:
            return
        base = self._current_base()
        if not base:
            return
        channel = self._label_channel()
        # Skip images already processed in this auto-label session
        key = (base, channel)
        if key in self._auto_detected_bases:
            return
        img_path = self._label_image_path(base)
        if not img_path or not img_path.exists():
            return
        import cv2
        image = cv2.imread(str(img_path))
        if image is None:
            return
        task_id = config.progress_task_label_detect
        self.progress_tracker.set_busy(task_id, "Auto-detecting\u2026")
        QApplication.processEvents()
        try:
            annotations = self.label_service.auto_detect(base, channel, image)
        finally:
            self.progress_tracker.finish(task_id)
        self._auto_detected_bases.add(key)
        if annotations:
            self._safe_status_message(
                f"Auto-detected {len(annotations)} objects.", 2000,
            )
            self.invalidate_overlay_cache(base)
            self.load_image_pair(base)

    def _handle_manual_selection_canceled(self, channel: str) -> None:  # noqa: ARG002
        if self._manual_label_mode:
            self._safe_status_message("Label selection cancelled.", 2000)

    def _transform_display_to_original_coords(
        self,
        channel: str,
        base: str,
        left: float,
        top: float,
        right: float,
        bottom: float,
    ) -> Tuple[float, float, float, float]:
        """Transform bbox from display coordinates back to original image coordinates.

        When stereo alignment or undistort is active, the displayed image is transformed.
        Annotations are always stored relative to the ORIGINAL image, so we need
        to apply the inverse transform when saving manually-drawn boxes.

        IMPORTANT: This must be the inverse of _transform_original_to_display_coords
        and must reverse the transformation done by overlay_workflow._transform_label_boxes.

        Args:
            channel: 'lwir' or 'visible'
            base: Image base name
            left, top, right, bottom: Normalized coordinates in display space [0,1]

        Returns:
            (left, top, right, bottom) in original image normalized coords [0,1]
        """
        import numpy as np

        # Check if any transform is active
        if self._align_mode == "disabled" and not self.view_rectified:
            return (left, top, right, bottom)

        # Get alignment transform from state
        alignment_transform = self.state.cache_data.get("_alignment_transform")

        if alignment_transform is None and not self.view_rectified:
            return (left, top, right, bottom)

        # Get sizes
        original_size = self.session.get_original_image_size(base, channel)
        if not original_size:
            log_debug("No original size available for inverse transform", "MANUAL_LABEL")
            return (left, top, right, bottom)

        orig_w, orig_h = original_size

        # Get display size from alignment transform
        if alignment_transform and alignment_transform.output_size:
            disp_w, disp_h = alignment_transform.output_size
        else:
            disp_w, disp_h = orig_w, orig_h

        # Get camera matrices if available (for undistort)
        cd = self.state.cache_data
        matrices = cd.get("_matrices") or {}
        chan_matrices = matrices.get(channel) or {}
        camera_matrix = chan_matrices.get("camera_matrix")
        distortion = chan_matrices.get("distortion")

        # Convert normalized display coords to display pixels
        corners_disp = np.array([
            [left * disp_w, top * disp_h],
            [right * disp_w, top * disp_h],
            [right * disp_w, bottom * disp_h],
            [left * disp_w, bottom * disp_h],
        ], dtype=np.float32)

        # Apply inverse transform (reverse order of forward: alignment^-1 then redistort)
        if alignment_transform:
            # Step 1: Inverse alignment matrix
            if channel == "visible":
                corners_undist = alignment_transform.transform_vis_points_inverse(corners_disp)
            else:
                corners_undist = alignment_transform.transform_lwir_points_inverse(corners_disp)

            # Step 2: If undistort was applied, we need to "redistort" the points
            # This is an approximation - exact inverse undistort requires iteration
            if self.view_rectified and camera_matrix is not None and distortion is not None:
                try:
                    import cv2
                    cam = np.array(camera_matrix, dtype=np.float32)
                    dist = np.array(distortion, dtype=np.float32).reshape(-1)
                    # We have undistorted coords, need to apply distortion back
                    # Using iterative projectPoints approach
                    corners_orig = self._redistort_points(corners_undist, cam, dist, (orig_w, orig_h))
                except Exception as e:
                    log_debug(f"Redistortion failed, using linear approx: {e}", "MANUAL_LABEL")
                    corners_orig = corners_undist
            else:
                corners_orig = corners_undist
        elif self.view_rectified and camera_matrix is not None and distortion is not None:
            # Only undistort active (no alignment) - need to redistort
            try:
                import cv2
                cam = np.array(camera_matrix, dtype=np.float32)
                dist = np.array(distortion, dtype=np.float32).reshape(-1)
                corners_orig = self._redistort_points(corners_disp, cam, dist, (orig_w, orig_h))
            except Exception:
                corners_orig = corners_disp
        else:
            corners_orig = corners_disp

        # Convert back to normalized original coords
        corners_orig = corners_orig.astype(np.float32)
        corners_orig[:, 0] /= orig_w
        corners_orig[:, 1] /= orig_h

        # Clamp to [0, 1]
        corners_orig = np.clip(corners_orig, 0.0, 1.0)

        # Extract bbox bounds
        new_left = float(corners_orig[:, 0].min())
        new_right = float(corners_orig[:, 0].max())
        new_top = float(corners_orig[:, 1].min())
        new_bottom = float(corners_orig[:, 1].max())

        log_debug(
            f"Inverse transform [{channel}]: display ({left:.3f},{top:.3f},{right:.3f},{bottom:.3f}) "
            f"-> original ({new_left:.3f},{new_top:.3f},{new_right:.3f},{new_bottom:.3f})",
            "MANUAL_LABEL"
        )

        return (new_left, new_top, new_right, new_bottom)

    def _redistort_points(
        self,
        points: "np.ndarray",
        camera_matrix: "np.ndarray",
        distortion: "np.ndarray",
        image_size: Tuple[int, int],
    ) -> "np.ndarray":
        """Apply distortion to undistorted points (inverse of undistortPoints).

        This is an approximation that works well for points near the image center.
        For exact results, an iterative approach would be needed.

        Args:
            points: Nx2 array of undistorted pixel coordinates
            camera_matrix: 3x3 camera intrinsic matrix
            distortion: Distortion coefficients
            image_size: (width, height) of original image

        Returns:
            Nx2 array of distorted (original) pixel coordinates
        """
        import numpy as np
        import cv2

        # Get the new camera matrix that was used for undistortion
        new_cam, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix, distortion, image_size, 1, image_size
        )

        # Normalize points using the new camera matrix
        fx, fy = new_cam[0, 0], new_cam[1, 1]
        cx, cy = new_cam[0, 2], new_cam[1, 2]

        # Convert to normalized coordinates
        x_norm = (points[:, 0] - cx) / fx
        y_norm = (points[:, 1] - cy) / fy

        # Apply distortion model
        k1, k2, p1, p2 = distortion[0], distortion[1], distortion[2], distortion[3]
        k3 = distortion[4] if len(distortion) > 4 else 0

        r2 = x_norm**2 + y_norm**2
        r4 = r2**2
        r6 = r4 * r2

        # Radial distortion
        radial = 1 + k1*r2 + k2*r4 + k3*r6

        # Tangential distortion
        x_dist = x_norm * radial + 2*p1*x_norm*y_norm + p2*(r2 + 2*x_norm**2)
        y_dist = y_norm * radial + p1*(r2 + 2*y_norm**2) + 2*p2*x_norm*y_norm

        # Convert back to pixel coordinates using original camera matrix
        fx_orig, fy_orig = camera_matrix[0, 0], camera_matrix[1, 1]
        cx_orig, cy_orig = camera_matrix[0, 2], camera_matrix[1, 2]

        x_pixel = x_dist * fx_orig + cx_orig
        y_pixel = y_dist * fy_orig + cy_orig

        return np.column_stack([x_pixel, y_pixel])

    def _transform_original_to_display_coords(
        self,
        channel: str,
        base: str,
        left: float,
        top: float,
        right: float,
        bottom: float,
    ) -> Tuple[float, float, float, float]:
        """Transform bbox from original image coordinates to display coordinates.

        When stereo alignment or undistort is active, the displayed image is transformed.
        This converts annotation coords (stored in original space) to display space.

        IMPORTANT: This method must be the exact inverse of _transform_display_to_original_coords
        and must match the transformation used by overlay_workflow._transform_label_boxes.

        Args:
            channel: 'lwir' or 'visible'
            base: Image base name
            left, top, right, bottom: Normalized coordinates in original space [0,1]

        Returns:
            (left, top, right, bottom) in display normalized coords [0,1]
        """
        import numpy as np

        # Check if any transform is active
        if self._align_mode == "disabled" and not self.view_rectified:
            return (left, top, right, bottom)

        # Get alignment transform from state
        alignment_transform = self.state.cache_data.get("_alignment_transform")

        if alignment_transform is None and not self.view_rectified:
            return (left, top, right, bottom)

        # Get sizes
        original_size = self.session.get_original_image_size(base, channel)
        if not original_size:
            return (left, top, right, bottom)

        orig_w, orig_h = original_size

        # Get display size from alignment transform
        if alignment_transform and alignment_transform.output_size:
            disp_w, disp_h = alignment_transform.output_size
        else:
            disp_w, disp_h = orig_w, orig_h

        # Get camera matrices if available (for undistort)
        cd = self.state.cache_data
        matrices = cd.get("_matrices") or {}
        chan_matrices = matrices.get(channel) or {}
        camera_matrix = chan_matrices.get("camera_matrix")
        distortion = chan_matrices.get("distortion")

        # Build normalized corners array (same format as _transform_label_boxes)
        corners_norm = np.array([
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
        ], dtype=np.float32)

        # Use the same complete transformation as overlay_workflow._transform_label_boxes
        if alignment_transform:
            if channel == "visible":
                corners_disp = alignment_transform.transform_vis_corners_complete(
                    corners_norm,
                    original_size,
                    self.view_rectified,
                    camera_matrix,
                    distortion,
                )
            else:
                corners_disp = alignment_transform.transform_lwir_corners_complete(
                    corners_norm,
                    original_size,
                    self.view_rectified,
                    camera_matrix,
                    distortion,
                )
            # corners_disp is now in output pixel coords
        elif self.view_rectified and camera_matrix is not None and distortion is not None:
            # Only undistort, no alignment
            try:
                import cv2
                cam = np.array(camera_matrix, dtype=np.float32)
                dist = np.array(distortion, dtype=np.float32).reshape(-1)
                new_cam, _ = cv2.getOptimalNewCameraMatrix(
                    cam, dist, (orig_w, orig_h), 1, (orig_w, orig_h)
                )
                # Denormalize to pixel coords first
                corners_px = corners_norm.copy()
                corners_px[:, 0] *= orig_w
                corners_px[:, 1] *= orig_h
                pts_reshaped = corners_px.reshape(-1, 1, 2)
                undistorted = cv2.undistortPoints(pts_reshaped, cam, dist, P=new_cam)
                corners_disp = undistorted.reshape(-1, 2)
            except Exception:
                corners_disp = corners_norm.copy()
                corners_disp[:, 0] *= orig_w
                corners_disp[:, 1] *= orig_h
        else:
            corners_disp = corners_norm.copy()
            corners_disp[:, 0] *= orig_w
            corners_disp[:, 1] *= orig_h

        # Convert to normalized display coords
        corners_disp = corners_disp.astype(np.float32)
        corners_disp[:, 0] /= disp_w
        corners_disp[:, 1] /= disp_h

        # Clamp to [0, 1]
        corners_disp = np.clip(corners_disp, 0.0, 1.0)

        # Extract bbox bounds
        new_left = float(corners_disp[:, 0].min())
        new_right = float(corners_disp[:, 0].max())
        new_top = float(corners_disp[:, 1].min())
        new_bottom = float(corners_disp[:, 1].max())

        return (new_left, new_top, new_right, new_bottom)

    def _handle_manual_box_defined(self, channel: str, left: float, top: float, right: float, bottom: float) -> None:
        """Handle manual label box definition.

        The coordinates come in DISPLAY space (potentially aligned/undistorted).
        The dialog works in DISPLAY space for intuitive editing.
        Transformation to ORIGINAL image coordinates happens only when saving.
        """
        log_debug(f"box_defined signal: channel={channel}, display_coords=({left:.3f},{top:.3f},{right:.3f},{bottom:.3f})", "MANUAL_LABEL")
        if not self._manual_label_mode:
            log_debug("Ignored: manual_label_mode=False", "MANUAL_LABEL")
            return
        base = self._current_base()
        if not base or not self.session.dataset_path:
            log_debug(f"Ignored: base={base}, dataset_path={self.session.dataset_path}", "MANUAL_LABEL")
            return

        # Keep display coordinates for dialog (user edits what they see)
        display_left, display_top, display_right, display_bottom = left, top, right, bottom

        # Clamp coordinates to [0, 1]
        coords = [max(0.0, min(1.0, c)) for c in (display_left, display_top, display_right, display_bottom)]
        display_left, display_top, display_right, display_bottom = coords
        width = display_right - display_left
        height = display_bottom - display_top
        if width <= 0 or height <= 0:
            self._safe_status_message("Ignored zero-area box.", 2000)
            return

        x_center = display_left + width / 2
        y_center = display_top + height / 2

        if not self.label_service:
            log_debug("No label_service available", "MANUAL_LABEL")
            return

        log_debug(f"label_service.config present: {self.label_service.config is not None}", "MANUAL_LABEL")

        # Use full edit dialog for new annotations
        from frontend.widgets.label_edit_dialog import LabelEditDialog

        # Dialog works in DISPLAY coordinates (what user sees)
        bbox = (x_center, y_center, width, height)

        # Get the view for live preview
        view = self.lwir_view if channel == "lwir" else self.vis_view

        # Use VIEW size (display), not original image size
        view_size = view.get_pixmap_size()
        if not view_size:
            view_size = self.session.get_original_image_size(base, channel)

        # Callback for live bbox preview (in display coords)
        def on_bbox_changed(x1: float, y1: float, x2: float, y2: float) -> None:
            view.set_edit_highlight((x1, y1, x2, y2))

        # Set initial highlight
        on_bbox_changed(display_left, display_top, display_right, display_bottom)

        try:
            result = LabelEditDialog.new_annotation(
                self, self.label_service.config, bbox,
                image_size=view_size,
                on_bbox_changed=on_bbox_changed,
            )
        finally:
            # Clear highlight when dialog closes
            view.clear_edit_highlight()

        if not result:
            self._safe_status_message("Label entry cancelled.", 2000)
            return

        cls_value = result.get("class_id")
        display_bbox = result.get("bbox", bbox)  # In DISPLAY coordinates
        attributes = result.get("attributes", {})

        resolved_cls = self.label_service.class_id_for_value(cls_value) if cls_value else None
        log_debug(f"Resolved class: '{cls_value}' -> '{resolved_cls}'", "MANUAL_LABEL")

        if resolved_cls is None:
            QMessageBox.warning(self, "Label class", "Select a class from the list or type a valid name/id.")
            return

        # Transform display bbox to original coordinates for storage
        dx, dy, dw, dh = display_bbox
        display_left = dx - dw / 2
        display_top = dy - dh / 2
        display_right = dx + dw / 2
        display_bottom = dy + dh / 2

        orig_left, orig_top, orig_right, orig_bottom = self._transform_display_to_original_coords(
            channel, base, display_left, display_top, display_right, display_bottom
        )

        # Convert back to center format
        orig_width = orig_right - orig_left
        orig_height = orig_bottom - orig_top
        orig_x = orig_left + orig_width / 2
        orig_y = orig_top + orig_height / 2

        log_debug(f"Final original coords: center=({orig_x:.3f},{orig_y:.3f}) size=({orig_width:.3f},{orig_height:.3f})", "MANUAL_LABEL")

        self.label_service.add_manual_box(
            base, channel, resolved_cls, orig_x, orig_y, orig_width, orig_height, attributes
        )
        log_info(
            f"Added manual label: {channel}/{base} class={resolved_cls} "
            f"box=({orig_x:.3f},{orig_y:.3f},{orig_width:.3f},{orig_height:.3f}) attrs={attributes}",
            "MANUAL_LABEL"
        )

        self.invalidate_overlay_cache(base)
        self.load_current()
        self._safe_status_message(f"Added label to {channel}:{base}.", 3000)

    def _handle_manual_delete_request(self, channel: str, x_norm: float, y_norm: float, global_pos) -> None:
        if not self._manual_label_mode and not self._auto_label_active:
            return
        base = self._current_base()
        if not base or not self.session.dataset_path:
            return

        # Build context menu
        menu = QMenu(self)

        # Collect ALL overlapping annotations at click point
        overlapping: list = []
        if self.label_service:
            overlapping = self.label_service.find_all_annotations_at(
                base, channel, x_norm, y_norm
            )

        # --- Helper to format display name ---
        _HIDDEN = {"model", "model_version", "raw_label", "confidence", "source",
                    "occlusion", "truncation"}

        def _display_name(ann, short: bool = False) -> str:
            cls_name = ann.class_id
            if self.label_service and self.label_service.config:
                cls_def = self.label_service.config.classes.get(ann.class_id)
                if cls_def:
                    cls_name = cls_def.name
            label = f"{ann.class_id}: {cls_name}" if ann.class_id != cls_name else cls_name
            if not short:
                # Append class-specific attributes for disambiguation
                visible_attrs = [
                    str(v) for k, v in ann.attributes.items()
                    if k not in _HIDDEN and v not in (None, "", "unknown")
                ]
                if visible_attrs:
                    label += f"  ({', '.join(visible_attrs)})"
                # Confidence for auto detections
                if ann.source.value == "auto" and ann.confidence < 1.0:
                    label += f"  {ann.confidence:.0%}"
            return label

        # Build edit / delete / accept actions for each overlapping annotation
        edit_actions: list = []    # (QAction, idx, annotation)
        delete_actions: list = []  # (QAction, idx, annotation)
        accept_actions: list = []  # (QAction, idx, annotation)

        # Determine which annotations are unreviewed (source=AUTO)
        auto_anns = [ann for _, ann in overlapping if ann.source.value == "auto"]

        if len(overlapping) == 1:
            # Single annotation — flat menu (same UX as before)
            idx, ann = overlapping[0]
            cls_display = _display_name(ann)
            if ann.source.value == "auto":
                aa = menu.addAction(f"✓ Accept label  {cls_display}")
                accept_actions.append((aa, idx, ann))
            ea = menu.addAction(f"Edit label  {cls_display}")
            da = menu.addAction(f"Delete label  {cls_display}")
            edit_actions.append((ea, idx, ann))
            delete_actions.append((da, idx, ann))
            # Accept-all shortcut when there are auto annotations
            if auto_anns:
                menu.addSeparator()
                accept_all_action = menu.addAction(f"✓ Accept all labels on this image")
            else:
                accept_all_action = None
            menu.addSeparator()
        elif len(overlapping) > 1:
            # Multiple overlapping annotations — sub-menus
            # Accept sub-menu (only shown if there are auto annotations)
            if auto_anns:
                accept_menu = menu.addMenu(f"✓ Accept label  ({len(auto_anns)} pending)")
                for idx, ann in overlapping:
                    if ann.source.value == "auto":
                        cls_display = _display_name(ann)
                        aa = accept_menu.addAction(cls_display)
                        accept_actions.append((aa, idx, ann))
            edit_menu = menu.addMenu(f"Edit label  ({len(overlapping)} overlapping)")
            delete_menu = menu.addMenu(f"Delete label  ({len(overlapping)} overlapping)")
            for idx, ann in overlapping:
                cls_display = _display_name(ann)
                ea = edit_menu.addAction(cls_display)
                da = delete_menu.addAction(cls_display)
                edit_actions.append((ea, idx, ann))
                delete_actions.append((da, idx, ann))
            if auto_anns:
                menu.addSeparator()
                accept_all_action = menu.addAction(f"✓ Accept all labels on this image")
            else:
                accept_all_action = None
            menu.addSeparator()
        else:
            accept_all_action = None

        # Always show "Accept all" when there are unreviewed AUTO labels
        # on this image, even if no annotation was under the cursor
        if accept_all_action is None and self.label_service and self.label_service.has_unreviewed(base, channel):
            accept_all_action = menu.addAction("✓ Accept all labels on this image")
            menu.addSeparator()

        # Mode switching and exit actions
        switch_action = None
        if self._manual_label_mode:
            switch_action = menu.addAction("Switch to auto labelling mode")
            if switch_action is not None:
                switch_action.setShortcut("Ctrl+Shift+L")
                switch_action.setShortcutVisibleInContextMenu(True)
            exit_mode_action = menu.addAction("Exit manual labelling mode")
            exit_mode_action.setShortcut("Ctrl+L")
            exit_mode_action.setShortcutVisibleInContextMenu(True)
        elif self._auto_label_active:
            switch_action = menu.addAction("Switch to manual labelling mode")
            if switch_action is not None:
                switch_action.setShortcut("Ctrl+L")
                switch_action.setShortcutVisibleInContextMenu(True)
            exit_mode_action = menu.addAction("Exit auto labelling mode")
            exit_mode_action.setShortcut("Ctrl+Shift+L")
            exit_mode_action.setShortcutVisibleInContextMenu(True)
        else:
            exit_mode_action = None
        menu.addSeparator()
        menu.addAction("Cancel")

        chosen = menu.exec(global_pos)

        # --- Dispatch accept actions (individual) ---
        for aa, idx, ann in accept_actions:
            if chosen is aa and self.label_service:
                count = self.label_service.mark_reviewed(base, channel, indices=[idx])
                if count:
                    self.invalidate_overlay_cache(base)
                    self.load_current()
                    self._safe_status_message(f"Accepted label {_display_name(ann)}.", 2000)
                return

        # --- Dispatch accept-all ---
        if accept_all_action is not None and chosen is accept_all_action and self.label_service:
            count = self.label_service.mark_all_reviewed(base, channel)
            if count:
                self.invalidate_overlay_cache(base)
                self.load_current()
                self._safe_status_message(f"Accepted {count} label(s).", 2000)
            return

        # --- Dispatch edit actions ---
        for ea, idx, ann in edit_actions:
            if chosen is ea:
                self._edit_annotation(base, channel, ann)
                return

        # --- Dispatch delete actions ---
        for da, idx, ann in delete_actions:
            if chosen is da and self.label_service:
                removed = self.label_service.remove_annotation(base, channel, idx)
                if removed:
                    self.invalidate_overlay_cache(base)
                    self.load_current()
                    self._safe_status_message(f"Deleted label {_display_name(ann)}.", 2000)
                return

        # --- Mode switching ---
        if chosen is switch_action:
            if self._manual_label_mode:
                self._toggle_manual_label_mode(False)
                self._toggle_auto_label_mode(True)
            elif self._auto_label_active:
                self._toggle_auto_label_mode(False)
                self._toggle_manual_label_mode(True)
        elif chosen is exit_mode_action:
            if self._manual_label_mode:
                self._toggle_manual_label_mode(False)
            elif self._auto_label_active:
                self._toggle_auto_label_mode(False)

    def _edit_annotation(self, base: str, channel: str, annotation) -> None:
        """Open dialog to edit an existing annotation.

        Annotation bbox is stored in ORIGINAL coords, but we display/edit in DISPLAY coords.
        """
        from frontend.widgets.label_edit_dialog import LabelEditDialog

        if not self.label_service:
            return

        config = self.label_service.config

        # Get the view for the channel being edited
        view = self.lwir_view if channel == "lwir" else self.vis_view

        # Use VIEW size (display), not original image size
        view_size = view.get_pixmap_size()
        if not view_size:
            view_size = self.session.get_original_image_size(base, channel)

        # Transform annotation bbox from ORIGINAL to DISPLAY coords
        orig_bbox = annotation.bbox  # (x_center, y_center, width, height) in original coords
        ox, oy, ow, oh = orig_bbox
        orig_left = ox - ow / 2
        orig_top = oy - oh / 2
        orig_right = ox + ow / 2
        orig_bottom = oy + oh / 2

        disp_left, disp_top, disp_right, disp_bottom = self._transform_original_to_display_coords(
            channel, base, orig_left, orig_top, orig_right, orig_bottom
        )

        # Convert to center format for dialog
        disp_w = disp_right - disp_left
        disp_h = disp_bottom - disp_top
        disp_x = disp_left + disp_w / 2
        disp_y = disp_top + disp_h / 2
        display_bbox = (disp_x, disp_y, disp_w, disp_h)

        # Callback for live bbox preview (in display coords)
        def on_bbox_changed(x1: float, y1: float, x2: float, y2: float) -> None:
            view.set_edit_highlight((x1, y1, x2, y2))

        # Set initial highlight
        on_bbox_changed(disp_left, disp_top, disp_right, disp_bottom)

        # Create a modified annotation with display bbox for the dialog
        import dataclasses
        display_annotation = dataclasses.replace(annotation, bbox=display_bbox)

        try:
            result = LabelEditDialog.edit_annotation(
                self, config, display_annotation,
                image_size=view_size,
                on_bbox_changed=on_bbox_changed,
            )
        finally:
            # Clear highlight when dialog closes
            view.clear_edit_highlight()

        if result:
            # Handle delete request from the edit dialog
            if result.get("deleted"):
                if self.label_service:
                    # Find list index for this annotation_id
                    anns = self.label_service.get_annotations(base, channel)
                    idx = next(
                        (i for i, a in enumerate(anns)
                         if a.annotation_id == annotation.annotation_id),
                        None,
                    )
                    if idx is not None:
                        removed = self.label_service.remove_annotation(
                            base, channel, idx,
                        )
                        if removed:
                            self.invalidate_overlay_cache(base)
                            self.load_current()
                            self._safe_status_message("Deleted annotation.", 2000)
                return

            cls_value = result.get("class_id")
            new_display_bbox = result.get("bbox")  # In DISPLAY coordinates
            new_attrs = result.get("attributes", {})

            # Validate class_id against config (same as new_annotation flow)
            new_class_id = self.label_service.class_id_for_value(cls_value) if cls_value else None
            if new_class_id is None:
                QMessageBox.warning(self, "Label class", "Select a class from the list or type a valid name/id.")
                return

            # Transform display bbox back to original coordinates for storage
            if new_display_bbox:
                dx, dy, dw, dh = new_display_bbox
                disp_left = dx - dw / 2
                disp_top = dy - dh / 2
                disp_right = dx + dw / 2
                disp_bottom = dy + dh / 2

                orig_left, orig_top, orig_right, orig_bottom = self._transform_display_to_original_coords(
                    channel, base, disp_left, disp_top, disp_right, disp_bottom
                )

                # Convert back to center format
                new_orig_w = orig_right - orig_left
                new_orig_h = orig_bottom - orig_top
                new_orig_x = orig_left + new_orig_w / 2
                new_orig_y = orig_top + new_orig_h / 2
                new_bbox = (new_orig_x, new_orig_y, new_orig_w, new_orig_h)
            else:
                new_bbox = None

            # Update the annotation
            if self.label_service.update_annotation(
                base, channel, annotation.annotation_id, new_class_id, new_attrs, new_bbox
            ):
                # Auto-accept: editing an AUTO annotation implies review
                if annotation.source.value == "auto":
                    self.label_service.accept_annotation(
                        base, channel, annotation.annotation_id,
                    )
                self.invalidate_overlay_cache(base)
                self.load_current()
                self._safe_status_message(f"Updated label to class {new_class_id}.", 2000)

    def _toggle_manual_label_mode(self, enabled: bool) -> None:
        """Toggle manual labelling mode programmatically."""
        if hasattr(self.ui, "action_label_manual_mode"):
            self.ui.action_label_manual_mode.setChecked(enabled)

    def _handle_label_current(self) -> None:
        if not require_images(self, "Labelling"):
            return
        base = self._current_base()
        if not base:
            return
        if not self._ensure_label_service() or not self.label_service:
            QMessageBox.information(self, "Labelling", "Configure a model and labels YAML first.")
            return
        if not self.label_service.has_detector:
            self._handle_configure_label_model()
            if not self.label_service or not self.label_service.has_detector:
                return
        img_path = self._label_image_path(base)
        if not img_path or not img_path.exists():
            QMessageBox.information(self, "Labelling", "No image available for the selected channel.")
            return
        channel = self._label_channel()
        import cv2
        image = cv2.imread(str(img_path))
        if image is None:
            QMessageBox.warning(self, "Labelling", "Failed to load image.")
            return
        # Show progress while detection runs (blocks the UI thread but
        # the progress bar gives visual feedback before the heavy work).
        task_id = config.progress_task_label_detect
        self.progress_tracker.set_busy(task_id, "Running detection model\u2026")
        QApplication.processEvents()  # flush so the bar appears
        try:
            annotations = self.label_service.auto_detect(base, channel, image)
        finally:
            self.progress_tracker.finish(task_id)
        self._safe_status_message(f"Detected {len(annotations)} objects for {channel}:{base}.", 3000)
        self.invalidate_overlay_cache(base)
        self.load_current()

    def _handle_label_dataset(self) -> None:
        if not require_images(self, "Labelling"):
            return
        if not self._ensure_label_service() or not self.label_service:
            QMessageBox.information(self, "Labelling", "Configure a model and labels YAML first.")
            return
        if not self.label_service.has_detector:
            self._handle_configure_label_model()
            if not self.label_service or not self.label_service.has_detector:
                return
        loader = self.session.loader
        if loader is None:
            return
        channel = self._label_channel()
        bases = list(loader.image_bases)
        total = len(bases)
        task_id = self.config.progress_task_label_dataset
        self.progress_tracker.start(task_id, "Labelling dataset…", total)
        QApplication.processEvents()

        import cv2

        def _load_image(base: str, ch: str):
            img_path = loader.get_image_path(base, ch)
            if not img_path or not img_path.exists():
                return None
            return cv2.imread(str(img_path))

        def _on_progress(current: int, total_count: int, _base: str) -> None:
            self.progress_tracker.update(task_id, current)
            if current % 5 == 0:
                QApplication.processEvents()

        try:
            results = self.label_service.auto_detect_batch(
                bases, channel, _load_image, progress_callback=_on_progress,
            )
        finally:
            self.progress_tracker.finish(task_id)
        done = sum(1 for v in results.values() if v > 0)
        self.label_service.save_all()
        self._safe_status_message(f"Labelling finished for channel {channel} ({done}/{total}).", 4000)
        self.invalidate_overlay_cache()
        self.load_current()

    def _handle_clear_labels_current(self) -> None:
        base = self._current_base()
        if not base:
            return
        if not self.session.dataset_path:
            return
        if self.label_service:
            self.label_service.clear_labels(base)
        self.invalidate_overlay_cache(base)
        self.load_current()
        self._safe_status_message("Labels cleared for current image.", 3000)

    def _mark_cache_dirty(self) -> None:

        self.session.mark_cache_dirty()

        # CRITICAL: Also mark the handler as dirty so it persists and notifies workspace
        if hasattr(self, 'handler') and self.handler:
            log_debug(f"Marking handler dirty for {self.handler.dataset_path.name}", "ImageViewer")
            self.handler.mark_dirty()

        # Show brief notification that changes will be saved
        self._safe_status_message("Workspace changes pending save...", 1000)

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
            self._dispatch_cache_flush(payload)

    def _dispatch_cache_flush(self, payload: CachePersistPayload) -> None:
        # Use coordinator for thread-safe handling of concurrent flush requests
        if not self._cache_flush_coordinator.request_flush(payload):
            # Flush is already running, payload queued as pending
            return
        runnable = CacheFlushRunnable(payload, self._cache_flush_notifier)
        self.thread_pool.start(runnable)

    def _handle_cache_flush_finished(self) -> None:
        # Centralized invalidation (workspace index + collection aggregation) after a cache write completes.
        coordinator = get_cache_coordinator()
        try:
            if self.workspace_dir:
                coordinator.set_workspace(Path(self.workspace_dir))
            if self.session.dataset_path:
                coordinator.mark_dataset_dirty(Path(self.session.dataset_path))
            coordinator.flush_workspace()
        except Exception:
            pass

        # Show notification that workspace was saved
        self._safe_status_message("Workspace saved", 2000)
        # Check if there's pending work and start it
        pending = self._cache_flush_coordinator.mark_completed()
        if pending is not None:
            self._dispatch_cache_flush(pending)

    def _wait_for_cache_flush(self) -> None:
        if self._cache_flush_coordinator.is_idle():
            return
        loop = QEventLoop(self)

        def _check_state() -> None:
            if self._cache_flush_coordinator.is_idle():
                loop.quit()

        self._cache_flush_notifier.finished.connect(_check_state)
        _check_state()
        if self._cache_flush_coordinator.has_pending_work():
            loop.exec()
        self._cache_flush_notifier.finished.disconnect(_check_state)

    def invalidate_overlay_cache(self, base: Optional[str] = None) -> None:
        self.overlay_workflow.invalidate(base)

    def _read_label_boxes(self, base: str, channel: str) -> List[Tuple[str, float, float, float, float, QColor, bool]]:
        """Get direct labels for a channel (not projected).

        Note: Returns 7-tuple: (display, x, y, w, h, color, is_auto).
        The OverlayOrchestrator adds the is_projected flag.
        """
        if not self.label_service:
            return []
        overlay_boxes = self.label_service.get_overlay_boxes(base, channel)
        # Convert (r,g,b) tuple to QColor, keep is_auto
        return [(display, x, y, w, h, QColor(r, g, b), is_auto)
                for display, x, y, w, h, (r, g, b), is_auto in overlay_boxes]

    def _label_signature(
        self,
        base: str,
        channel: str,
        boxes: List[Tuple[str, float, float, float, float, QColor, bool]],  # noqa: ARG002
    ) -> Optional[Tuple[Any, ...]]:
        if not self.label_service:
            return None
        return self.label_service.label_signature(base, channel)

    def _label_path(self, base: str, channel: str) -> Optional[Path]:
        if not self.session.dataset_path or not self.label_service:
            return None
        return self.label_service.label_file_path(base, channel)

    # ============================================================================
    # CACHE & STATE MANAGEMENT
    # ============================================================================

    def _reset_queue_progress_state(self) -> None:
        self.queue_manager.reset()
        self.signature_manager.reset_progress()
        self.quality_manager.reset()

    def _evict_pixmap_cache_entry(self, base: str) -> None:
        # Compatibility wrapper: evict pixmap cache for moved/deleted entries.
        if hasattr(self.session, "_evict_pixmap_cache_entry"):
            self.session._evict_pixmap_cache_entry(base)

    def _handle_progress_snapshot(self, snapshot) -> None:
        if self.progress_panel:
            self.progress_panel.set_snapshot(snapshot)
        # Also update outlier dialog if open
        if self._outlier_dialog and self._outlier_dialog.isVisible():
            self._outlier_dialog.set_progress(snapshot)

    def _handle_calibration_activity_changed(self, pending: int) -> None:
        self.queue_manager.update(
            pending=pending,
            label="Detecting chessboards",
            task_id=config.progress_task_detection,
            cancel_handler=self.calibration_controller.cancel_all,
        )
        self._update_cancel_button()

    def _start_refinement_progress(self, total: int) -> None:
        if total <= 0:
            return
        self._refine_total = total
        self._refine_progress = 0
        self.progress_tracker.start(
            config.progress_task_refinement,
            "Refining chessboard corners",
            total,
        )
        self.cancel_controller.register(
            config.progress_task_refinement,
            self.calibration_refiner.cancel,
        )
        self._update_cancel_button()
        # Create tqdm bar for terminal output
        try:
            self._refine_tqdm = tqdm(
                total=total,
                desc="Refining corners",
                unit="img",
                leave=False,
                ncols=100,
            )
        except ImportError:
            self._refine_tqdm = None

    def _advance_refinement_progress(self) -> None:
        if self._refine_total <= 0:
            return
        self._refine_progress = min(self._refine_total, self._refine_progress + 1)
        self.progress_tracker.update(
            config.progress_task_refinement,
            self._refine_progress,
            self._refine_total,
        )
        # Update tqdm bar
        if self._refine_tqdm:
            self._refine_tqdm.update(1)

    def _finish_refinement_progress(self) -> None:
        if self._refine_total <= 0:
            return
        self._refine_total = 0
        self._refine_progress = 0
        self.progress_tracker.finish(config.progress_task_refinement)
        self.cancel_controller.unregister(config.progress_task_refinement)
        self._update_cancel_button()
        # Close tqdm bar
        if self._refine_tqdm:
            self._refine_tqdm.close()
            self._refine_tqdm = None

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
        if not self.session.loader:  # Keep original - silent check in callback
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
        self._safe_status_message(f"Duplicate analysis failed for {base}: {message}", 4000)

    def _handle_signature_sweep_completed(self) -> None:
        """Mark duplicate sweep as completed and update workspace."""
        log_info("Duplicate sweep completed successfully", "VIEWER")
        self.session.state.cache_data["sweep_flags"]["duplicates"] = True
        self._mark_cache_dirty()

        # Force immediate save of session with sweep flag
        if self.session.cache_dirty:
            payload = self.session.snapshot_cache_payload()
            if payload:
                write_cache_payload(payload)
                log_info("Sweep flag saved to dataset cache", "VIEWER")

        # If this is a collection, propagate sweep flag to all children
        if self.session.loaded_kind == "collection":
            log_info("Propagating sweep completion to collection children", "VIEWER")

            # Save current dataset path before flushing children
            current_dataset = self.session.cache_service._cache.get("last_dataset")

            collection = self.session.collection
            if isinstance(collection, Collection):
                registry = get_handler_registry()
                for child_key, child_dir in collection._child_dirs.items():
                    try:
                        child_handler = registry.get_or_create(child_dir)
                        if child_handler:
                            # Load child session to mark sweep flag
                            child_session = child_handler.load_session()
                            if child_session:
                                child_session.state.cache_data["sweep_flags"]["duplicates"] = True
                                child_session.cache_dirty = True
                                # Save child session with sweep flag
                                payload = child_session.snapshot_cache_payload()
                                if payload:
                                    write_cache_payload(payload)
                                log_info(f"Marked sweep done for child: {child_key}", "VIEWER")

                                # CRITICAL: Update handler's session if it's already loaded
                                # Otherwise the handler's old session will overwrite our changes on next flush
                                if child_handler.session is not None and child_handler.session is not child_session:
                                    child_handler.session.state.cache_data["sweep_flags"]["duplicates"] = True
                                    child_handler.session.cache_dirty = False  # Already saved above

                            # Also update summary for workspace table
                            if child_handler.summary:
                                child_handler.summary.set_sweep_duplicates_done(True)
                                child_handler.mark_dirty()
                                child_handler.force_flush()
                    except Exception as e:
                        log_error(f"Failed to update child {child_key}: {e}", "VIEWER")

            # Restore last_dataset to the collection (not last child)
            if current_dataset:
                self.session.cache_service._cache["last_dataset"] = current_dataset
                self.session.cache_service.save()  # Persist last_dataset to disk
                log_info(f"Restored last_dataset to collection: {current_dataset}", "VIEWER")

        # For regular datasets, notify handler to save sweep state
        elif hasattr(self, 'handler') and self.handler:
            if self.handler.summary:
                self.handler.summary.sweep_duplicates_done = True
                log_info("Handler summary updated with sweep completion", "VIEWER")
            self.handler.mark_dirty()
            self.handler.force_flush()  # Force immediate save
            log_info("Handler notified of sweep completion", "VIEWER")

    def _handle_calibration_detection_completed(self, base: str, has_detection: bool) -> None:
        """Handle calibration detection completed signal for auto-search progress tracking."""
        # Check if we have an active auto-search in progress
        if not hasattr(self, "_calib_search_task_id") or not self._calib_search_task_id:
            return

        self._calib_search_completed += 1
        if has_detection:
            self._calib_search_found += 1

        # Update tqdm progress bar
        if hasattr(self, "_calib_search_tqdm") and self._calib_search_tqdm is not None:
            self._calib_search_tqdm.update(1)

        # Update GUI progress tracker (value only, label was set in start())
        self.progress_tracker.update(
            self._calib_search_task_id,
            self._calib_search_completed,
        )

        # Check if search is complete
        if self._calib_search_completed >= self._calib_search_total:
            # Close tqdm bar
            if hasattr(self, "_calib_search_tqdm") and self._calib_search_tqdm is not None:
                self._calib_search_tqdm.close()
                self._calib_search_tqdm = None

            # Finish GUI progress
            self.progress_tracker.finish(self._calib_search_task_id)
            self._calib_search_task_id = None

            # Show summary
            msg = f"Auto-search complete: found {self._calib_search_found} calibration candidates in {self._calib_search_total} images"
            self._safe_status_message(msg, 8000)
            log_info(msg, "AUTO_CALIB")

            # Update stats panel to reflect new calibration marks
            self._update_stats_panel()

            # Mark cache as dirty to save new auto-marked images
            self._mark_cache_dirty()

    def _handle_refinement_ready(
        self,
        base: str,
        refined: Dict[str, Optional[List[List[float]]]],
    ) -> None:
        self._advance_refinement_progress()
        if base not in self.state.calibration_marked:
            return
        # Load existing corners or create new bucket
        bucket = self.session.get_corners(base) or {}
        updated = False
        for channel, points in refined.items():
            if not points:
                continue
            # Store refined corners in *_subpixel keys (preserve originals)
            subpixel_key = f"{channel}_subpixel"
            bucket[subpixel_key] = points
            updated = True
        if not updated:
            return
        # Save corners immediately to disk (triggers flush_dirty_corners on next snapshot)
        self.session.set_corners(base, bucket)
        self.invalidate_overlay_cache(base)
        if self._current_base() == base:
            self.load_image_pair(base)
        self._mark_cache_dirty()

    def _handle_refinement_failed(self, base: str, message: str) -> None:
        self._advance_refinement_progress()
        self._safe_status_message(f"Subpixel refinement failed for {base}: {message}", 5000)

    def _handle_refinement_batch_finished(self, success: int, failed: int) -> None:
        self._finish_refinement_progress()
        if success == 0 and failed == 0:
            self._safe_status_message("Corner refinement cancelled.", 4000)
            return
        summary = f"Corner refinement finished ({success} updated"
        summary += f", {failed} skipped)" if failed else ")"
        self._safe_status_message(summary, 4000)

    def _handle_calibration_solved(self, payload: Dict[str, Any]) -> None:
        self.progress_tracker.finish(config.progress_task_solver)
        self.cancel_controller.unregister(config.progress_task_solver)
        self._update_cancel_button()
        channels = payload.get("channels", {}) if isinstance(payload, dict) else {}
        if not channels:
            self._safe_status_message("Calibration solver returned no data.", 4000)
            return
        for channel, data in channels.items():
            if not isinstance(data, dict):
                continue
            self.state.cache_data["_matrices"][channel] = data
            per_view = data.get("per_view_errors") if isinstance(data, dict) else None
            if isinstance(per_view, dict):
                self.state.cache_data["reproj_errors"][channel] = {
                    base: float(err)
                    for base, err in per_view.items()
                    if isinstance(base, str) and isinstance(err, (int, float))
                }
        self._mark_cache_dirty()
        if self.view_rectified and self.session.has_images():
            self.load_current()
        file_path = payload.get("file_path") if isinstance(payload, dict) else None
        if file_path:
            self._safe_status_message(f"Calibration saved to {Path(file_path).name}", 6000)
        else:
            self._safe_status_message("Calibration matrices updated.", 6000)
        self._update_stats_panel()
        self._refresh_outlier_dialog_rows()
        self._refresh_calibration_report_if_open()

    def _open_calibration_outlier_dialog(self) -> None:
        if not require_dataset(self, "Calibration outliers"):
            return
        bases = set(self.state.calibration_marked)
        bases.update(self.state.calibration_outliers_extrinsic)
        bases.update(self.state.calibration_outliers_intrinsic.get("lwir", set()))
        bases.update(self.state.calibration_outliers_intrinsic.get("visible", set()))
        if not bases:
            QMessageBox.information(self, "Calibration outliers", "No calibration images to review.")
            return
        rows = self.session.build_outlier_rows(sorted(bases))

        # Save outliers table to logs
        try:

            # Convert rows format to outliers_data format
            outliers_data = {"lwir": [], "visible": [], "stereo": []}
            for row in rows:
                base = row.get("base", "")
                for channel in ["lwir", "visible", "stereo"]:
                    error = row.get(channel)
                    if isinstance(error, (int, float)) and error > 0:
                        # For now, use simple heuristic for threshold (can be enhanced)
                        # We'll compute a median-based threshold from all errors
                        outliers_data[channel].append({
                            "base": base,
                            "error": float(error),
                            "threshold": 0.0,  # Will be computed globally
                        })

            # Compute thresholds per channel (median + 3.5*MAD)
            for channel, channel_outliers in outliers_data.items():
                if not channel_outliers:
                    continue
                errors = [o["error"] for o in channel_outliers]
                median_err = median(errors) if errors else 0.0
                mad = median([abs(e - median_err) for e in errors]) if errors else 0.0
                threshold = median_err + 3.5* 1.4826 * mad
                threshold = max(threshold, 0.5)  # Floor at 0.5px

                # Update threshold for all entries in this channel
                for outlier in channel_outliers:
                    outlier["threshold"] = threshold

            # Write table
            table_data, headers = format_outliers_table(outliers_data)
            if table_data:  # Only write if there's data
                write_table_to_log(table_data, headers, "calibration_outliers", overwrite=True)
        except Exception as e:
            log_warning(f"Failed to save outliers table: {e}", "VIEWER")

        if self._outlier_dialog and self._outlier_dialog.isVisible():
            self._outlier_dialog.update_rows(rows)
            self._outlier_dialog.raise_()
            self._outlier_dialog.activateWindow()
            return

        # Build calibration info for the dialog
        calibration_info = self._build_calibration_info()

        dialog = CalibrationOutliersDialog(
            rows,
            self,
            refresh_intrinsic_callback=self._refresh_outlier_selection_and_intrinsic,
            refresh_extrinsic_callback=self._refresh_outlier_selection_and_extrinsic,
            calibration_info=calibration_info,
        )
        dialog.setModal(False)
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        self._outlier_dialog = dialog
        dialog.finished.connect(self._handle_outlier_dialog_finished)
        dialog.show()

    def _build_calibration_info(self) -> Dict[str, Any]:
        """Build calibration info dict for the outliers dialog."""

        info: Dict[str, Any] = {}
        if not self.session.dataset_path:
            return info

        intrinsic_path = self.session.dataset_path / config.calibration_intrinsic_filename
        if intrinsic_path.exists():
            info["intrinsic_path"] = intrinsic_path.name
            try:
                mtime = os.path.getmtime(intrinsic_path)
                info["intrinsic_date"] = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            except OSError:
                pass

        extrinsic_path = self.session.dataset_path / config.calibration_extrinsic_filename
        if extrinsic_path.exists():
            info["extrinsic_path"] = extrinsic_path.name
            try:
                mtime = os.path.getmtime(extrinsic_path)
                info["extrinsic_date"] = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            except OSError:
                pass

        return info

    def _apply_outlier_selection(
        self,
        include: Dict[str, Set[str]],
        exclude: Dict[str, Set[str]],
        *,
        show_status: bool,
    ) -> None:
        # Support both datasets (loader) and collections
        if self.session.loader:
            valid = set(self.session.loader.image_bases)
        elif self.session.collection:
            valid = set(self.session.collection.image_bases)
        else:
            log_debug("No loader or collection available for outlier selection", "OUTLIERS")
            return

        include = {k: {b for b in include.get(k, set()) if b in valid} for k in ("lwir", "visible", "stereo")}
        exclude = {k: {b for b in exclude.get(k, set()) if b in valid} for k in ("lwir", "visible", "stereo")}

        # Log outlier changes for debugging
        total_include = sum(len(v) for v in include.values())
        total_exclude = sum(len(v) for v in exclude.values())
        log_debug(f"Applying outlier selection: include={total_include} exclude={total_exclude}", "OUTLIERS")

        # Directly modify cache_data to persist outlier flags (new nested format)
        calib = self.state.cache_data.setdefault("calibration", {})

        for channel in ("lwir", "visible", "stereo"):
            # Include = remove outlier flag (include in calibration)
            for base in include.get(channel, set()):
                if base in calib and isinstance(calib[base], dict):
                    if "outlier" not in calib[base]:
                        calib[base]["outlier"] = {"lwir": False, "visible": False, "stereo": False}
                    calib[base]["outlier"][channel] = False
            # Exclude = set outlier flag (exclude from calibration)
            for base in exclude.get(channel, set()):
                if base not in calib:
                    calib[base] = {"auto": False, "outlier": {"lwir": False, "visible": False, "stereo": False}, "results": {}}
                if isinstance(calib[base], dict):
                    if "outlier" not in calib[base]:
                        calib[base]["outlier"] = {"lwir": False, "visible": False, "stereo": False}
                    calib[base]["outlier"][channel] = True

        self.state.rebuild_calibration_summary()
        self._mark_cache_dirty()
        self._update_stats_panel()
        changed = (
            include["lwir"]
            | include["visible"]
            | include["stereo"]
            | exclude["lwir"]
            | exclude["visible"]
            | exclude["stereo"]
        )
        current = self._current_base()
        if current in changed:
            self.invalidate_overlay_cache(current)
            self.load_current()
        if show_status:
            summary = (
                f"LWIR -{len(include['lwir'])}/+{len(exclude['lwir'])}; "
                f"Visible -{len(include['visible'])}/+{len(exclude['visible'])}; "
                f"Stereo -{len(include['stereo'])}/+{len(exclude['stereo'])}"
            )
            self._safe_status_message(f"Calibration outliers updated ({summary})", 6000)

    def _refresh_outlier_selection_and_intrinsic(self, include: Dict[str, Set[str]], exclude: Dict[str, Set[str]]) -> None:
        self._apply_outlier_selection(include, exclude, show_status=False)
        self._handle_compute_calibration_action()

    def _refresh_outlier_selection_and_extrinsic(self, include: Dict[str, Set[str]], exclude: Dict[str, Set[str]]) -> None:
        self._apply_outlier_selection(include, exclude, show_status=False)
        self._handle_compute_extrinsic_action()

    def _refresh_outlier_dialog_rows(self) -> None:
        if not self._outlier_dialog or not self._outlier_dialog.isVisible():
            return
        bases = set(self.state.calibration_marked)
        bases.update(self.state.calibration_outliers_extrinsic)
        bases.update(self.state.calibration_outliers_intrinsic.get("lwir", set()))
        bases.update(self.state.calibration_outliers_intrinsic.get("visible", set()))
        if not bases:
            return
        rows = self.session.build_outlier_rows(sorted(bases))
        self._outlier_dialog.update_rows(rows)

    def _handle_outlier_dialog_finished(self, _code: int) -> None:
        dialog = self._outlier_dialog
        self._outlier_dialog = None
        if not dialog:
            return
        include, exclude = dialog.selected_channel_sets()
        self._apply_outlier_selection(include, exclude, show_status=True)

    def _handle_calibration_solver_failed(self, message: str) -> None:
        self.progress_tracker.finish(config.progress_task_solver)
        self.cancel_controller.unregister(config.progress_task_solver)
        self._update_cancel_button()
        self._safe_status_message(f"Calibration solve failed: {message}", 6000)

    def _handle_extrinsic_solved(self, payload: Dict[str, Any]) -> None:
        self.progress_tracker.finish(config.progress_task_extrinsic)
        self.cancel_controller.unregister(config.progress_task_extrinsic)
        self._update_cancel_button()
        snapshot = dict(payload)
        file_path = snapshot.pop("file_path", None)
        per_pair = snapshot.pop("per_pair_errors", None)
        if isinstance(per_pair, list):
            self.state.cache_data["extrinsic_errors"] = {
                entry.get("base"): float(entry.get("translation_error", 0.0))
                for entry in per_pair
                if isinstance(entry, dict) and isinstance(entry.get("base"), str)
            }
        self.state.cache_data["_extrinsic"] = snapshot
        self._mark_cache_dirty()
        message = "Stereo calibration updated"
        if file_path:
            message = f"Stereo calibration saved to {Path(file_path).name}"
        self._safe_status_message(message, 6000)
        self._update_stats_panel()
        self._refresh_outlier_dialog_rows()
        self._refresh_calibration_report_if_open()

    def _handle_extrinsic_failed(self, message: str) -> None:
        self.progress_tracker.finish(config.progress_task_extrinsic)
        self.cancel_controller.unregister(config.progress_task_extrinsic)
        self._update_cancel_button()
        self._safe_status_message(f"Stereo calibration failed: {message}", 6000)

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
        self.filter_controller.reconcile_filter_state(show_warning=True)

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
        corners = self.session.get_corners(base)  # Lazy load from disk
        if not results:
            QMessageBox.information(
                self,
                "Calibration debug",
                "No calibration detection data available yet. Run calibration detection first.",
            )
            return
        self.calibration_workflow.emit_calibration_debug(base, results, corners)

    def closeEvent(self, event):  # type: ignore[override]
        log_info("closeEvent triggered - beginning shutdown sequence", "VIEWER")

        self._is_closing = True

        log_info("Flushing pending handler changes...", "VIEWER")
        # Flush any pending handler changes before shutdown
        get_handler_registry().flush_all()

        log_info("Saving label service data...", "VIEWER")
        # Save any pending label changes
        if self.label_service:
            self.label_service.shutdown()

        log_info("Shutting down background tasks...", "VIEWER")
        self._shutdown_background_tasks()

        log_info("Flushing cache...", "VIEWER")
        self._flush_cache(wait=True)

        # Save recent datasets cache
        log_info("Saving recent datasets cache...", "VIEWER")
        self.session.cache_service.save()

        log_info("Calling parent closeEvent...", "VIEWER")
        super().closeEvent(event)
        log_info("closeEvent complete", "VIEWER")

    def _shutdown_background_tasks(self) -> None:
        # Stop timers and queues
        if self.cache_timer.isActive():
            self.cache_timer.stop()
        self.overlay_prefetcher.clear()
        self.calibration_queue.clear()
        panel = getattr(self, "workspace_panel", None)
        if panel is not None:
            try:
                panel.cancel_all()
            except Exception:
                pass
        # Cancel managers and controllers
        self.signature_manager.cancel_all()
        self.quality_manager.cancel_all()
        self.calibration_controller.cancel_all()
        self.calibration_refiner.cancel()
        self.calibration_solver.cancel()
        self.calibration_extrinsic_solver.cancel()
        self.dataset_actions.cancel_background_jobs()
        # Shut down pending progress/cancel state
        self.cancel_controller.clear()
        self.progress_tracker.clear()
        # Wait briefly for pools to finish work without blocking shutdown too long
        try:
            self.calibration_thread_pool.waitForDone(500)
            self.thread_pool.waitForDone(500)
        except Exception:
            pass

    def _collect_refinement_candidates(self) -> List[str]:
        """Collect images with corners that can be refined."""
        # Support both datasets and collections
        image_bases = self.session.get_all_bases()
        if not image_bases:
            return []
        candidates: List[str] = []
        for base in image_bases:
            if base not in self.state.calibration_marked:
                continue
            # Check results instead of loading corners
            results = self.state.calibration_results.get(base, {})
            if results.get("lwir") is True or results.get("visible") is True:
                candidates.append(base)
        return candidates

    def _handle_refine_calibration_action(self) -> None:
        if not require_dataset(self, "Calibration"):
            return
        targets = self._collect_refinement_candidates()
        log_debug(f"[ImageViewer] Refine candidates: {len(targets)} targets")
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
        log_debug(f"[ImageViewer] Calling refiner.refine with {len(targets)} targets")
        queued = self.calibration_refiner.refine(targets)
        log_debug(f"[ImageViewer] Refiner returned queued={queued}")
        if queued:
            self._start_refinement_progress(queued)
            self._safe_status_message(
                f"Refining chessboard corners for {queued} image(s)…",
                4000,
            )
        else:
            log_warning("[ImageViewer] Refiner returned 0 queued tasks")

    def _handle_auto_calibration_search(self) -> None:
        """Auto-search for calibration candidates by detecting chessboards in unmarked images.

        Searches all images that are NOT marked for delete AND NOT marked for calibration.
        If a chessboard pattern is detected, the image is auto-marked as calibration candidate.
        """
        if not require_dataset(self, "Auto calibration search"):
            return

        # Get all image bases
        if self.session.collection:
            all_bases = self.session.collection.image_bases
        elif self.session.loader:
            all_bases = self.session.loader.image_bases
        else:
            return

        # Get current marks and calibration data
        marks = self.state.cache_data.get("marks", {})
        calibration = self.state.cache_data.get("calibration", {})

        # Filter: exclude images already tagged for delete or calibration
        candidates = []
        for base in all_bases:
            # Skip if marked for delete
            if base in marks:
                continue
            # Skip if already marked as calibration (user or auto)
            # Presence in dict = marked
            calib_entry = calibration.get(base)
            if isinstance(calib_entry, dict):
                continue
            candidates.append(base)

        if not candidates:
            QMessageBox.information(
                self,
                "Auto calibration search",
                "No candidate images found.\n\n"
                "All images are either tagged for delete or already marked as calibration.",
            )
            return

        reply = QMessageBox.question(
            self,
            "Auto calibration search",
            f"Search for chessboard patterns in {len(candidates)} images?\n\n"
            "Images with detected patterns will be auto-tagged as calibration candidates.\n"
            "This may take a while for large datasets.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Start progress tracking
        total = len(candidates)
        task_id = "calibration_search"
        self.progress_tracker.start(task_id, f"Searching calibration patterns (0/{total})", total)

        # Track progress via calibration controller signals
        self._calib_search_total = total
        self._calib_search_completed = 0
        self._calib_search_found = 0
        self._calib_search_task_id = task_id

        # Create tqdm bar for terminal output
        try:
            self._calib_search_tqdm = tqdm(
                total=total,
                desc="Auto-search calibration",
                unit="img",
                leave=False,
                ncols=100,
            )
        except ImportError:
            self._calib_search_tqdm = None

        # Queue detection - the calibration controller will emit signals
        queued = self.calibration_controller.prefetch(candidates, force=False)
        if queued > 0:
            self._safe_status_message(f"Searching calibration patterns in {queued} images…", 4000)
        else:
            self.progress_tracker.finish(task_id)
            self._safe_status_message("No images queued for pattern detection", 3000)

    def _handle_compute_calibration_action(self) -> None:
        if not require_dataset(self, "Calibration"):
            return

        # Quick count WITHOUT loading corners (uses cached results)
        channel_counts = self.calibration_workflow.count_calibration_samples()
        total_samples = channel_counts.get("lwir", 0) + channel_counts.get("visible", 0)

        if total_samples == 0:
            QMessageBox.information(
                self,
                "Calibration",
                "No calibration samples available.\n\n"
                "Tag images for calibration and run detection first.",
            )
            return
        if all(count < 3 for count in channel_counts.values()):
            QMessageBox.information(
                self,
                "Calibration",
                "Need at least 3 valid samples per channel to calibrate.\n\n"
                f"Current: LWIR={channel_counts.get('lwir', 0)}, Visible={channel_counts.get('visible', 0)}",
            )
            return

        # Show confirmation BEFORE loading corners
        reply = QMessageBox.question(
            self,
            "Compute calibration",
            (
                f"Compute camera matrices using approximately "
                f"{channel_counts.get('lwir', 0)} LWIR and {channel_counts.get('visible', 0)} visible sample(s)?\n\n"
                "This will load corner data from disk."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # NOW load corners and collect samples
        samples = self.calibration_workflow.collect_calibration_samples()
        if not samples:
            QMessageBox.information(
                self,
                "Calibration",
                "No valid samples found after loading corners.",
            )
            return

        if self.calibration_solver.solve(samples):
            self.progress_tracker.set_busy(
                config.progress_task_solver,
                "Computing calibration matrices…",
            )
            self.cancel_controller.register(config.progress_task_solver, self.calibration_solver.cancel)
            self._update_cancel_button()
            self._safe_status_message("Computing calibration matrices…", 4000)

    def _handle_compute_extrinsic_action(self) -> None:
        if not require_dataset(self, "Calibration"):
            return

        # Check if intrinsic calibration is in progress
        if self.progress_tracker.is_busy(config.progress_task_solver):
            QMessageBox.warning(
                self,
                "Calibration in Progress",
                "Intrinsic calibration is currently running.\n\n"
                "Please wait for it to complete before computing extrinsic calibration, "
                "otherwise the extrinsic solver might use outdated intrinsic data.",
            )
            return

        # Quick count WITHOUT loading corners
        sample_count = self.calibration_workflow.count_extrinsic_samples()

        if sample_count < 3:
            QMessageBox.information(
                self,
                "Calibration",
                f"Need at least 3 calibration images with detections on both cameras.\n\n"
                f"Current: {sample_count} paired samples.",
            )
            return

        # Show confirmation BEFORE loading corners
        reply = QMessageBox.question(
            self,
            "Compute extrinsic transform",
            f"Compute LWIR ↔ Visible transform using approximately {sample_count} paired sample(s)?\n\n"
            "This will load corner data from disk.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # NOW load corners
        samples = self.calibration_workflow.collect_extrinsic_samples()
        if len(samples) < 3:
            QMessageBox.information(
                self,
                "Calibration",
                f"Only {len(samples)} valid paired samples found after loading corners.",
            )
            return

        if self.calibration_extrinsic_solver.solve(samples):
            self.progress_tracker.set_busy(
                config.progress_task_extrinsic,
                "Computing stereo calibration…",
            )
            self.cancel_controller.register(config.progress_task_extrinsic, self.calibration_extrinsic_solver.cancel)
            self._update_cancel_button()
            self._safe_status_message("Computing stereo calibration…", 4000)

    def _calibration_intrinsic_path(self) -> Optional[Path]:
        if not self.session.dataset_path:
            return None
        config = get_config()
        return self.session.dataset_path / config.calibration_intrinsic_filename

    def _calibration_extrinsic_path(self) -> Optional[Path]:
        if not self.session.dataset_path:
            return None
        config = get_config()
        return self.session.dataset_path / config.calibration_extrinsic_filename

    def _get_dataset_paths_for_calibration_dialog(self) -> Optional[List[str]]:
        """Get dataset paths for calibration dialog, showing children for collections."""
        if not self.session.dataset_path:
            return None
        # For collections, show collection name with child datasets
        if self.session.collection:
            child_names = sorted(self.session.collection._child_dirs.keys())
            if child_names:
                collection_name = self.session.dataset_path.name
                return [f"{collection_name} ({', '.join(child_names)})"]
        # For standalone dataset, just show path
        return [str(self.session.dataset_path)]

    def _handle_show_calibration_dialog(self) -> None:
        if not require_dataset(self, "Calibration"):
            return
        dialog = CalibrationCheckDialog(
            self,
            self.state.cache_data["_matrices"],
            self.state.cache_data["_extrinsic"],
            intrinsic_path=self._calibration_intrinsic_path(),
            extrinsic_path=self._calibration_extrinsic_path(),
            dataset_paths=self._get_dataset_paths_for_calibration_dialog(),
            dataset_path=self.session.dataset_path,
        )
        dialog.setModal(False)
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.show()
        self._calibration_report_dialog = dialog
        dialog.destroyed.connect(lambda _obj=None: setattr(self, "_calibration_report_dialog", None))

    def _refresh_calibration_report_if_open(self) -> None:
        if not getattr(self, "_calibration_report_dialog", None):
            return
        dialog = self._calibration_report_dialog
        if not dialog or not dialog.isVisible():
            return
        dialog.refresh_data(
            self.state.cache_data["_matrices"],
            self.state.cache_data["_extrinsic"],
            intrinsic_path=self._calibration_intrinsic_path(),
            extrinsic_path=self._calibration_extrinsic_path(),
            dataset_paths=self._get_dataset_paths_for_calibration_dialog(),
            dataset_path=self.session.dataset_path,
        )

    # ------------------------------------------------------------------
    # Label report
    # ------------------------------------------------------------------

    def _handle_label_report(self) -> None:
        """Open a label report dialog for the current dataset/collection."""
        if not require_dataset(self, "Label report"):
            return
        summary = self._build_label_summary(force=False)
        title = f"Label report — {self.session.dataset_path.name}" if self.session.dataset_path else "Label report"
        dialog = LabelReportDialog(self, summary=summary, title=title)
        dialog._refresh_callback = lambda: self._refresh_label_report(dialog)
        dialog.setModal(False)
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.show()
        self._label_report_dialog = dialog
        dialog.destroyed.connect(lambda _obj=None: setattr(self, "_label_report_dialog", None))

    def _handle_workspace_label_report(self) -> None:
        """Open a label report aggregated across all workspace datasets."""
        if not self.workspace_dir:
            QMessageBox.information(self, "Label report", "Open a workspace first.")
            return
        from backend.services.labels.label_summary_derivation import (
            derive_labels_summary_from_disk,
            load_labels_summary_cache,
            merge_labels_summaries,
            save_labels_summary_cache,
        )
        from backend.services.workspace_inspector import _is_dataset_dir
        workspace_path = Path(self.workspace_dir)
        summaries = []
        for entry in sorted(workspace_path.iterdir()):
            if not entry.is_dir():
                continue
            # Check child datasets (collections)
            children = [p for p in entry.iterdir() if p.is_dir() and _is_dataset_dir(p)]
            if children:
                for child in children:
                    summaries.append(self._get_or_derive_label_summary(child))
            elif _is_dataset_dir(entry):
                summaries.append(self._get_or_derive_label_summary(entry))

        merged = merge_labels_summaries(summaries) if summaries else {}
        title = f"Label report — Workspace ({workspace_path.name})"
        dialog = LabelReportDialog(self, summary=merged, title=title)
        dialog._refresh_callback = lambda: self._refresh_workspace_label_report(dialog, workspace_path)
        dialog.setModal(False)
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.show()

    def _get_or_derive_label_summary(self, dataset_path: Path) -> Dict[str, Any]:
        """Load cached or derive label summary for a dataset path."""
        from backend.services.labels.label_summary_derivation import (
            derive_labels_summary_from_disk,
            load_labels_summary_cache,
            save_labels_summary_cache,
        )
        cached = load_labels_summary_cache(dataset_path)
        if cached is not None:
            return cached
        ws_cfg = self._workspace_label_config_path()
        summary = derive_labels_summary_from_disk(dataset_path, config_path=ws_cfg)
        if summary.get("total_annotations", 0) > 0:
            save_labels_summary_cache(dataset_path, summary)
        return summary

    def _build_label_summary(self, force: bool = False) -> Dict[str, Any]:
        """Build label summary for the active dataset/collection."""
        if self.label_service:
            return self.label_service.get_labels_summary(force_rebuild=force)
        # Fallback: derive directly from disk
        if self.session.dataset_path:
            from backend.services.labels.label_summary_derivation import (
                derive_labels_summary_from_disk,
            )
            # For collections, merge child summaries
            ws_cfg = self._workspace_label_config_path()
            if self.session.collection:
                from backend.services.labels.label_summary_derivation import merge_labels_summaries
                child_summaries = []
                for child_path in self.session.collection._child_dirs.values():
                    child_summaries.append(derive_labels_summary_from_disk(child_path, config_path=ws_cfg))
                return merge_labels_summaries(child_summaries)
            return derive_labels_summary_from_disk(self.session.dataset_path, config_path=ws_cfg)
        return {}

    def _refresh_label_report(self, dialog: LabelReportDialog) -> None:
        """Refresh an existing dataset label report dialog."""
        summary = self._build_label_summary(force=True)
        dialog.refresh(summary)

    def _refresh_workspace_label_report(self, dialog: LabelReportDialog, workspace_path: Path) -> None:
        """Refresh an existing workspace label report dialog."""
        from backend.services.labels.label_summary_derivation import (
            derive_labels_summary_from_disk,
            merge_labels_summaries,
        )
        from backend.services.workspace_inspector import _is_dataset_dir
        ws_cfg = self._workspace_label_config_path()
        summaries = []
        for entry in sorted(workspace_path.iterdir()):
            if not entry.is_dir():
                continue
            children = [p for p in entry.iterdir() if p.is_dir() and _is_dataset_dir(p)]
            if children:
                for child in children:
                    summaries.append(derive_labels_summary_from_disk(child, config_path=ws_cfg))
            elif _is_dataset_dir(entry):
                summaries.append(derive_labels_summary_from_disk(entry, config_path=ws_cfg))
        merged = merge_labels_summaries(summaries) if summaries else {}
        title = f"Label report — Workspace ({workspace_path.name})"
        dialog.refresh(merged, title=title)

    def _handle_rectified_toggle(self, enabled: bool) -> None:
        success = self.view_state.toggle_rectified(enabled)
        if not success and hasattr(self.ui, "action_toggle_rectified"):
            # Revert checkbox if validation failed
            self.ui.action_toggle_rectified.blockSignals(True)
            self.ui.action_toggle_rectified.setChecked(False)
            self.ui.action_toggle_rectified.blockSignals(False)
        else:
            self.view_rectified = enabled
            self.overlay_orchestrator.set_view_rectified(enabled)
            self.session.cache_service.set_preference("view_rectified", enabled)
            self.invalidate_overlay_cache()
            base = self._current_base()
            if base:
                self.load_image_pair(base)
            status = "enabled" if enabled else "disabled"
            self._safe_status_message(f"Rectified view {status}", 2000)

    def _set_grid_mode(self, mode: str) -> None:
        """Set grid display mode.

        Args:
            mode: Grid mode - "off", "thirds", or "detailed"
        """
        # Skip if we're syncing action states (avoid recursion)
        if getattr(self, "_syncing_actions", False):
            return
        self.view_state.grid_mode = mode
        self.overlay_orchestrator.set_grid_mode(mode)
        self._safe_status_message(f"Grid: {mode}", 2000)

    def _set_align_mode(self, mode: str) -> None:
        """Set stereo alignment mode.

        Args:
            mode: Alignment mode - "disabled", "full", "fov_focus", or "max_overlap"
        """
        # Skip if we're syncing action states (avoid recursion)
        if getattr(self, "_syncing_actions", False):
            return
        # Check if extrinsic calibration is available for non-disabled modes
        if mode != "disabled":
            extrinsic = self.state.cache_data.get("_extrinsic")
            if not extrinsic:
                self._safe_status_message("No extrinsic calibration available. Run stereo calibration first.", 5000)
                # Force revert to disabled using the action group
                self._force_align_action("disabled")
                return

        self._align_mode = mode
        self.overlay_orchestrator.set_align_mode(mode)
        self.session.cache_service.set_preference("align_mode", mode)
        self.invalidate_overlay_cache()

        base = self._current_base()
        if base:
            self.load_image_pair(base)

        mode_labels = {
            "disabled": "Disabled",
            "full": "Full View",
            "fov_focus": "FOV Focus",
            "max_overlap": "Max Overlap",
        }
        self._safe_status_message(f"Stereo alignment: {mode_labels.get(mode, mode)}", 2000)

    def _force_align_action(self, mode: str) -> None:
        """Force a specific alignment action to be checked, unchecking all others."""
        align_actions = {
            "disabled": getattr(self.ui, "action_align_disabled", None),
            "full": getattr(self.ui, "action_align_full", None),
            "fov_focus": getattr(self.ui, "action_align_fov_focus", None),
            "max_overlap": getattr(self.ui, "action_align_max_overlap", None),
        }
        target = align_actions.get(mode)
        if target:
            # Use the action group to handle exclusivity properly
            # First disconnect to avoid recursion, then trigger, then reconnect
            for action in align_actions.values():
                if action:
                    action.blockSignals(True)
            target.setChecked(True)
            for action in align_actions.values():
                if action:
                    action.blockSignals(False)

    def _set_corner_display_mode(self, mode: str) -> None:
        """Set corner display mode.

        Args:
            mode: One of:
                - 'original': Show only original detected corners (blue circles)
                - 'subpixel': Show subpixel-refined corners (cyan crosses)
                - 'both': Show both for comparison (debug mode)
        """
        # Skip if we're syncing action states (avoid recursion)
        if getattr(self, "_syncing_actions", False):
            return

        # Check if subpixel corners exist when switching to subpixel/both mode
        if mode in ("subpixel", "both"):
            has_any_subpixel = self._check_has_subpixel_corners()
            if not has_any_subpixel:
                QMessageBox.information(
                    self,
                    "No Subpixel Corners",
                    "No subpixel-refined corners found.\n\n"
                    "Run 'Calibration → Refine chessboard corners' first to generate "
                    "subpixel-accurate corner positions.\n\n"
                    "Falling back to original corners.",
                )
                # Revert to original mode
                mode = "original"
                self._force_corner_action("original")

        self._corner_display_mode = mode
        self.overlay_orchestrator.set_corner_display_mode(mode)
        self.session.cache_service.set_preference("corner_display_mode", mode)
        self.invalidate_overlay_cache()

        base = self._current_base()
        if base:
            self.load_image_pair(base)

        mode_labels = {
            "original": "Original Only",
            "subpixel": "Subpixel Only",
            "both": "Both (Debug)",
        }
        self._safe_status_message(f"Corner display: {mode_labels.get(mode, mode)}", 2000)

    def _check_has_subpixel_corners(self) -> bool:
        """Check if any calibration image has subpixel-refined corners."""
        for base in self.state.calibration_marked:
            corners = self.session.get_corners(base)
            if corners:
                if corners.get("lwir_subpixel") or corners.get("visible_subpixel"):
                    return True
        return False

    def _force_corner_action(self, mode: str) -> None:
        """Force a specific corner display action to be checked."""
        corner_actions = {
            "original": getattr(self.ui, "action_corners_original", None),
            "subpixel": getattr(self.ui, "action_corners_subpixel", None),
            "both": getattr(self.ui, "action_corners_both", None),
        }
        target = corner_actions.get(mode)
        if target:
            for action in corner_actions.values():
                if action:
                    action.blockSignals(True)
            target.setChecked(True)
            for action in corner_actions.values():
                if action:
                    action.blockSignals(False)

    def _handle_show_labels_toggle(self, enabled: bool) -> None:
        self.view_state.toggle_labels(enabled)

    def _handle_show_overlays_toggle(self, enabled: bool) -> None:
        """Toggle visibility of image info overlays (status text, calibration markers, etc.)."""
        self.view_state.toggle_overlays(enabled)
        self._safe_status_message(f"Image info overlay: {'visible' if enabled else 'hidden'}", 2000)

    def _handle_use_subpixel_toggle(self, enabled: bool) -> None:
        """Toggle whether calibration uses subpixel-refined corners."""
        # Check if subpixel corners exist when enabling
        if enabled and not self._check_has_subpixel_corners():
            QMessageBox.information(
                self,
                "No Subpixel Corners",
                "No subpixel-refined corners found.\n\n"
                "Run 'Calibration → Refine chessboard corners' first to generate "
                "subpixel-accurate corner positions.",
            )
            # Revert toggle
            action = getattr(self.ui, "action_use_subpixel_corners", None)
            if action:
                action.blockSignals(True)
                action.setChecked(False)
                action.blockSignals(False)
            return

        self.session.cache_service.set_preference("use_subpixel_corners", enabled)
        status = "enabled" if enabled else "disabled"
        self._safe_status_message(f"Use subpixel corners for calibration: {status}", 3000)

    def clear_empty_datasets(self) -> None:
        self.dataset_actions.clear_empty_datasets()

    def delete_marked_images(self):
        debug = os.environ.get("DEBUG", "").lower() in {"1", "true", "on"}
        if debug:
            log_debug(f"delete_marked_images: {len(self.state.cache_data['marks'])} images marked", "VIEWER")
        self.dataset_actions.delete_marked_images()
        self.filter_controller.reconcile_filter_state(show_warning=True)
        if debug:
            log_debug("delete_marked_images completed", "VIEWER")

    def restore_images(self):
        debug = os.environ.get("DEBUG", "").lower() in {"1", "true", "on"}
        if debug:
            log_debug(f"restore_images: {self.session.count_trash_pairs()} images in trash", "VIEWER")
        self.dataset_actions.restore_images()
        self.filter_controller.reconcile_filter_state(show_warning=True)
        if debug:
            log_debug("restore_images completed", "VIEWER")

    def _untag_by_reason(self, reason: Optional[str], reason_label: str) -> None:
        """Untag (remove marks for) images with specific reason in current dataset."""
        if not self.session.has_images():
            self._safe_status_message("No dataset loaded", 3000)
            return

        marks = self.state.cache_data.get("marks", {})
        if reason is None:
            # Untag all - only this clears everything
            count = len(marks)
            if count == 0:
                self._safe_status_message("No images tagged for deletion", 3000)
                return
            self.state.cache_data["marks"] = {}
        else:
            # Untag specific reason - only affects that specific reason (new unified format)
            to_untag = [
                base for base, entry in marks.items()
                if isinstance(entry, dict) and entry.get("reason") == reason
            ]
            if not to_untag:
                self._safe_status_message(f"No images tagged as {reason_label}", 3000)
                return
            count = len(to_untag)
            for base in to_untag:
                marks.pop(base, None)

        self.state.rebuild_reason_counts()
        self.session.mark_cache_dirty()
        self._update_stats_panel()
        self._safe_status_message(f"Untagged {count} {reason_label} images", 3000)

    def _restore_by_reason(self, reason: str, reason_label: str) -> None:
        """Restore trashed images with specific reason in current dataset.

        Note: Currently restores ALL trashed images since selective restore
        by reason requires additional tracking not yet implemented.
        """
        if not self.session.has_images():
            self._safe_status_message("No dataset loaded", 3000)
            return

        # For now, restore all and inform user
        # TODO: Implement selective restore by reason when archived entries track reason
        trash_count = self.session.count_trash_pairs()
        if trash_count == 0:
            self._safe_status_message("No trashed images to restore", 3000)
            return

        reply = QMessageBox.question(
            self,
            f"Restore {reason_label}",
            f"Selective restore by reason is not yet fully implemented.\n\n"
            f"Would you like to restore ALL {trash_count} trashed images instead?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        restored = self.session.restore_from_trash()
        self.filter_controller.reconcile_filter_state(show_warning=True)
        self._safe_status_message(f"Restored {restored} images", 3000)

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
        self.ui.btn_prev_fast.setEnabled(False)
        self.ui.btn_next_fast.setEnabled(False)
        self.ui.btn_goto.setEnabled(False)
        self.setWindowTitle("Image Viewer")
        self.state.signatures = {}
        self.state.calibration_marked.clear()
        self.state.calibration_results.clear()
        # Clear all corners from cache_data["calibration"]
        for base_data in self.state.cache_data.get("calibration", {}).values():
            if isinstance(base_data, dict):
                base_data.pop("corners", None)
        self.signature_manager.cancel_all()
        self._reset_calibration_jobs()
        self._clear_pending_calibration_marks()
        self.overlay_prefetcher.clear()
        self.invalidate_overlay_cache()
        self.progress_tracker.clear()
        self._reset_queue_progress_state()
        if self.label_service:
            self.label_service.clear_cache()
        self._manual_label_mode = False
        self._update_labeling_views()
        self._sync_action_states()
        self._update_delete_button()
        self._update_restore_menu()
        self._update_stats_panel()
        self.progress_tracker.clear()

    def _on_class_map_updated(self) -> None:
        if self.label_service:
            self.label_service.clear_cache()
        self.invalidate_overlay_cache()
        if self.session.has_images():
            self.load_current()

    def _update_delete_button(self):
        self.ui_helper.update_delete_button()

    def _update_restore_menu(self):
        self.ui_helper.update_restore_menu()

    def _update_stats_panel(self) -> None:
        self.ui_helper.update_stats_panel()
