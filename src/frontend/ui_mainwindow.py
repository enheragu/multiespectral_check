"""Qt Designer-generated layout for the main viewer window.

Defines widgets, menus, and placeholders the application wires into runtime controllers.
"""

from PyQt6 import QtCore, QtGui, QtWidgets
from frontend.widgets import style


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.main_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.main_layout.setContentsMargins(6, 6, 6, 6)
        self.main_layout.setSpacing(6)

        self.tab_widget = QtWidgets.QTabWidget(self.centralwidget)
        self.tab_widget.setObjectName("tab_widget")
        self.main_layout.addWidget(self.tab_widget)

        # Workspace tab (embedded workspace view)
        self.tab_workspace = QtWidgets.QWidget()
        self.tab_workspace.setObjectName("tab_workspace")
        self.workspace_layout = QtWidgets.QVBoxLayout(self.tab_workspace)
        self.workspace_layout.setContentsMargins(6, 6, 6, 6)
        self.workspace_layout.setSpacing(6)
        self.workspace_panel_container = QtWidgets.QWidget(self.tab_workspace)
        self.workspace_panel_container.setObjectName("workspace_panel_container")
        self.workspace_panel_layout = QtWidgets.QVBoxLayout(self.workspace_panel_container)
        self.workspace_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.workspace_panel_layout.setSpacing(0)
        self.workspace_layout.addWidget(self.workspace_panel_container)
        self.tab_widget.addTab(self.tab_workspace, "Workspace view")

        # Dataset/collection viewer tab
        self.tab_dataset = QtWidgets.QWidget()
        self.tab_dataset.setObjectName("tab_dataset")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.tab_dataset)
        self.verticalLayout.setContentsMargins(8, 8, 8, 8)
        self.verticalLayout.setSpacing(8)

        # Dataset/collection header row (title + controls)
        header_row = QtWidgets.QHBoxLayout()
        header_row.setSpacing(8)
        self.label_dataset_path = QtWidgets.QLabel(self.tab_dataset)
        self.label_dataset_path.setObjectName("label_dataset_path")
        font = self.label_dataset_path.font()
        font.setPointSize(10)
        font.setBold(True)
        self.label_dataset_path.setFont(font)
        self.label_dataset_path.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            | QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.label_dataset_path.setText("No dataset loaded")
        header_row.addWidget(self.label_dataset_path)
        header_row.addStretch(1)

        self.btn_prev = QtWidgets.QPushButton(self.tab_dataset)
        self.btn_prev.setText("◀  Previous")
        self.btn_prev.setEnabled(False)
        self.btn_prev.setObjectName("btn_prev")
        header_row.addWidget(self.btn_prev)

        self.btn_next = QtWidgets.QPushButton(self.tab_dataset)
        self.btn_next.setText("Next  ▶")
        self.btn_next.setEnabled(False)
        self.btn_next.setObjectName("btn_next")
        header_row.addWidget(self.btn_next)

        self.btn_delete_marked = QtWidgets.QPushButton(self.tab_dataset)
        self.btn_delete_marked.setText("Delete selected")
        self.btn_delete_marked.setEnabled(False)
        self.btn_delete_marked.setObjectName("btn_delete_marked")
        header_row.addWidget(self.btn_delete_marked)

        self.verticalLayout.addLayout(header_row)

        # Side-by-side images
        self.images_layout = QtWidgets.QHBoxLayout()
        self.images_layout.setSpacing(10)

        self.label_lwir = QtWidgets.QLabel(self.tab_dataset)
        self.label_lwir.setMinimumSize(QtCore.QSize(100, 100))  # Reduced to allow narrower window
        self.label_lwir.setStyleSheet(
            f"border: 1px solid {style.GROUP_BORDER}; background: {style.GROUP_BG};"
        )
        self.label_lwir.setText("")
        self.label_lwir.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_lwir.setScaledContents(False)  # Keep aspect ratio via code scaling
        self.label_lwir.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.label_lwir.setObjectName("label_lwir")
        self.images_layout.addWidget(self.label_lwir, 1)  # Peso 1 = mitad

        self.label_vis = QtWidgets.QLabel(self.tab_dataset)
        self.label_vis.setMinimumSize(QtCore.QSize(100, 100))  # Reduced to allow narrower window
        self.label_vis.setStyleSheet(
            f"border: 1px solid {style.GROUP_BORDER}; background: {style.GROUP_BG};"
        )
        self.label_vis.setText("")
        self.label_vis.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_vis.setScaledContents(False)  # Keep aspect ratio via code scaling
        self.label_vis.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.label_vis.setObjectName("label_vis")
        self.images_layout.addWidget(self.label_vis, 1)  # Peso 1 = mitad

        self.verticalLayout.addLayout(self.images_layout, 3)

        # Metadata columns (card for consistent width, light gray background)
        self.metadata_container = QtWidgets.QWidget(self.tab_dataset)
        self.metadata_container.setObjectName("metadata_card")
        # self.metadata_container.setStyleSheet(style.panel_body_style("metadata_card"))
        self.metadata_layout = QtWidgets.QHBoxLayout(self.metadata_container)
        self.metadata_layout.setSpacing(10)
        self.metadata_layout.setContentsMargins(12, 10, 12, 10)

        # LWIR metadata column (external title + panel body)
        self.metadata_lwir_column = QtWidgets.QWidget()
        lwir_col_layout = QtWidgets.QVBoxLayout(self.metadata_lwir_column)
        lwir_col_layout.setContentsMargins(0, 0, 0, 0)
        lwir_col_layout.setSpacing(6)
        self.label_lwir_title = QtWidgets.QLabel("LWIR metadata:")
        self.label_lwir_title.setStyleSheet(style.heading_style())
        lwir_col_layout.addWidget(self.label_lwir_title)
        self.metadata_lwir_panel = QtWidgets.QWidget()
        self.metadata_lwir_panel.setObjectName("metadata_lwir_panel")
        self.metadata_lwir_panel.setStyleSheet(style.panel_body_style("metadata_lwir_panel"))
        self.metadata_lwir_layout = QtWidgets.QVBoxLayout(self.metadata_lwir_panel)
        self.metadata_lwir_layout.setContentsMargins(8, 6, 8, 8)
        self.metadata_lwir_layout.setSpacing(6)
        self.text_metadata_lwir = QtWidgets.QPlainTextEdit()
        self.text_metadata_lwir.setReadOnly(True)
        self.text_metadata_lwir.setMinimumHeight(100)
        self.text_metadata_lwir.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.text_metadata_lwir.setStyleSheet(
            style.monospace_text_style()
            + f" border: 0px solid {style.CARD_BG}; background: {style.CARD_BG};"
        )
        self.metadata_lwir_layout.addWidget(self.text_metadata_lwir)
        lwir_col_layout.addWidget(self.metadata_lwir_panel)
        self.metadata_layout.addWidget(self.metadata_lwir_column, 1)

        # Visible metadata column (external title + panel body)
        self.metadata_vis_column = QtWidgets.QWidget()
        vis_col_layout = QtWidgets.QVBoxLayout(self.metadata_vis_column)
        vis_col_layout.setContentsMargins(0, 0, 0, 0)
        vis_col_layout.setSpacing(6)
        self.label_vis_title = QtWidgets.QLabel("Visible metadata:")
        self.label_vis_title.setStyleSheet(style.heading_style())
        vis_col_layout.addWidget(self.label_vis_title)
        self.metadata_vis_panel = QtWidgets.QWidget()
        self.metadata_vis_panel.setObjectName("metadata_vis_panel")
        self.metadata_vis_panel.setStyleSheet(style.panel_body_style("metadata_vis_panel"))
        self.metadata_vis_layout = QtWidgets.QVBoxLayout(self.metadata_vis_panel)
        self.metadata_vis_layout.setContentsMargins(8, 6, 8, 8)
        self.metadata_vis_layout.setSpacing(6)
        self.text_metadata_vis = QtWidgets.QPlainTextEdit()
        self.text_metadata_vis.setReadOnly(True)
        self.text_metadata_vis.setMinimumHeight(100)
        self.text_metadata_vis.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.text_metadata_vis.setStyleSheet(
            style.monospace_text_style()
            + f" border: 0px solid {style.CARD_BG}; background: {style.CARD_BG};"
        )
        self.metadata_vis_layout.addWidget(self.text_metadata_vis)
        vis_col_layout.addWidget(self.metadata_vis_panel)
        self.metadata_layout.addWidget(self.metadata_vis_column, 1)

        self.verticalLayout.addWidget(self.metadata_container, 1)

        # Stats placeholder (custom widget will replace this)
        self.stats_placeholder = QtWidgets.QWidget(self.tab_dataset)
        self.stats_placeholder.setObjectName("stats_placeholder")

        self.verticalLayout.addWidget(self.stats_placeholder)
        self.tab_widget.addTab(self.tab_dataset, "Dataset / Collection view")

        # Footer progress row (shared across tabs)
        self.btn_layout = QtWidgets.QHBoxLayout()
        self.btn_layout.setSpacing(6)
        self.btn_layout.addStretch(1)
        self.progress_placeholder = QtWidgets.QWidget(self.centralwidget)
        self.progress_placeholder.setObjectName("progress_placeholder")
        self.progress_placeholder.setMinimumWidth(260)
        self.progress_placeholder.setMaximumHeight(style.BUTTON_HEIGHT)
        self.progress_placeholder.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.progress_placeholder.setVisible(False)
        self.btn_layout.addWidget(self.progress_placeholder)
        self.main_layout.addLayout(self.btn_layout)
        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1200, 22))
        self.menu_file = QtWidgets.QMenu("File", self.menubar)
        self.menu_view = QtWidgets.QMenu("View", self.menubar)
        self.menu_dataset = QtWidgets.QMenu("Dataset", self.menubar)
        self.menu_filter = QtWidgets.QMenu("Filter", self.menu_view)
        self.menu_calibration = QtWidgets.QMenu("Calibration", self.menubar)
        self.menu_labelling = QtWidgets.QMenu("Labelling", self.menubar)
        self.menu_help = QtWidgets.QMenu("Help", self.menubar)
        self.action_set_workspace = QtGui.QAction("Set workspace directory", MainWindow)
        self.action_check_datasets = QtGui.QAction("Check datasets", MainWindow)
        self.action_load_dataset = QtGui.QAction("Load dataset…", MainWindow)
        self.action_load_recent = QtWidgets.QMenu("Load recent", self.menu_dataset)
        self.action_save_status = QtGui.QAction("Save current status", MainWindow)
        self.action_save_status.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        self.action_run_duplicate_scan = QtGui.QAction("Run duplicate sweep", MainWindow)
        self.action_run_pattern_scan = QtGui.QAction("Run pattern sweep", MainWindow)
        self.action_delete_selected = QtGui.QAction("Delete selected pairs", MainWindow)
        self.action_restore_images = QtGui.QAction("Restore from trash", MainWindow)
        self.action_reset_dataset = QtGui.QAction("Reset dataset (dangerous)…", MainWindow)
        self.action_import_calibration = QtGui.QAction("Import calibration data", MainWindow)
        self.action_clear_empty_datasets = QtGui.QAction("Clear empty dataset folders", MainWindow)

        # Detect candidates submenu (sweeps)
        self.action_detect_menu = QtWidgets.QMenu("Detect delete candidates", self.menu_dataset)

        # Untag submenu
        self.action_untag_menu = QtWidgets.QMenu("Untag delete candidates", self.menu_dataset)
        self.action_untag_all = QtGui.QAction("Untag all", MainWindow)
        self.action_untag_blurry = QtGui.QAction("Untag blurry", MainWindow)
        self.action_untag_motion = QtGui.QAction("Untag motion-blur", MainWindow)
        self.action_untag_sync = QtGui.QAction("Untag sync-mismatch", MainWindow)
        self.action_untag_missing = QtGui.QAction("Untag missing-pair", MainWindow)
        self.action_untag_duplicates = QtGui.QAction("Untag duplicates", MainWindow)

        # Delete submenu
        self.action_delete_menu = QtWidgets.QMenu("Delete marked", self.menu_dataset)

        # Restore submenu
        self.action_restore_menu = QtWidgets.QMenu("Restore from trash", self.menu_dataset)
        self.action_restore_all = QtGui.QAction("Restore all", MainWindow)
        self.action_restore_blurry = QtGui.QAction("Restore blurry", MainWindow)
        self.action_restore_motion = QtGui.QAction("Restore motion-blur", MainWindow)
        self.action_restore_sync = QtGui.QAction("Restore sync-mismatch", MainWindow)
        self.action_restore_missing = QtGui.QAction("Restore missing-pair", MainWindow)
        self.action_restore_duplicates = QtGui.QAction("Restore duplicates", MainWindow)
        self.action_delete_blurry = QtGui.QAction("Delete blurry images", MainWindow)
        self.action_delete_motion = QtGui.QAction("Delete motion-blur images", MainWindow)
        self.action_delete_duplicates = QtGui.QAction("Delete duplicate images", MainWindow)
        self.action_delete_sync = QtGui.QAction("Delete sync-mismatch images", MainWindow)

        # View menu actions
        self.action_toggle_rectified = QtGui.QAction("Undistort images", MainWindow)
        self.action_toggle_rectified.setCheckable(True)
        self.action_show_labels = QtGui.QAction("Show labels", MainWindow)
        self.action_show_labels.setCheckable(True)
        self.action_show_overlays = QtGui.QAction("Show image info overlay", MainWindow)
        self.action_show_overlays.setCheckable(True)
        self.action_show_overlays.setChecked(True)  # Default: overlays visible

        # Grid submenu
        self.menu_grid = QtWidgets.QMenu("Show Grid", MainWindow)
        self.action_grid_off = QtGui.QAction("Off", MainWindow)
        self.action_grid_off.setCheckable(True)
        self.action_grid_thirds = QtGui.QAction("Thirds", MainWindow)
        self.action_grid_thirds.setCheckable(True)
        self.action_grid_detailed = QtGui.QAction("Detailed (9×9)", MainWindow)
        self.action_grid_detailed.setCheckable(True)
        self.grid_action_group = QtGui.QActionGroup(MainWindow)
        self.grid_action_group.addAction(self.action_grid_off)
        self.grid_action_group.addAction(self.action_grid_thirds)
        self.grid_action_group.addAction(self.action_grid_detailed)
        self.grid_action_group.setExclusive(True)

        # Stereo Alignment submenu
        self.menu_stereo_alignment = QtWidgets.QMenu("Stereo Alignment", MainWindow)
        self.action_align_disabled = QtGui.QAction("Disabled", MainWindow)
        self.action_align_disabled.setCheckable(True)
        self.action_align_full = QtGui.QAction("Full View", MainWindow)
        self.action_align_full.setCheckable(True)
        self.action_align_fov_focus = QtGui.QAction("FOV Focus", MainWindow)
        self.action_align_fov_focus.setCheckable(True)
        self.action_align_max_overlap = QtGui.QAction("Max Overlap", MainWindow)
        self.action_align_max_overlap.setCheckable(True)
        self.align_action_group = QtGui.QActionGroup(MainWindow)
        self.align_action_group.addAction(self.action_align_disabled)
        self.align_action_group.addAction(self.action_align_full)
        self.align_action_group.addAction(self.action_align_fov_focus)
        self.align_action_group.addAction(self.action_align_max_overlap)
        self.align_action_group.setExclusive(True)

        # Corner Display submenu
        self.menu_corner_display = QtWidgets.QMenu("Corner Display", MainWindow)
        self.action_corners_original = QtGui.QAction("Original Only", MainWindow)
        self.action_corners_original.setCheckable(True)
        self.action_corners_subpixel = QtGui.QAction("Subpixel Only", MainWindow)
        self.action_corners_subpixel.setCheckable(True)
        self.action_corners_both = QtGui.QAction("Both (Debug)", MainWindow)
        self.action_corners_both.setCheckable(True)
        self.corner_action_group = QtGui.QActionGroup(MainWindow)
        self.corner_action_group.addAction(self.action_corners_original)
        self.corner_action_group.addAction(self.action_corners_subpixel)
        self.corner_action_group.addAction(self.action_corners_both)
        self.corner_action_group.setExclusive(True)
        self.action_corners_subpixel.setChecked(True)  # Default to subpixel

        # Use Subpixel Corners toggle for calibration computation
        self.action_use_subpixel_corners = QtGui.QAction("Use Subpixel Corners", MainWindow)
        self.action_use_subpixel_corners.setCheckable(True)
        self.action_use_subpixel_corners.setChecked(False)  # Default off (use original)

        self.action_calibration_debug = QtGui.QAction("Export calibration debug overlays", MainWindow)
        self.action_auto_calibration_search = QtGui.QAction("Auto search calibration candidates…", MainWindow)
        self.action_run_calibration = QtGui.QAction("Detect chessboards", MainWindow)
        self.action_show_help = QtGui.QAction("See help", MainWindow)
        self.action_show_help.setShortcut(QtGui.QKeySequence("Ctrl+H"))
        self.action_label_config_model = QtGui.QAction("Configure model…", MainWindow)
        self.action_label_config_labels = QtGui.QAction("Configure labels YAML…", MainWindow)
        self.action_label_current = QtGui.QAction("Run labelling on current", MainWindow)
        self.action_label_dataset = QtGui.QAction("Run labelling on dataset…", MainWindow)
        self.action_label_clear_current = QtGui.QAction("Clear labels for current", MainWindow)
        self.action_label_manual_mode = QtGui.QAction("Manual labelling mode", MainWindow)
        self.action_label_manual_mode.setCheckable(True)
        self.action_filter_all = QtGui.QAction("Show all images", MainWindow)
        self.action_filter_calibration_any = QtGui.QAction("Calibration candidates", MainWindow)
        self.action_filter_calibration_both = QtGui.QAction("Calibration with both detections", MainWindow)
        self.action_filter_calibration_partial = QtGui.QAction("Calibration with single detection", MainWindow)
        self.action_filter_calibration_missing = QtGui.QAction("Calibration without detections", MainWindow)
        self.action_filter_delete_candidates = QtGui.QAction("Delete candidates", MainWindow)
        self.action_run_quality_scan = QtGui.QAction("Run blur/motion sweep", MainWindow)
        self.action_filter_delete_manual = QtGui.QAction("Delete (manual)", MainWindow)
        self.action_filter_delete_duplicate = QtGui.QAction("Delete (duplicate)", MainWindow)
        self.action_filter_delete_blurry = QtGui.QAction("Delete (blurry)", MainWindow)
        self.action_filter_delete_motion = QtGui.QAction("Delete (motion)", MainWindow)
        self.action_filter_delete_sync = QtGui.QAction("Delete (sync)", MainWindow)
        self.action_filter_delete_missing_pair = QtGui.QAction("Delete (missing pair)", MainWindow)
        self.action_delete_missing_pair = QtGui.QAction("Delete missing-pair", MainWindow)
        for action in (
            self.action_filter_all,
            self.action_filter_calibration_any,
            self.action_filter_calibration_both,
            self.action_filter_calibration_partial,
            self.action_filter_calibration_missing,
            self.action_filter_delete_candidates,
            self.action_filter_delete_manual,
            self.action_filter_delete_duplicate,
            self.action_filter_delete_blurry,
            self.action_filter_delete_motion,
            self.action_filter_delete_sync,
            self.action_filter_delete_missing_pair,
        ):
            action.setCheckable(True)
        self.action_calibration_refine = QtGui.QAction("Refine chessboard corners", MainWindow)
        self.action_calibration_compute = QtGui.QAction("Compute calibration matrices", MainWindow)
        self.action_calibration_extrinsic = QtGui.QAction("Compute extrinsic transform", MainWindow)
        self.action_calibration_check = QtGui.QAction("Check calibration report", MainWindow)
        self.action_exit = QtGui.QAction("Exit", MainWindow)
        self.action_exit.setShortcut(QtGui.QKeySequence("Ctrl+Q"))
        qt_style = MainWindow.style()
        self.action_load_dataset.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton))
        self.action_load_recent.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton))
        self.action_save_status.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton))
        self.action_exit.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogCloseButton))
        self.action_delete_selected.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TrashIcon))
        self.action_restore_images.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload))
        self.action_reset_dataset.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning))
        self.action_show_help.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogHelpButton))
        self.action_run_duplicate_scan.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay))
        self.action_run_quality_scan.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay))
        self.action_run_pattern_scan.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay))
        self.action_clear_empty_datasets.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TrashIcon))

        # Delete action icons
        self.action_delete_duplicates.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TrashIcon))
        self.action_delete_blurry.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TrashIcon))
        self.action_delete_motion.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TrashIcon))
        self.action_delete_sync.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TrashIcon))
        self.action_delete_missing_pair.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TrashIcon))

        # Untag action icons
        self.action_untag_all.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogResetButton))
        self.action_untag_blurry.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogResetButton))
        self.action_untag_motion.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogResetButton))
        self.action_untag_sync.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogResetButton))
        self.action_untag_missing.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogResetButton))
        self.action_untag_duplicates.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogResetButton))

        # Restore action icons
        self.action_restore_all.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload))
        self.action_restore_blurry.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload))
        self.action_restore_motion.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload))
        self.action_restore_sync.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload))
        self.action_restore_missing.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload))
        self.action_restore_duplicates.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload))

        # Submenu icons
        self.action_detect_menu.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay))
        self.action_untag_menu.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogResetButton))
        self.action_delete_menu.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TrashIcon))
        self.action_restore_menu.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload))

        # Calibration icons
        self.action_auto_calibration_search.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay))

        self.menu_file.addAction(self.action_save_status)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.action_exit)

        # Dataset menu reorganization
        self.menu_dataset.addAction(self.action_load_dataset)
        self.menu_dataset.addMenu(self.action_load_recent)
        self.menu_dataset.addSeparator()

        # Detect candidates submenu (sweeps)
        self.action_detect_menu.addAction(self.action_run_duplicate_scan)
        self.action_detect_menu.addAction(self.action_run_quality_scan)
        self.action_detect_menu.addAction(self.action_run_pattern_scan)
        self.menu_dataset.addMenu(self.action_detect_menu)

        # Untag delete candidates submenu
        self.action_untag_menu.addAction(self.action_untag_all)
        self.action_untag_menu.addSeparator()
        self.action_untag_menu.addAction(self.action_untag_blurry)
        self.action_untag_menu.addAction(self.action_untag_motion)
        self.action_untag_menu.addAction(self.action_untag_sync)
        self.action_untag_menu.addAction(self.action_untag_missing)
        self.action_untag_menu.addAction(self.action_untag_duplicates)
        self.menu_dataset.addMenu(self.action_untag_menu)

        # Delete marked submenu
        self.action_delete_menu.addAction(self.action_delete_selected)
        self.action_delete_menu.addSeparator()
        self.action_delete_menu.addAction(self.action_delete_blurry)
        self.action_delete_menu.addAction(self.action_delete_motion)
        self.action_delete_menu.addAction(self.action_delete_sync)
        self.action_delete_menu.addAction(self.action_delete_missing_pair)
        self.action_delete_menu.addAction(self.action_delete_duplicates)
        self.menu_dataset.addMenu(self.action_delete_menu)

        # Restore from trash submenu
        self.action_restore_menu.addAction(self.action_restore_all)
        self.action_restore_menu.addSeparator()
        self.action_restore_menu.addAction(self.action_restore_blurry)
        self.action_restore_menu.addAction(self.action_restore_motion)
        self.action_restore_menu.addAction(self.action_restore_sync)
        self.action_restore_menu.addAction(self.action_restore_missing)
        self.action_restore_menu.addAction(self.action_restore_duplicates)
        self.menu_dataset.addMenu(self.action_restore_menu)

        self.menu_dataset.addSeparator()
        self.menu_dataset.addAction(self.action_reset_dataset)

        # View menu with submenus
        self.menu_view.addAction(self.action_toggle_rectified)
        self.menu_view.addSeparator()

        # Grid submenu
        self.menu_grid.addAction(self.action_grid_off)
        self.menu_grid.addAction(self.action_grid_thirds)
        self.menu_grid.addAction(self.action_grid_detailed)
        self.menu_view.addMenu(self.menu_grid)

        # Stereo Alignment submenu
        self.menu_stereo_alignment.addAction(self.action_align_disabled)
        self.menu_stereo_alignment.addAction(self.action_align_full)
        self.menu_stereo_alignment.addAction(self.action_align_fov_focus)
        self.menu_stereo_alignment.addAction(self.action_align_max_overlap)
        self.menu_view.addMenu(self.menu_stereo_alignment)

        # Corner Display submenu
        self.menu_corner_display.addAction(self.action_corners_original)
        self.menu_corner_display.addAction(self.action_corners_subpixel)
        self.menu_corner_display.addAction(self.action_corners_both)
        self.menu_view.addMenu(self.menu_corner_display)

        self.menu_view.addSeparator()
        self.menu_view.addAction(self.action_show_labels)
        self.menu_view.addAction(self.action_show_overlays)
        self.menu_view.addSeparator()
        self.menu_filter.addAction(self.action_filter_all)
        self.menu_filter.addSeparator()
        self.menu_filter.addAction(self.action_filter_calibration_any)
        self.menu_filter.addAction(self.action_filter_calibration_both)
        self.menu_filter.addAction(self.action_filter_calibration_partial)
        self.menu_filter.addAction(self.action_filter_calibration_missing)
        self.menu_filter.addSeparator()
        self.menu_filter.addAction(self.action_filter_delete_candidates)
        self.menu_filter.addAction(self.action_filter_delete_manual)
        self.menu_filter.addAction(self.action_filter_delete_duplicate)
        self.menu_filter.addAction(self.action_filter_delete_blurry)
        self.menu_filter.addAction(self.action_filter_delete_motion)
        self.menu_filter.addAction(self.action_filter_delete_sync)
        self.menu_filter.addAction(self.action_filter_delete_missing_pair)
        self.menu_view.addMenu(self.menu_filter)
        self.menu_calibration.addAction(self.action_auto_calibration_search)
        self.menu_calibration.addAction(self.action_run_calibration)
        self.menu_calibration.addAction(self.action_calibration_refine)
        self.menu_calibration.addSeparator()
        self.menu_calibration.addAction(self.action_use_subpixel_corners)
        self.menu_calibration.addSeparator()
        self.menu_calibration.addAction(self.action_calibration_compute)
        self.menu_calibration.addAction(self.action_calibration_extrinsic)
        self.menu_calibration.addSeparator()
        self.menu_calibration.addAction(self.action_calibration_check)
        self.menu_calibration.addSeparator()
        self.menu_calibration.addAction(self.action_calibration_debug)
        self.menu_calibration.addSeparator()
        self.menu_calibration.addAction(self.action_import_calibration)
        self.menu_labelling.addAction(self.action_label_config_model)
        self.menu_labelling.addAction(self.action_label_config_labels)
        self.menu_labelling.addSeparator()
        self.menu_labelling.addAction(self.action_label_current)
        self.menu_labelling.addAction(self.action_label_dataset)
        self.menu_labelling.addSeparator()
        self.menu_labelling.addAction(self.action_label_clear_current)
        self.menu_labelling.addSeparator()
        self.menu_labelling.addAction(self.action_label_manual_mode)
        self.menu_help.addAction(self.action_show_help)
        self.menubar.addAction(self.menu_file.menuAction())
        self.menubar.addAction(self.menu_view.menuAction())
        self.menubar.addAction(self.menu_dataset.menuAction())
        self.menubar.addAction(self.menu_calibration.menuAction())
        self.menubar.addAction(self.menu_labelling.menuAction())
        self.menubar.addAction(self.menu_help.menuAction())
        MainWindow.setMenuBar(self.menubar)
