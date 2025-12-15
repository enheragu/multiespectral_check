
from PyQt6 import QtCore, QtGui, QtWidgets
from widgets import style


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout.setSpacing(10)
        
        # Side-by-side images
        self.images_layout = QtWidgets.QHBoxLayout()
        self.images_layout.setSpacing(10)
        
        self.label_lwir = QtWidgets.QLabel(self.centralwidget)
        self.label_lwir.setMinimumSize(QtCore.QSize(500, 500))
        self.label_lwir.setStyleSheet(
            f"border: 1px solid {style.GROUP_BORDER}; background: {style.GROUP_BG};"
        )
        self.label_lwir.setText("")
        self.label_lwir.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_lwir.setScaledContents(False)  # Keep aspect ratio via code scaling
        self.label_lwir.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.label_lwir.setObjectName("label_lwir")
        self.images_layout.addWidget(self.label_lwir, 1)  # Peso 1 = mitad
        
        self.label_vis = QtWidgets.QLabel(self.centralwidget)
        self.label_vis.setMinimumSize(QtCore.QSize(500, 500))
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
        self.metadata_container = QtWidgets.QWidget(self.centralwidget)
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
        self.text_metadata_lwir.setMinimumHeight(140)
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
        self.text_metadata_vis.setMinimumHeight(140)
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
        self.stats_placeholder = QtWidgets.QWidget(self.centralwidget)
        self.stats_placeholder.setObjectName("stats_placeholder")

        # Buttons
        self.btn_layout = QtWidgets.QHBoxLayout()

        self.progress_placeholder = QtWidgets.QWidget(self.centralwidget)
        self.progress_placeholder.setObjectName("progress_placeholder")
        self.progress_placeholder.setMinimumWidth(240)
        self.progress_placeholder.setMaximumHeight(24)
        self.progress_placeholder.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.progress_placeholder.setVisible(False)
        self.btn_layout.addWidget(self.progress_placeholder)
        self.btn_layout.addSpacing(10)

        spacer = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.btn_layout.addItem(spacer)
        
        self.btn_prev = QtWidgets.QPushButton(self.centralwidget)
        self.btn_prev.setText("◀ Previous")
        self.btn_prev.setEnabled(False)
        self.btn_prev.setObjectName("btn_prev")
        self.btn_layout.addWidget(self.btn_prev)
        
        self.btn_next = QtWidgets.QPushButton(self.centralwidget)
        self.btn_next.setText("Next ▶")
        self.btn_next.setEnabled(False)
        self.btn_next.setObjectName("btn_next")
        self.btn_layout.addWidget(self.btn_next)

        self.btn_delete_marked = QtWidgets.QPushButton(self.centralwidget)
        self.btn_delete_marked.setText("Delete selected")
        self.btn_delete_marked.setEnabled(False)
        self.btn_delete_marked.setObjectName("btn_delete_marked")
        self.btn_layout.addWidget(self.btn_delete_marked)
        
        self.verticalLayout.addLayout(self.btn_layout)
        self.verticalLayout.addWidget(self.stats_placeholder)
        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1200, 22))
        self.menu_file = QtWidgets.QMenu("File", self.menubar)
        self.menu_view = QtWidgets.QMenu("View", self.menubar)
        self.menu_dataset = QtWidgets.QMenu("Dataset", self.menubar)
        self.menu_filter = QtWidgets.QMenu("Filter", self.menu_view)
        self.menu_calibration = QtWidgets.QMenu("Calibration", self.menubar)
        self.menu_help = QtWidgets.QMenu("Help", self.menubar)
        self.action_load_dataset = QtGui.QAction("Load dataset", MainWindow)
        self.action_save_status = QtGui.QAction("Save current status", MainWindow)
        self.action_save_status.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        self.action_run_duplicate_scan = QtGui.QAction("Run duplicate sweep", MainWindow)
        self.action_delete_selected = QtGui.QAction("Delete selected pairs", MainWindow)
        self.action_restore_images = QtGui.QAction("Restore from trash", MainWindow)
        self.action_import_calibration = QtGui.QAction("Import calibration data", MainWindow)
        self.action_clear_empty_datasets = QtGui.QAction("Clear empty dataset folders", MainWindow)
        self.action_toggle_rectified = QtGui.QAction("View rectified images", MainWindow)
        self.action_toggle_rectified.setCheckable(True)
        self.action_toggle_grid = QtGui.QAction("Show grid", MainWindow)
        self.action_toggle_grid.setCheckable(True)
        self.action_toggle_grid.setChecked(True)
        self.action_calibration_debug = QtGui.QAction("Export calibration debug overlays", MainWindow)
        self.action_run_calibration = QtGui.QAction("Re-run calibration now", MainWindow)
        self.action_show_help = QtGui.QAction("See help", MainWindow)
        self.action_show_help.setShortcut(QtGui.QKeySequence("Ctrl+H"))
        self.action_filter_all = QtGui.QAction("Show all images", MainWindow)
        self.action_filter_calibration_any = QtGui.QAction("Calibration candidates", MainWindow)
        self.action_filter_calibration_both = QtGui.QAction("Calibration with both detections", MainWindow)
        self.action_filter_calibration_partial = QtGui.QAction("Calibration with single detection", MainWindow)
        self.action_filter_calibration_missing = QtGui.QAction("Calibration without detections", MainWindow)
        self.action_filter_calibration_suspect = QtGui.QAction("Calibration suspect detections", MainWindow)
        for action in (
            self.action_filter_all,
            self.action_filter_calibration_any,
            self.action_filter_calibration_both,
            self.action_filter_calibration_partial,
            self.action_filter_calibration_missing,
            self.action_filter_calibration_suspect,
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
        self.action_save_status.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton))
        self.action_exit.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogCloseButton))
        self.action_delete_selected.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TrashIcon))
        self.action_restore_images.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload))
        self.action_show_help.setIcon(qt_style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogHelpButton))
        self.menu_file.addAction(self.action_save_status)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.action_exit)
        self.menu_dataset.addAction(self.action_load_dataset)
        self.menu_dataset.addAction(self.action_run_duplicate_scan)
        self.menu_dataset.addSeparator()
        self.menu_dataset.addAction(self.action_delete_selected)
        self.menu_dataset.addAction(self.action_restore_images)
        self.menu_dataset.addSeparator()
        self.menu_dataset.addAction(self.action_clear_empty_datasets)
        self.menu_view.addAction(self.action_toggle_rectified)
        self.menu_view.addAction(self.action_toggle_grid)
        self.menu_view.addSeparator()
        self.menu_filter.addAction(self.action_filter_all)
        self.menu_filter.addAction(self.action_filter_calibration_any)
        self.menu_filter.addAction(self.action_filter_calibration_both)
        self.menu_filter.addAction(self.action_filter_calibration_partial)
        self.menu_filter.addAction(self.action_filter_calibration_missing)
        self.menu_filter.addAction(self.action_filter_calibration_suspect)
        self.menu_view.addMenu(self.menu_filter)
        self.menu_calibration.addAction(self.action_run_calibration)
        self.menu_calibration.addAction(self.action_calibration_refine)
        self.menu_calibration.addAction(self.action_calibration_compute)
        self.menu_calibration.addAction(self.action_calibration_extrinsic)
        self.menu_calibration.addSeparator()
        self.menu_calibration.addAction(self.action_calibration_check)
        self.menu_calibration.addSeparator()
        self.menu_calibration.addAction(self.action_calibration_debug)
        self.menu_calibration.addSeparator()
        self.menu_calibration.addAction(self.action_import_calibration)
        self.menu_help.addAction(self.action_show_help)
        self.menubar.addAction(self.menu_file.menuAction())
        self.menubar.addAction(self.menu_view.menuAction())
        self.menubar.addAction(self.menu_dataset.menuAction())
        self.menubar.addAction(self.menu_calibration.menuAction())
        self.menubar.addAction(self.menu_help.menuAction())
        MainWindow.setMenuBar(self.menubar)
