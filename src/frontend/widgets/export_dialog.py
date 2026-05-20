"""Export dialog: configure and run a dataset export to disk.

Reuses the workspace scan to populate a tree of datasets/collections,
runs ``backend.services.export.exporter.run_export`` in a worker thread,
and reports progress via a built-in QProgressBar.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from backend.services.export.exporter import (
    DatasetExportPlan,
    ExportRequest,
    ExportResult,
    run_export,
)
from backend.services.export.image_pipeline import (
    RESOLUTION_DOWNSAMPLE,
    RESOLUTION_UPSAMPLE,
    TransformParams,
)
from backend.services.workspace_inspector import WorkspaceDatasetInfo, scan_workspace
from common.log_utils import log_warning
from frontend.widgets import style


# Roles to stash the dataset path / has-calibration flag on tree items.
_PATH_ROLE = Qt.ItemDataRole.UserRole + 1
_HAS_CALIB_ROLE = Qt.ItemDataRole.UserRole + 2


class _ExportWorker(QObject):
    """Runs ``run_export`` in a thread; emits progress and completion."""
    progress = pyqtSignal(str, int, int)
    finished = pyqtSignal(object)   # ExportResult or Exception
    failed = pyqtSignal(str)

    def __init__(self, request: ExportRequest) -> None:
        super().__init__()
        self.request = request
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            result = run_export(
                self.request,
                progress=lambda m, c, t: self.progress.emit(m, c, t),
                cancelled=lambda: self._cancel,
            )
            self.finished.emit(result)
        except Exception as e:
            log_warning(f"Export worker failed: {e}", "EXPORT")
            self.failed.emit(str(e))


class ExportDialog(QDialog):
    """Configure and run an export."""

    def __init__(self, workspace_path: Path, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Dataset")
        self.setMinimumSize(640, 600)
        self._workspace_path = workspace_path
        self._worker_thread: Optional[QThread] = None
        self._worker: Optional[_ExportWorker] = None

        self._build_ui()
        self._populate_tree()
        self._update_alignment_availability()

    # ── UI ───────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # ── Card ────────────────────────────────────────────────────────────
        card = QWidget(self)
        card.setObjectName("export_card")
        card.setStyleSheet(style.card_style("export_card"))
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 14, 14, 14)
        card_layout.setSpacing(10)

        # Output directory
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output directory:"))
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Pick the directory where the export will live")
        out_row.addWidget(self.output_edit, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse)
        out_row.addWidget(browse)
        card_layout.addLayout(out_row)

        self.output_hint = QLabel("")
        self.output_hint.setVisible(False)
        card_layout.addWidget(self.output_hint)

        # Datasets tree
        card_layout.addWidget(style.section_heading_label("Datasets to export"))
        tree_panel, tree_layout = style.make_panel("export_tree_panel")

        select_row = QHBoxLayout()
        btn_select_all = QPushButton("Select all")
        btn_select_all.clicked.connect(lambda: self._set_all_check(Qt.CheckState.Checked))
        btn_deselect_all = QPushButton("Deselect all")
        btn_deselect_all.clicked.connect(lambda: self._set_all_check(Qt.CheckState.Unchecked))
        btn_invert = QPushButton("Invert selection")
        btn_invert.clicked.connect(self._invert_selection)
        select_row.addWidget(btn_select_all)
        select_row.addWidget(btn_deselect_all)
        select_row.addWidget(btn_invert)
        select_row.addStretch(1)
        tree_layout.addLayout(select_row)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Images", "Labels", "Status"])
        self.tree.setColumnWidth(0, 320)
        self.tree.setColumnWidth(1, 80)
        self.tree.setColumnWidth(2, 80)
        tree_layout.addWidget(self.tree)
        card_layout.addWidget(tree_panel, 1)

        # ── 3-column options row: Channels | Transforms (2 inner cols) ──────
        options_row = QHBoxLayout()
        options_row.setSpacing(10)

        # — Channels column —
        ch_col = QVBoxLayout()
        ch_col.setSpacing(4)
        ch_col.addWidget(style.section_heading_label("Channels"))
        ch_panel, ch_layout = style.make_panel("export_channels_panel")
        self.cb_lwir = QCheckBox("LWIR")
        self.cb_lwir.setChecked(True)
        self.cb_lwir.toggled.connect(self._update_alignment_availability)
        self.cb_visible = QCheckBox("Visible")
        self.cb_visible.setChecked(True)
        self.cb_visible.toggled.connect(self._update_alignment_availability)
        ch_layout.addWidget(self.cb_lwir)
        ch_layout.addWidget(self.cb_visible)
        ch_layout.addStretch(1)
        ch_col.addWidget(ch_panel)
        options_row.addLayout(ch_col, 1)

        # — Transforms column (2 inner sub-columns) —
        tr_col = QVBoxLayout()
        tr_col.setSpacing(4)
        tr_col.addWidget(style.section_heading_label("Transforms"))
        tr_panel, tr_layout = style.make_panel("export_transforms_panel", spacing=6)

        tr_inner = QHBoxLayout()
        tr_inner.setSpacing(16)

        # Left: undistort / align / parallax
        checks_col = QVBoxLayout()
        checks_col.setSpacing(6)
        self.cb_undistort = QCheckBox("Undistort (intrinsic calibration)")
        self.cb_undistort.setChecked(True)
        self.cb_align = QCheckBox("FOV alignment (extrinsic calibration)")
        self.cb_align.setChecked(True)
        self.cb_align.toggled.connect(self._update_alignment_availability)
        self.cb_parallax = QCheckBox("Parallax correction")
        self.cb_parallax.setChecked(True)
        checks_col.addWidget(self.cb_undistort)
        checks_col.addWidget(self.cb_align)
        checks_col.addWidget(self.cb_parallax)
        checks_col.addStretch(1)
        tr_inner.addLayout(checks_col, 1)

        # Right: resolution
        res_col = QVBoxLayout()
        res_col.setSpacing(6)
        res_lbl = QLabel("Resolution:")
        res_lbl.setStyleSheet(style.SECTION_TITLE_STYLE)
        self.rb_upsample = QRadioButton("Upsample to largest")
        self.rb_upsample.setChecked(True)
        self.rb_downsample = QRadioButton("Downsample to smallest")
        self._res_group = QButtonGroup(self)
        self._res_group.addButton(self.rb_upsample)
        self._res_group.addButton(self.rb_downsample)
        res_col.addWidget(res_lbl)
        res_col.addWidget(self.rb_upsample)
        res_col.addWidget(self.rb_downsample)
        res_col.addStretch(1)
        tr_inner.addLayout(res_col, 1)

        tr_layout.addLayout(tr_inner)
        self.transform_hint = QLabel("")
        self.transform_hint.setVisible(False)
        self.transform_hint.setWordWrap(True)
        tr_layout.addWidget(self.transform_hint)
        tr_col.addWidget(tr_panel)
        options_row.addLayout(tr_col, 2)

        card_layout.addLayout(options_row)

        # Labels info
        card_layout.addWidget(style.section_heading_label("Labels"))
        lbl_panel, lbl_layout = style.make_panel("export_labels_panel")
        lbl_layout.addWidget(QLabel(
            "Source: manual + reviewed (auto-pending excluded). "
            "Output: union of both channels, cross-channel labels projected via H."
        ))
        card_layout.addWidget(lbl_panel)

        layout.addWidget(card, 1)

        # ── Progress + buttons (outside card, always visible) ──────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")
        layout.addWidget(self.progress_bar)

        self.progress_msg = QLabel("")
        self.progress_msg.setVisible(False)
        self.progress_msg.setWordWrap(True)
        layout.addWidget(self.progress_msg)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Export")
        self.button_box.accepted.connect(self._on_export)
        self.button_box.rejected.connect(self._on_cancel_clicked)
        layout.addWidget(self.button_box)

    # ── tree population ─────────────────────────────────────────────────

    def _populate_tree(self) -> None:
        entries = scan_workspace(self._workspace_path)
        # ``scan_workspace`` returns a flat list with collections AND their
        # children at the same level. Skip child entries here — they're
        # already attached inside their collection's ``children`` list.
        top_entries = [e for e in entries if not e.parent]
        self.tree.blockSignals(True)
        try:
            for entry in top_entries:
                top = self._build_tree_item(entry)
                self.tree.addTopLevelItem(top)
                top.setExpanded(True)
        finally:
            self.tree.blockSignals(False)
        self._refresh_calibration_status()

    def _build_tree_item(self, info: WorkspaceDatasetInfo) -> QTreeWidgetItem:
        label = info.name + ("  (collection)" if info.is_collection else "")
        images = str(info.stats.total_pairs) if info.stats else ""
        labels = str(info.labels_total) if info.labels_total else ""
        item = QTreeWidgetItem([label, images, labels, ""])
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate)
        item.setCheckState(0, Qt.CheckState.Checked)
        item.setData(0, _PATH_ROLE, str(info.path))
        for child in info.children:
            item.addChild(self._build_tree_item(child))
        return item

    def _set_all_check(self, state: Qt.CheckState) -> None:
        """Apply a check state to every leaf in the tree (collections follow tristate)."""
        self.tree.blockSignals(True)
        try:
            def _walk(item: QTreeWidgetItem) -> None:
                if item.childCount() == 0:
                    item.setCheckState(0, state)
                else:
                    for i in range(item.childCount()):
                        _walk(item.child(i))
            for i in range(self.tree.topLevelItemCount()):
                _walk(self.tree.topLevelItem(i))
        finally:
            self.tree.blockSignals(False)

    def _invert_selection(self) -> None:
        """Flip every leaf's checkstate."""
        self.tree.blockSignals(True)
        try:
            def _walk(item: QTreeWidgetItem) -> None:
                if item.childCount() == 0:
                    new = (
                        Qt.CheckState.Unchecked
                        if item.checkState(0) == Qt.CheckState.Checked
                        else Qt.CheckState.Checked
                    )
                    item.setCheckState(0, new)
                else:
                    for i in range(item.childCount()):
                        _walk(item.child(i))
            for i in range(self.tree.topLevelItemCount()):
                _walk(self.tree.topLevelItem(i))
        finally:
            self.tree.blockSignals(False)

    def _refresh_calibration_status(self) -> None:
        """Walk the tree and tag each leaf as having calibration or not."""
        from backend.services.export.exporter import _load_calibration_for

        def _walk(item: QTreeWidgetItem) -> None:
            if item.childCount() == 0:
                path = Path(item.data(0, _PATH_ROLE))
                has_calib = _load_calibration_for(path) is not None
                item.setData(0, _HAS_CALIB_ROLE, has_calib)
                item.setText(3, "✓ calibration" if has_calib else "no calibration")
                if not has_calib:
                    item.setForeground(3, Qt.GlobalColor.darkRed)
            else:
                for i in range(item.childCount()):
                    _walk(item.child(i))

        for i in range(self.tree.topLevelItemCount()):
            _walk(self.tree.topLevelItem(i))

    # ── reactive updates ────────────────────────────────────────────────

    def _update_alignment_availability(self) -> None:
        """Disable alignment / parallax / resolution if only one channel is selected."""
        both = self.cb_lwir.isChecked() and self.cb_visible.isChecked()
        none = not (self.cb_lwir.isChecked() or self.cb_visible.isChecked())

        self.cb_align.setEnabled(both)
        if not both:
            self.cb_align.setChecked(False)

        self.cb_parallax.setEnabled(both and self.cb_align.isChecked())
        if not (both and self.cb_align.isChecked()):
            self.cb_parallax.setChecked(False)

        self.rb_upsample.setEnabled(both and self.cb_align.isChecked())
        self.rb_downsample.setEnabled(both and self.cb_align.isChecked())

        if none:
            msg = "Select at least one channel."
        elif not both:
            msg = (
                "Single-channel export: no alignment between channels. "
                "Other-channel labels still get projected onto the exported image."
            )
        else:
            msg = ""
        self.transform_hint.setText(msg)
        self.transform_hint.setVisible(bool(msg))

    def _on_browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self.output_edit.setText(path)
            self.output_hint.setText(
                f"Will create: {Path(path) / (self._workspace_path.name + '_export')}"
            )
            self.output_hint.setVisible(True)

    # ── selection collection ────────────────────────────────────────────

    def _gather_selected_datasets(self) -> List[DatasetExportPlan]:
        """Walk the tree and collect checked LEAF dataset paths.

        Skips nodes without calibration when alignment is requested.
        """
        plans: List[DatasetExportPlan] = []
        require_calib = self.cb_align.isChecked() and self.cb_lwir.isChecked() and self.cb_visible.isChecked()

        def _visit(item: QTreeWidgetItem) -> None:
            if item.childCount() == 0:
                if item.checkState(0) == Qt.CheckState.Unchecked:
                    return
                path = Path(item.data(0, _PATH_ROLE))
                has_calib = bool(item.data(0, _HAS_CALIB_ROLE))
                if require_calib and not has_calib:
                    return
                rel = self._relative_to_workspace(path)
                plans.append(DatasetExportPlan(dataset_path=path, relative_path=rel))
                return
            for i in range(item.childCount()):
                _visit(item.child(i))

        for i in range(self.tree.topLevelItemCount()):
            _visit(self.tree.topLevelItem(i))
        return plans

    def _gather_excluded_no_calib(self) -> List[str]:
        """Return relative paths of checked datasets that will be skipped
        because alignment is requested but they have no calibration."""
        require_calib = self.cb_align.isChecked() and self.cb_lwir.isChecked() and self.cb_visible.isChecked()
        if not require_calib:
            return []
        excluded: List[str] = []

        def _visit(item: QTreeWidgetItem) -> None:
            if item.childCount() == 0:
                if item.checkState(0) == Qt.CheckState.Unchecked:
                    return
                if not bool(item.data(0, _HAS_CALIB_ROLE)):
                    path = Path(item.data(0, _PATH_ROLE))
                    excluded.append(str(self._relative_to_workspace(path)))
                return
            for i in range(item.childCount()):
                _visit(item.child(i))

        for i in range(self.tree.topLevelItemCount()):
            _visit(self.tree.topLevelItem(i))
        return excluded

    def _relative_to_workspace(self, path: Path) -> Path:
        try:
            return path.relative_to(self._workspace_path)
        except ValueError:
            return Path(path.name)

    # ── export run ──────────────────────────────────────────────────────

    def _channels(self) -> Tuple[str, ...]:
        chans: List[str] = []
        if self.cb_lwir.isChecked():
            chans.append("lwir")
        if self.cb_visible.isChecked():
            chans.append("visible")
        return tuple(chans)

    def _build_request(self) -> Optional[ExportRequest]:
        out_dir = self.output_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "Export", "Pick an output directory first.")
            return None
        channels = self._channels()
        if not channels:
            QMessageBox.warning(self, "Export", "Select at least one channel.")
            return None

        plans = self._gather_selected_datasets()
        excluded_no_calib = self._gather_excluded_no_calib()

        if not plans:
            QMessageBox.warning(
                self,
                "Export",
                "No datasets selected. Datasets without calibration are skipped when "
                "alignment is requested — uncheck FOV alignment to include them.",
            )
            return None

        # Warn upfront if some checked datasets will be silently dropped.
        if excluded_no_calib:
            names = "\n".join(f"  • {p}" for p in excluded_no_calib)
            answer = QMessageBox.question(
                self,
                "Datasets without calibration",
                "FOV alignment is enabled, so the following selected datasets "
                "will be skipped because they lack calibration:\n\n"
                f"{names}\n\n"
                "Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return None

        params = TransformParams(
            undistort=self.cb_undistort.isChecked(),
            align_fov=self.cb_align.isChecked(),
            parallax=self.cb_parallax.isChecked(),
            resolution_mode=(
                RESOLUTION_UPSAMPLE if self.rb_upsample.isChecked() else RESOLUTION_DOWNSAMPLE
            ),
        )
        return ExportRequest(
            workspace_path=self._workspace_path,
            output_dir=Path(out_dir),
            datasets=plans,
            channels=channels,
            params=params,
        )

    def _on_export(self) -> None:
        request = self._build_request()
        if request is None:
            return

        self._set_running(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting…")

        self._worker_thread = QThread(self)
        self._worker = _ExportWorker(request)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker_thread.start()

    def _on_progress(self, message: str, current: int, total: int) -> None:
        if total > 0:
            pct = int(min(100, max(0, (current * 100) / total)))
            self.progress_bar.setValue(pct)
            self.progress_bar.setFormat(f"{pct}%  ({current}/{total})")
        self.progress_msg.setText(message)
        self.progress_msg.setVisible(bool(message))

    def _on_finished(self, result: ExportResult) -> None:
        self._teardown_worker()
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Done")

        skipped_no_calib = [r for r in result.datasets if r.skipped_no_calibration]
        errored = [r for r in result.datasets if r.error]

        if result.cancelled:
            msg = "Export cancelled."
        else:
            msg = f"✓ {result.total_images} images, {result.total_labels} labels → {result.output_root}"
            warnings: List[str] = []
            if skipped_no_calib:
                warnings.append(
                    f"{len(skipped_no_calib)} dataset(s) skipped: no calibration available"
                )
            if errored:
                warnings.append(f"{len(errored)} dataset(s) failed (see log)")
            if warnings:
                msg += "\n⚠ " + "; ".join(warnings)

            if skipped_no_calib or errored:
                detail_lines = []
                if skipped_no_calib:
                    detail_lines.append("Skipped — no calibration available:")
                    detail_lines.extend(f"  • {r.relative_path}" for r in skipped_no_calib)
                if errored:
                    detail_lines.append("\nFailed during export:")
                    detail_lines.extend(
                        f"  • {r.relative_path}: {r.error}" for r in errored
                    )
                QMessageBox.warning(self, "Export completed with warnings", "\n".join(detail_lines))

        self.progress_msg.setText(msg)
        self.progress_msg.setVisible(True)
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setText("Close")
        self._set_running(False)

    def _on_failed(self, message: str) -> None:
        self._teardown_worker()
        self.progress_bar.setFormat("Failed")
        self.progress_msg.setText(f"Failed: {message}")
        self.progress_msg.setVisible(True)
        QMessageBox.critical(self, "Export failed", message)
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setText("Close")
        self._set_running(False)

    def _on_cancel_clicked(self) -> None:
        if self._worker is not None and self._worker_thread is not None and self._worker_thread.isRunning():
            self._worker.cancel()
            self.progress_msg.setText("Cancelling…")
            self.progress_msg.setVisible(True)
            return
        self.reject()

    def _teardown_worker(self) -> None:
        if self._worker_thread is not None:
            self._worker_thread.quit()
            self._worker_thread.wait()
        self._worker_thread = None
        self._worker = None

    def _set_running(self, running: bool) -> None:
        """Disable inputs while an export is running.

        After a run finishes, controls are re-enabled so the user can
        adjust settings and trigger another export without re-opening
        the dialog.
        """
        self.tree.setEnabled(not running)
        self.output_edit.setEnabled(not running)
        self.cb_lwir.setEnabled(not running)
        self.cb_visible.setEnabled(not running)
        self.cb_undistort.setEnabled(not running)
        self.cb_align.setEnabled(not running)
        self.cb_parallax.setEnabled(not running)
        self.rb_upsample.setEnabled(not running)
        self.rb_downsample.setEnabled(not running)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(not running)
        if not running:
            self._update_alignment_availability()


__all__ = ["ExportDialog"]
