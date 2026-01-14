"""Workspace dialog to review datasets and edit per-dataset notes."""
from __future__ import annotations

import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Callable, Dict, List, Optional, Tuple

from PyQt6.QtCore import (QObject, QRunnable, Qt, QThreadPool, QTimer,
                          pyqtSignal)
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox,
                             QDialog, QDialogButtonBox, QHBoxLayout,
                             QHeaderView, QInputDialog, QLabel, QMessageBox,
                             QPlainTextEdit, QPushButton, QSizePolicy,
                             QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QWidget)
from tqdm import tqdm

from backend.services.cache_writer import write_cache_payload
from backend.services.dataset_session import DatasetSession
from backend.services.patterns.pattern_matcher import PatternMatcher
from backend.services.progress_tracker import ProgressTracker
from backend.services.workspace_cache_coordinator import get_cache_coordinator
from backend.services.workspace_inspector import (WorkspaceDatasetInfo,
                                                  save_dataset_note,
                                                  scan_workspace)
from backend.services.workspace_manager import WorkspaceManager
from common.log_utils import log_debug, log_info, log_perf, log_warning
from config import get_config
from frontend.widgets import style


class _ScanSignals(QObject):
    finished = pyqtSignal(list)


class _ScanWorker(QRunnable):
    def __init__(self, workspace_dir: Path, workspace_manager: Optional[WorkspaceManager] = None) -> None:
        super().__init__()
        self.workspace_dir = workspace_dir
        self.workspace_manager = workspace_manager
        self.signals = _ScanSignals()

    def run(self) -> None:
        if self.workspace_manager:
            results = self.workspace_manager.scan_workspace()
        else:
            results = scan_workspace(self.workspace_dir)
        try:
            self.signals.finished.emit(results)
        except RuntimeError:
            # Dialog may be gone; ignore late emission
            pass


class _SweepSignals(QObject):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(int, int, int, int, int, int, int)


class _ResetSignals(QObject):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(int, int)


class _WorkspaceSweepWorker(QRunnable):
    def __init__(
        self,
        workspace_dir: Path,
        *,
        run_missing: bool,
        delete_marked: bool,
        restore_all: bool,
        run_duplicates: bool,
        run_quality: bool,
        run_patterns: bool,
    ) -> None:
        super().__init__()
        self.workspace_dir = workspace_dir
        self.run_missing = run_missing
        self.delete_marked = delete_marked
        self.restore_all = restore_all
        self.run_duplicates = run_duplicates
        self.run_quality = run_quality
        self.run_patterns = run_patterns
        self.signals = _SweepSignals()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        """Execute sweep across workspace using batch_operations module."""
        from backend.services.batch_operations import (
            SweepConfig, run_sweeps_on_session, BatchSweepResult
        )
        from backend.services.cache_writer import write_cache_payload

        sweep_start = time.perf_counter()
        infos = scan_workspace(self.workspace_dir)

        # Collect all leaf dataset paths (children of collections + standalone)
        dataset_paths: List[Path] = []
        for info in infos:
            if not info.is_collection:
                dataset_paths.append(info.path)

        total_datasets = len(dataset_paths)
        if total_datasets == 0:
            try:
                self.signals.finished.emit(0, 0, 0, 0, 0, 0, 0)
            except RuntimeError:
                pass
            return

        cfg = get_config()
        matcher = PatternMatcher(Path(cfg.patterns_dir), threshold=float(getattr(cfg, "pattern_match_threshold", 0.85)))
        matcher.load()

        coordinator = get_cache_coordinator()
        coordinator.set_workspace(self.workspace_dir)

        # Build sweep config from worker options
        config = SweepConfig(
            run_missing=self.run_missing,
            run_duplicates=self.run_duplicates,
            run_quality=self.run_quality,
            run_patterns=self.run_patterns,
            restore_all=self.restore_all,
            delete_marked=self.delete_marked,
            matcher=matcher if self.run_patterns else None,
        )

        # Aggregate results
        batch_result = BatchSweepResult(total_datasets=total_datasets)

        # Thread-safe progress tracking
        progress_lock = Lock()
        datasets_completed = 0
        any_payload_saved = False

        # Build dataset path to collection name mapping
        dataset_to_collection: Dict[str, str] = {}
        for info in infos:
            if info.is_collection:
                for child in info.children:
                    dataset_to_collection[str(child.path)] = info.name

        # Pool of positions for nested tqdm bars (1 to max_workers)
        max_workers = max(2, min(4, (os.cpu_count() or 2) // 2))
        position_pool: List[int] = list(range(1, max_workers + 1))
        position_lock = Lock()

        # Calculate max label width for alignment (find longest collection/dataset name)
        max_label_len = max(
            (len(f"{dataset_to_collection.get(str(p), '')}/{p.name}" if dataset_to_collection.get(str(p)) else p.name)
             for p in dataset_paths),
            default=20
        )
        max_label_len = min(max_label_len + 2, 50)  # Cap at 50 chars, add 2 for indent

        # Main progress bar at position 0 (datasets level)
        workspace_pbar = tqdm(
            total=total_datasets,
            desc="Workspace".ljust(max_label_len),
            position=0,
            leave=True,
            ncols=120,
            bar_format='{desc} {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )

        def get_position() -> Optional[int]:
            """Get a free position from pool."""
            with position_lock:
                if position_pool:
                    return position_pool.pop(0)
            return None

        def release_position(pos: int) -> None:
            """Return position to pool."""
            with position_lock:
                if pos not in position_pool:
                    position_pool.append(pos)
                    position_pool.sort()

        def process_dataset(path: Path) -> None:
            """Process a single dataset with sweep config."""
            nonlocal datasets_completed, any_payload_saved

            if self._cancelled:
                return

            # Build label with collection/dataset format - NO truncation
            collection_name = dataset_to_collection.get(str(path), "")
            label = f"{collection_name}/{path.name}" if collection_name else path.name

            # Get a free position for nested tqdm
            pos = get_position()

            dataset_pbar = None
            try:
                if self._cancelled:
                    return

                session = DatasetSession()
                if not session.load(path):
                    with progress_lock:
                        datasets_completed += 1
                    workspace_pbar.update(1)
                    try:
                        self.signals.progress.emit(datasets_completed, total_datasets, f"Skip {path.name} (load failed)")
                    except RuntimeError:
                        pass
                    return

                # Get total images for this dataset
                total_images = session.total_pairs()

                # Create nested tqdm for this dataset WITH progress bar (aligned with main bar)
                if pos is not None:
                    dataset_pbar = tqdm(
                        total=total_images,
                        desc=label.ljust(max_label_len),
                        position=pos,
                        leave=False,
                        ncols=120,
                        bar_format='{desc} {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
                    )

                # Minimal reporter that updates tqdm and checks cancellation
                class TqdmReporter:
                    def __init__(self, pbar, cancel_fn):
                        self._pbar = pbar
                        self._cancel_fn = cancel_fn

                    def advance(self, n: int = 1, suffix: str = "") -> None:
                        if self._pbar:
                            self._pbar.update(n)

                    def is_cancelled(self) -> bool:
                        return self._cancel_fn()

                reporter = TqdmReporter(dataset_pbar, lambda: self._cancelled)

                # Run sweeps with our reporter
                result = run_sweeps_on_session(
                    session,
                    config,
                    cancel_check=lambda: self._cancelled,
                    reporter=reporter,
                )

                # Ensure tqdm shows complete
                if dataset_pbar:
                    dataset_pbar.n = total_images
                    dataset_pbar.refresh()

                # Save dataset cache
                payload = session.snapshot_cache_payload()
                if payload:
                    write_cache_payload(payload)
                    coordinator.mark_dataset_dirty(path)
                    with progress_lock:
                        any_payload_saved = True

                # Aggregate results
                with progress_lock:
                    batch_result.add(result)
                    datasets_completed += 1

                workspace_pbar.update(1)

                # Report progress to UI
                result_label = f"{path.name}: " + (", ".join(result.summary_parts()) or "ok")
                try:
                    self.signals.progress.emit(datasets_completed, total_datasets, result_label)
                except RuntimeError:
                    pass

            finally:
                # Close dataset progress bar and release position
                if dataset_pbar:
                    dataset_pbar.close()
                if pos is not None:
                    release_position(pos)

        # Process all datasets in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_dataset, path): path for path in dataset_paths}
            for future in as_completed(futures):
                if self._cancelled:
                    for fut in futures:
                        fut.cancel()
                    break
                try:
                    future.result()
                except Exception as e:
                    log_debug(f"Error processing dataset: {e}", "WorkspaceDialog")


        workspace_pbar.close()

        # Ensure cursor is on a clean line after tqdm bars
        sys.stderr.flush()
        sys.stdout.flush()

        log_perf(
            f"sweep datasets={total_datasets} restored={batch_result.restored} deleted={batch_result.deleted} "
            f"dups={batch_result.duplicates} blur={batch_result.blurry} motion={batch_result.motion} "
            f"in {time.perf_counter()-sweep_start:.3f}s"
        )

        if any_payload_saved:
            try:
                coordinator.flush_workspace()
            except Exception:
                pass

        # Save workspace table to logs after sweep completes
        try:
            from backend.utils.table_writer import (format_workspace_table,
                                                    write_table_to_log)

            # Re-scan workspace to get updated stats
            updated_infos = scan_workspace(self.workspace_dir)
            table_data, headers = format_workspace_table(updated_infos)
            write_table_to_log(table_data, headers, "workspace_sweep", overwrite=True)
        except Exception as e:
            log_warning(f"Failed to save workspace table: {e}", "WorkspaceSweep")

        try:
            self.signals.finished.emit(
                total_datasets,
                batch_result.restored,
                batch_result.deleted,
                batch_result.duplicates,
                batch_result.blurry,
                batch_result.motion,
                batch_result.patterns,
            )
        except RuntimeError:
            pass


class _WorkspaceResetWorker(QRunnable):
    def __init__(self, workspace_dir: Path) -> None:
        super().__init__()
        self.workspace_dir = workspace_dir
        self.signals = _ResetSignals()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        reset_start = time.perf_counter()
        try:
            self.signals.progress.emit(0, 1, "Scanning workspace…")
        except RuntimeError:
            pass
        infos = scan_workspace(self.workspace_dir)
        leaves = [info for info in infos if not info.is_collection]
        total = len(leaves)
        if total == 0:
            try:
                self.signals.finished.emit(0, 0)
            except RuntimeError:
                pass
            return

        def _process(info: WorkspaceDatasetInfo) -> Tuple[bool, str]:
            if self._cancelled:
                return (False, info.name)
            session = DatasetSession()
            ok = False
            try:
                if session.load(info.path):
                    ok = session.reset_dataset_pristine()
            except Exception:
                ok = False
            return (ok, info.name)

        max_workers = max(2, min(6, (os.cpu_count() or 2)))
        try:
            self.signals.progress.emit(0, total, f"Reset queued: {total} dataset(s)")
        except RuntimeError:
            pass
        success = 0
        failures = 0
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_process, info): info for info in leaves}
            for future in as_completed(future_map):
                if self._cancelled:
                    for fut in future_map:
                        fut.cancel()
                    break
                try:
                    ok, name = future.result()
                except Exception:
                    ok, name = (False, future_map[future].name)
                completed += 1
                if ok:
                    success += 1
                else:
                    failures += 1
                try:
                    self.signals.progress.emit(completed, total, f"Reset {name}")
                except RuntimeError:
                    return
        log_perf(
            f"reset datasets={total} success={success} failures={failures} in {time.perf_counter()-reset_start:.3f}s"
        )
        try:
            self.signals.finished.emit(success, failures)
        except RuntimeError:
            pass


class _SweepConfirmDialog(QDialog):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Workspace sweep")
        layout = QVBoxLayout(self)
        description = QLabel(
            "Run maintenance across the entire workspace."
            "\nSelect the actions to apply to every dataset/collection."
            "\nThese operations can take time."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        self.chk_missing = QCheckBox("Re-scan missing pairs (non-destructive)")
        self.chk_missing.setChecked(False)
        self.chk_duplicates = QCheckBox("Re-run duplicate sweep")
        self.chk_quality = QCheckBox("Run blur/motion sweep")
        self.chk_patterns = QCheckBox("Run pattern sweep")
        self.chk_delete = QCheckBox("Delete all currently marked images (all reasons)")
        self.chk_restore = QCheckBox("Restore all trashed images")
        layout.addWidget(self.chk_missing)
        layout.addWidget(self.chk_duplicates)
        layout.addWidget(self.chk_quality)
        layout.addWidget(self.chk_patterns)
        layout.addWidget(self.chk_delete)
        layout.addWidget(self.chk_restore)

        self.chk_ack = QCheckBox("I understand this affects EVERY dataset in the workspace")
        layout.addWidget(self.chk_ack)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        ok_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setText("Run sweep")
            ok_button.setEnabled(False)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.chk_ack.stateChanged.connect(self._toggle_ok)

    def _toggle_ok(self) -> None:
        ok_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setEnabled(self.chk_ack.isChecked())


class WorkspacePanel(QWidget):
    refreshed = pyqtSignal(int, int)  # datasets, collections
    def __init__(
        self,
        parent: QWidget,
        workspace_dir: Path,
        on_open: Optional[Callable[[WorkspaceDatasetInfo], None]] = None,
        progress_tracker: Optional[ProgressTracker] = None,
        cancel_controller=None,
        on_cancel_state: Optional[Callable[[], None]] = None,
        *,
        autoload: bool = True,
    ) -> None:
        super().__init__(parent)
        self.workspace_dir = workspace_dir
        self._on_open = on_open
        self._progress_tracker = progress_tracker
        self._cancel_controller = cancel_controller
        self._on_cancel_state = on_cancel_state
        self.sweep_button = None
        self.refresh_button: Optional[QPushButton] = None
        self.reset_button: Optional[QPushButton] = None
        self.progress_placeholder: Optional[QWidget] = None
        self.setMinimumWidth(820)
        self._infos: List[WorkspaceDatasetInfo] = []
        self._selected_index: Optional[int] = None
        self._workers: List[_ScanWorker] = []
        self._sweep_workers: List[_WorkspaceSweepWorker] = []
        self._sweep_updater = None
        self._sweep_task_id = "workspace-sweep"
        self._sweep_cancelled = False
        self._reset_workers: List[_WorkspaceResetWorker] = []
        self._reset_updater = None
        self._reset_task_id = "workspace-reset"
        self._reset_cancelled = False
        self._closing = False
        self._has_loaded = False
        self._sweep_in_progress = False  # CRITICAL: Prevent refresh during sweeps
        self._note_timer = QTimer(self)
        self._note_timer.setSingleShot(True)
        self._note_timer.setInterval(800)
        self._note_timer.timeout.connect(self._save_note)

        # Create workspace manager for efficient stats handling (must be in main thread)
        self.workspace_manager = WorkspaceManager(workspace_dir, on_dataset_changed=self._on_dataset_changed)

        self._build_ui()
        self.thread_pool = QThreadPool.globalInstance()
        if autoload:
            self._refresh()

    def _on_dataset_changed(self, dataset_path: Path) -> None:
        """Called when a dataset in this workspace changes - refresh the table."""

        # CRITICAL: Don't refresh during sweep to avoid refresh loop
        if self._sweep_in_progress:
            log_debug(f"Dataset {dataset_path.name} changed but skipping refresh (sweep in progress)", "WorkspaceDialog")
            return

        log_info(f"Dataset {dataset_path.name} changed, refreshing table...", "WorkspaceDialog")
        # Debounce refreshes to avoid excessive updates
        if not hasattr(self, '_refresh_timer'):
            self._refresh_timer = QTimer(self)
            self._refresh_timer.setSingleShot(True)
            self._refresh_timer.setInterval(1000)  # 1 second debounce
            self._refresh_timer.timeout.connect(lambda: self._refresh() if not self._closing else None)
        self._refresh_timer.start()

    def closeEvent(self, event) -> None:
        # Stop queued workers to avoid callbacks after dialog destruction
        self.cancel_all()
        # Flush all handlers before closing
        if hasattr(self, 'workspace_manager'):
            self.workspace_manager.flush_all()
        super().closeEvent(event)

    def set_workspace_dir(self, workspace_dir: Path, *, force_refresh: bool = False) -> None:
        same = self.workspace_dir == workspace_dir
        self.workspace_dir = workspace_dir
        self.path_label.setText(f"Workspace: {self.workspace_dir}")

        # Update workspace manager
        if hasattr(self, 'workspace_manager'):
            self.workspace_manager.workspace_path = workspace_dir
        else:
            self.workspace_manager = WorkspaceManager(workspace_dir)

        if same and not force_refresh:
            return
        self._has_loaded = False
        self._refresh()

    def ensure_loaded(self) -> None:
        if self._has_loaded or self._workers:
            return
        self._refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.path_label = QLabel(f"Workspace: {self.workspace_dir}")
        font = self.path_label.font()
        font.setBold(True)
        self.path_label.setFont(font)
        header_row = QHBoxLayout()
        header_row.setSpacing(8)
        header_row.addWidget(self.path_label)
        header_row.addStretch(1)
        self.open_button = QPushButton("Open selected", self)
        self.open_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.open_button.clicked.connect(self._open_selected)
        self.open_button.setStyleSheet(style.scoped_button_style("open_button"))
        # Placeholder for progress panel mounted by parent if needed
        self.progress_placeholder = QWidget(self)
        self.progress_placeholder.setObjectName("workspace_progress_placeholder")
        self.progress_placeholder.setMinimumWidth(220)
        self.progress_placeholder.setMaximumHeight(24)
        self.progress_placeholder.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.progress_placeholder.setVisible(False)
        header_row.addWidget(self.progress_placeholder)
        header_row.addWidget(self.open_button)
        layout.addLayout(header_row)

        self.table = QTableWidget(0, 8, self)  # 8 columns now (added Patterns)
        self.table.setHorizontalHeaderLabels([
            "Dataset",
            "Pairs",
            "Deleted",
            "Delete Reasons",
            "Tagged to delete",
            "Calibration",
            "Patterns",  # NEW COLUMN
            "Sweeps",
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._handle_selection_changed)
        copy_shortcut = QShortcut(QKeySequence.StandardKey.Copy, self.table)
        copy_shortcut.activated.connect(self._copy_selected_row)
        header = self.table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            header.setStretchLastSection(True)
        vert_header = self.table.verticalHeader()
        if vert_header is not None:
            vert_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table, 6)

        buttons = QHBoxLayout()
        layout.addLayout(buttons)

        note_label = QLabel("Dataset Notes:")
        note_label.setStyleSheet(style.heading_style())
        layout.addWidget(note_label)
        self.note_edit = QPlainTextEdit(self)
        self.note_edit.setMaximumHeight(120)
        self.note_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.note_edit.textChanged.connect(self._handle_note_changed)
        layout.addWidget(self.note_edit, 2)
        self.note_status = QLabel("")
        self.note_status.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.note_status.setStyleSheet("color: #4b5563; font-size: 12px;")
        layout.addWidget(self.note_status)

    def _refresh(self) -> None:
        log_info("Refresh requested, scheduling scan...", "WORKSPACE")

        self._has_loaded = True
        if self.refresh_button:
            self.refresh_button.setEnabled(False)
        self.table.setRowCount(1)
        self.table.setItem(0, 0, QTableWidgetItem("Scanning workspace..."))
        for col in range(1, 4):
            self.table.setItem(0, col, QTableWidgetItem(""))
        self.note_edit.clear()
        self._note_status("Workspace saved")
        self.note_edit.setEnabled(False)
        if self.progress_placeholder:
            self.progress_placeholder.setVisible(True)

        # CRITICAL: scan_workspace creates QObject handlers which MUST be in main thread
        # Use QTimer to ensure we're in main thread
        QTimer.singleShot(0, self._do_scan)

    def _do_scan(self) -> None:
        """Execute scan in main thread (called via QTimer)."""
        log_info("Executing workspace scan...", "WORKSPACE")

        # CRITICAL: Always scan fresh from handlers, NO memory cache
        from backend.services.workspace_inspector import \
            invalidate_workspace_cache
        try:
            invalidate_workspace_cache(self.workspace_dir)
        except Exception:
            pass

        infos = self.workspace_manager.scan_workspace()

        log_info(f"Scan complete, applying {len(infos)} results to table...", "WORKSPACE")
        self._apply_scan_results(infos)
        datasets = len([i for i in self._infos if not i.is_collection])
        collections = len([i for i in self._infos if i.is_collection])
        self.refreshed.emit(datasets, collections)
        if self.reset_button:
            self.reset_button.setEnabled(bool(self._infos))
        if self.progress_placeholder:
            self.progress_placeholder.setVisible(bool(self._reset_workers))

    def _open_sweep_dialog(self) -> None:
        dialog = _SweepConfirmDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        run_missing = dialog.chk_missing.isChecked()
        run_duplicates = dialog.chk_duplicates.isChecked()
        run_quality = dialog.chk_quality.isChecked()
        run_patterns = dialog.chk_patterns.isChecked()
        delete_marked = dialog.chk_delete.isChecked()
        restore_all = dialog.chk_restore.isChecked()
        if not any([run_missing, run_duplicates, run_quality, run_patterns, delete_marked, restore_all]):
            return

        # Mark sweep as in progress
        self._sweep_in_progress = True
        self._start_sweep(
            run_missing=run_missing,
            run_duplicates=run_duplicates,
            run_quality=run_quality,
            run_patterns=run_patterns,
            delete_marked=delete_marked,
            restore_all=restore_all,
        )

    def _start_sweep(
        self,
        *,
        run_missing: bool,
        run_duplicates: bool,
        run_quality: bool,
        run_patterns: bool,
        delete_marked: bool,
        restore_all: bool,
    ) -> None:
        # Cancel any existing sweep before starting a new one
        self._cancel_sweep()
        worker = _WorkspaceSweepWorker(
            self.workspace_dir,
            run_missing=run_missing,
            delete_marked=delete_marked,
            restore_all=restore_all,
            run_duplicates=run_duplicates,
            run_quality=run_quality,
            run_patterns=run_patterns,
        )
        worker.signals.progress.connect(self._handle_sweep_progress)
        worker.signals.finished.connect(self._handle_sweep_finished)
        self._sweep_workers.append(worker)
        self._sweep_cancelled = False
        if self.refresh_button:
            self.refresh_button.setEnabled(False)
        if self._progress_tracker:
            total = max(1, len([i for i in self._infos if not i.is_collection])) if self._infos else 1
            self._sweep_updater = self._progress_tracker.start_task(
                self._sweep_task_id, "Workspace sweep", total
            )
        if self._cancel_controller:
            self._cancel_controller.register(self._sweep_task_id, self._cancel_sweep)
        if self._on_cancel_state:
            self._on_cancel_state()
        if self.thread_pool is not None:
            self.thread_pool.start(worker)

    def _start_reset(self) -> None:
        self._cancel_reset()
        leaves = [info for info in self._infos if not info.is_collection]
        worker = _WorkspaceResetWorker(self.workspace_dir)
        worker.signals.progress.connect(self._handle_reset_progress)
        worker.signals.finished.connect(self._handle_reset_finished)
        self._reset_workers.append(worker)
        self._reset_cancelled = False
        if self.refresh_button:
            self.refresh_button.setEnabled(False)
        if self._progress_tracker:
            total = max(1, len(leaves)) if leaves else 1
            self._reset_updater = self._progress_tracker.start_task(
                self._reset_task_id, "Workspace reset", total
            )
        if self.progress_placeholder:
            self.progress_placeholder.setVisible(True)
        if self._cancel_controller:
            self._cancel_controller.register(self._reset_task_id, self._cancel_reset)
        if self._on_cancel_state:
            self._on_cancel_state()
        if self.thread_pool is not None:
            self.thread_pool.start(worker)

    def _handle_sweep_progress(self, idx: int, total: int, label: str) -> None:
        if self._sweep_updater:
            self._sweep_updater(idx, total)
        self.path_label.setText(f"Workspace: {self.workspace_dir} — {label}")

    def _handle_sweep_finished(
        self,
        total: int,
        restored: int,
        deleted: int,
        duplicates: int,
        blurry: int,
        motion: int,
        patterns: int,
    ) -> None:
        from common.timing import get_timestamp

        # Mark sweep as finished
        self._sweep_in_progress = False

        log_info(f"Sweep finished: total={total}, restored={restored}, "
              f"deleted={deleted}, duplicates={duplicates}, blurry={blurry}, motion={motion}, patterns={patterns}", "WORKSPACE")

        self._cleanup_sweep_state()
        self.path_label.setText(f"Workspace: {self.workspace_dir}")
        if self._sweep_cancelled:
            self._sweep_cancelled = False
            log_info("Sweep was cancelled, refreshing table...", "WORKSPACE")
            self._refresh()
            return
        summary_lines = [
            f"Processed {total} dataset(s).",
            f"Restored: {restored}",
            f"Deleted: {deleted}",
            f"Duplicates: {duplicates}",
        ]
        if blurry or motion:
            summary_lines.append(f"Blurry: {blurry}")
            summary_lines.append(f"Motion: {motion}")
        if patterns:
            summary_lines.append(f"Patterns: {patterns}")
        QMessageBox.information(
            self,
            "Workspace sweep",
            "\n".join(summary_lines),
        )
        log_debug("Refreshing table after sweep...", "WORKSPACE")
        self._refresh()

    def _handle_reset_progress(self, idx: int, total: int, label: str) -> None:
        if self._reset_updater:
            self._reset_updater(idx, total)
        self.path_label.setText(f"Workspace: {self.workspace_dir} — {label}")

    def _handle_reset_finished(self, success: int, failures: int) -> None:
        self._cleanup_reset_state()
        self.path_label.setText(f"Workspace: {self.workspace_dir}")
        if self._reset_cancelled:
            self._reset_cancelled = False
            self._refresh()
            return
        QMessageBox.information(
            self,
            "Reset workspace",
            f"Reset finished. Success: {success}; Failed: {failures}.",
        )
        self._refresh()

    def _cleanup_sweep_state(self) -> None:
        if self._sweep_updater and self._progress_tracker:
            self._progress_tracker.finish(self._sweep_task_id)
        self._sweep_updater = None
        if self._cancel_controller:
            self._cancel_controller.unregister(self._sweep_task_id)
        if self._on_cancel_state:
            self._on_cancel_state()
        if self._sweep_workers:
            self._sweep_workers.clear()
        if self.refresh_button:
            self.refresh_button.setEnabled(True)
        if self.progress_placeholder:
            self.progress_placeholder.setVisible(bool(self._sweep_workers))

    def _cancel_sweep(self) -> None:
        if not self._sweep_workers:
            return
        self._sweep_cancelled = True
        for worker in list(self._sweep_workers):
            try:
                worker.cancel()
            except Exception:
                pass
        self._cleanup_sweep_state()
        self._note_status("Sweep cancelled")

    def _cleanup_reset_state(self) -> None:
        if self._reset_updater and self._progress_tracker:
            self._progress_tracker.finish(self._reset_task_id)
        self._reset_updater = None
        if self._cancel_controller:
            self._cancel_controller.unregister(self._reset_task_id)
        if self._on_cancel_state:
            self._on_cancel_state()
        if self._reset_workers:
            self._reset_workers.clear()
        if self.refresh_button:
            self.refresh_button.setEnabled(True)
        if self.progress_placeholder:
            self.progress_placeholder.setVisible(False)

    def _cancel_reset(self) -> None:
        if not self._reset_workers:
            return
        self._reset_cancelled = True
        for worker in list(self._reset_workers):
            try:
                worker.cancel()
            except Exception:
                pass
        self._cleanup_reset_state()
        self._note_status("Workspace reset cancelled")
        self.path_label.setText(f"Workspace: {self.workspace_dir}")

    def cancel_all(self) -> None:
        self._closing = True
        self._cancel_sweep()
        self._cancel_reset()
        for worker in list(self._workers):
            try:
                worker.signals.finished.disconnect()
            except Exception:
                pass
        self._workers.clear()
        try:
            if self.thread_pool is not None:
                self.thread_pool.clear()
        except Exception:
            pass

    def _format_reasons(self, info: WorkspaceDatasetInfo) -> str:
        """Format breakdown of items already deleted (delegate to stats)."""
        return info.stats.format_removed_reasons(compact=True)

    def _format_removed(self, info: WorkspaceDatasetInfo) -> str:
        """Format removed/deleted count (delegate to stats)."""
        return info.stats.format_removed_count(compact=True)

    def _format_calibration(self, info: WorkspaceDatasetInfo) -> str:
        """Format calibration statistics (delegate to stats)."""
        return info.stats.format_calibration(compact=True)

    def _format_patterns(self, info: WorkspaceDatasetInfo) -> str:
        """Format pattern matches: pattern_name (count), ..."""
        if not info.stats.pattern_matches:
            return ""
        # Sort by count descending, then by name
        sorted_patterns = sorted(
            info.stats.pattern_matches.items(),
            key=lambda x: (-x[1], x[0])
        )
        return ", ".join(f"{name} ({count})" for name, count in sorted_patterns)

    def _format_sweeps(self, info: WorkspaceDatasetInfo) -> str:
        """Format sweep status with better readability."""
        result = ""
        parts = []
        if info.stats.sweep_duplicates_done:
            parts.append("Duplicate check ✓")
        if info.stats.sweep_quality_done:
            parts.append("Quality check ✓")
        if info.stats.sweep_patterns_done:
            parts.append("Pattern check ✓")

        # Show which ones are NOT done
        not_done = []
        if not info.stats.sweep_duplicates_done:
            not_done.append("Duplicate check ")
        if not info.stats.sweep_quality_done:
            not_done.append("Quality check ")
        if not info.stats.sweep_patterns_done:
            not_done.append("Pattern check ")

        # Format: "D✓ Q✓" if some done, or "—" if none done
        if parts:
            result = " ".join(parts)

        if not_done:
            result = result + " (pending: " + ", ".join(not_done) + ")"
        return result

    def _format_outliers(self, info: WorkspaceDatasetInfo) -> str:
        """Format outlier statistics (delegate to stats)."""
        return info.stats.format_outliers(compact=True)

    def _format_tagged(self, info: WorkspaceDatasetInfo) -> str:
        """Format tagged (marked for deletion) summary (delegate to stats)."""
        return info.stats.format_tagged_summary(compact=True, multiline=True)

    def _copy_selected_row(self) -> None:
        sel_model = self.table.selectionModel()
        if sel_model is None:
            return
        selected = sel_model.selectedRows()
        if not selected:
            return
        row = selected[0].row()
        parts = []
        for col in range(self.table.columnCount()):
            item = self.table.item(row, col)
            parts.append(item.text() if item else "")
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText("\t".join(parts))

    def _handle_selection_changed(self) -> None:
        sel_model = self.table.selectionModel()
        if sel_model is None:
            return
        selected = sel_model.selectedRows()
        if not selected:
            self._selected_index = None
            self.note_edit.clear()
            return
        table_row = selected[0].row()
        # Account for "Workspace total" row at index 0
        # Table rows: [0=Workspace total, 1=_infos[0], 2=_infos[1], ...]
        if table_row == 0:
            # Workspace total row selected - can't open
            self._selected_index = None
            self.note_edit.clear()
            self.open_button.setEnabled(False)
            self.note_edit.setEnabled(False)
            return
        self._selected_index = table_row - 1
        log_debug(f"_handle_selection_changed: table_row={table_row}, _selected_index={self._selected_index}", "WORKSPACE")
        if self._selected_index < 0 or self._selected_index >= len(self._infos):
            return
        info = self._infos[self._selected_index]
        log_debug(f"Selected info: name={info.name}, is_collection={info.is_collection}, parent={info.parent}", "WORKSPACE")
        self.note_edit.blockSignals(True)
        self.note_edit.setPlainText(info.note or "")
        self.note_edit.blockSignals(False)
        self.open_button.setEnabled(bool(self._on_open and self._infos))
        self.note_edit.setEnabled(True)

    def _handle_note_changed(self) -> None:
        if self._selected_index is None:
            return
        current_note = self._infos[self._selected_index].note or ""
        if self.note_edit.toPlainText() != current_note:
            self._note_status("Saving workspace…")
            self._note_timer.start()
        else:
            self._note_status("Workspace saved")

    def _save_note(self) -> None:
        if self._selected_index is None:
            return
        if self._note_timer.isActive():
            self._note_timer.stop()
        info = self._infos[self._selected_index]
        note = self.note_edit.toPlainText()
        save_dataset_note(info.path, note)
        self._infos[self._selected_index] = WorkspaceDatasetInfo(
            name=info.name,
            path=info.path,
            note=note,
            is_collection=info.is_collection,
            children=info.children,
            parent=info.parent,
            stats=info.stats,
        )
        self._note_status("Workspace saved")

    def _note_status(self, text: str) -> None:
        if hasattr(self, "note_status"):
            self.note_status.setText(text)

    def _reset_selected_dataset(self) -> None:
        if self._selected_index is None or self._selected_index >= len(self._infos):
            QMessageBox.information(self, "Reset dataset", "Select a dataset first.")
            return
        info = self._infos[self._selected_index]
        if info.is_collection:
            QMessageBox.information(self, "Reset dataset", "Reset is only available for individual datasets.")
            return
        dataset_name = info.name
        msg = QMessageBox(self)
        msg.setWindowTitle("Reset dataset (dangerous)")
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setText(
            "This will restore trashed images, clear labels/marks/calibration, and wipe cached state.\n"
            "Re-run sweeps after reset. Continue?"
        )
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg.setDefaultButton(QMessageBox.StandardButton.No)
        if msg.exec() != QMessageBox.StandardButton.Yes:
            return
        text, ok = QInputDialog.getText(
            self,
            "Confirm reset",
            f"Type the dataset name to confirm reset:\n{dataset_name}",
        )
        if not ok or text.strip() != dataset_name:
            QMessageBox.information(self, "Reset dataset", "Confirmation text did not match. Reset cancelled.")
            return
        session = DatasetSession()
        if not session.load(info.path):
            QMessageBox.warning(self, "Reset dataset", "Could not load dataset.")
            return
        if not session.reset_dataset_pristine():
            QMessageBox.warning(self, "Reset dataset", "Reset failed.")
            return
        QMessageBox.information(self, "Reset dataset", "Dataset reset to pristine state. Re-run sweeps as needed.")
        self._refresh()

    def reset_selected(self) -> None:
        """Expose reset for external callers (workspace menu)."""
        self._reset_selected_dataset()

    def refresh_workspace(self) -> None:
        self._refresh()

    def reset_workspace(self) -> None:
        leaves = [info for info in self._infos if not info.is_collection]
        if not leaves:
            QMessageBox.information(self, "Reset workspace", "No datasets loaded from this workspace.")
            return
        msg = QMessageBox(self)
        msg.setWindowTitle("Reset entire workspace")
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setText(
            "This will reset every dataset: restore trashed images, clear labels/marks/calibration, and wipe cached state."
        )
        msg.setInformativeText("This cannot be undone. Continue?")
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg.setDefaultButton(QMessageBox.StandardButton.No)
        if msg.exec() != QMessageBox.StandardButton.Yes:
            return
        text, ok = QInputDialog.getText(
            self,
            "Confirm workspace reset",
            "Type RESET to confirm resetting ALL datasets:",
        )
        if not ok or text.strip().upper() != "RESET":
            QMessageBox.information(self, "Reset workspace", "Confirmation text did not match. Reset cancelled.")
            return
        self._start_reset()

    def _apply_scan_results(self, infos: List[WorkspaceDatasetInfo]) -> None:
        if self._closing:
            return
        self._infos = infos or []

        debug = os.environ.get("DEBUG_WORKSPACE", "").lower() in {"1", "true", "on"}

        if debug:
            log_debug("\\n===== _apply_scan_results START =====", "WORKSPACE")
            log_debug(f"Received {len(self._infos)} infos:", "WORKSPACE")
            for info in self._infos:
                log_debug(f"  - {info.name}: is_collection={info.is_collection}, parent={info.parent}", "WORKSPACE")

        # Sort: single datasets (no children) first, then collections (with children)
        single_datasets = [info for info in self._infos if not info.is_collection and not info.parent]

        if debug:
            log_debug(f"Single datasets (no parent, not collection): {len(single_datasets)}", "WORKSPACE")
            for info in single_datasets:
                log_debug(f"  - {info.name}", "WORKSPACE")

        # Build collections with children
        collections_with_children = []
        processed_collections = set()

        for info in self._infos:
            if info.is_collection and info.name not in processed_collections:
                # Add collection
                collections_with_children.append(info)
                processed_collections.add(info.name)

                # Add its children
                children = [child for child in self._infos if child.parent == info.name]
                if debug:
                    log_debug(f"Collection '{info.name}' has {len(children)} children: {[child.name for child in children]}", "WORKSPACE")

                collections_with_children.extend(children)

        # Final order: workspace total, single datasets, collections+children
        workspace_total = self._workspace_totals() if self._infos else None
        sorted_infos = single_datasets + collections_with_children
        rows = ([workspace_total] if workspace_total else []) + sorted_infos

        # Build row_to_info_index mapping for _open_selected
        self._row_to_info_index = {}
        for row_idx, row_data in enumerate(rows):
            if row_data is workspace_total:
                # Workspace total has no info
                self._row_to_info_index[row_idx] = None
            else:
                # Find actual index in self._infos
                try:
                    info_idx = self._infos.index(row_data)
                    self._row_to_info_index[row_idx] = info_idx
                except ValueError:
                    self._row_to_info_index[row_idx] = None

        if debug:
            log_debug(f"Final table: {len(rows)} rows", "WORKSPACE")
            log_debug(f"  - Workspace total: {'Yes' if workspace_total else 'No'}", "WORKSPACE")
            log_debug(f"  - Single datasets: {len(single_datasets)}", "WORKSPACE")
            log_debug(f"  - Collections+children: {len(collections_with_children)}", "WORKSPACE")
            log_debug(f"Row to info index mapping: {self._row_to_info_index}", "WORKSPACE")
            log_debug("===== _apply_scan_results END =====\n", "WORKSPACE")

        self.table.setRowCount(len(rows))
        for row, info in enumerate(rows):
            # DEBUG: Log stats for each row
            if os.environ.get("DEBUG_WORKSPACE", "").lower() in {"1", "true", "on"}:
                log_debug(f"Row {row}: {info.name} - pairs={info.stats.total_pairs}, "
                      f"manual={info.stats.tagged_manual}, auto={info.stats.tagged_auto}, "
                      f"calib_marked={info.stats.calibration_marked}, calib_both={info.stats.calibration_both}", "WORKSPACE")

            # Determine display name
            if info.name == "Workspace total":
                display_name = info.name
            elif info.is_collection:
                display_name = info.name
            elif info.parent and not info.is_collection:
                display_name = f"  ↳ {info.name}"
            else:
                display_name = info.name
            # Indent all columns for child rows using a shared prefix
            child_prefix = "    " if (info.parent and not info.is_collection) else ""
            name_item = QTableWidgetItem(display_name)
            if info.name == "Workspace total" or info.is_collection:
                font = name_item.font()
                font.setBold(True)
                name_item.setFont(font)
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, QTableWidgetItem(child_prefix + str(info.stats.total_pairs)))
            self.table.setItem(row, 2, QTableWidgetItem(child_prefix + self._format_removed(info)))
            self.table.setItem(row, 3, QTableWidgetItem(child_prefix + self._format_reasons(info)))
            self.table.setItem(row, 4, QTableWidgetItem(child_prefix + self._format_tagged(info)))
            self.table.setItem(row, 5, QTableWidgetItem(child_prefix + self._format_calibration(info)))
            self.table.setItem(row, 6, QTableWidgetItem(child_prefix + self._format_patterns(info)))
            self.table.setItem(row, 7, QTableWidgetItem(child_prefix + self._format_sweeps(info)))
            for col in range(8):
                item = self.table.item(row, col)
                if item:
                    if info.is_collection or info.name == "Workspace total":
                        font = item.font()
                        font.setBold(True)
                        item.setFont(font)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    item.setToolTip(str(info.path))
        if self._infos:
            self.table.blockSignals(True)
            self.table.selectRow(0)
            self.table.blockSignals(False)
        else:
            self.note_edit.clear()
            self._selected_index = None
        self.note_edit.setEnabled(bool(self._infos))
        if self.refresh_button:
            self.refresh_button.setEnabled(True)
        if self.sweep_button:
            self.sweep_button.setEnabled(True)
        self.open_button.setEnabled(bool(self._on_open and self._infos))
        if self.reset_button:
            self.reset_button.setEnabled(bool(self._infos))

    def _open_selected(self) -> None:
        if not self._on_open or self._selected_index is None:
            return

        # Get selected row from table
        selected_row = self.table.currentRow()
        if selected_row < 0:
            log_debug("No row selected in table", "WORKSPACE")
            return

        # Use mapping from row to info index
        if not hasattr(self, '_row_to_info_index'):
            log_debug("No row_to_info_index mapping available", "WORKSPACE")
            return

        actual_index = self._row_to_info_index.get(selected_row)

        if actual_index is None:
            log_debug(f"Selected row {selected_row} has no corresponding info (workspace total?)", "WORKSPACE")
            return

        if actual_index < 0 or actual_index >= len(self._infos):
            log_debug(f"Invalid index: actual_index={actual_index}, infos_len={len(self._infos)}", "WORKSPACE")
            return

        info = self._infos[actual_index]
        log_debug(f"Opening: name={info.name}, path={info.path}, is_collection={info.is_collection}", "WORKSPACE")
        self._on_open(info)

    def _workspace_totals(self) -> WorkspaceDatasetInfo:
        """Aggregate all dataset stats into workspace total using DatasetStats.merge()."""
        leaves = [info for info in self._infos if not info.is_collection]

        from backend.services.stats_manager import DatasetStats
        total_stats = DatasetStats()
        for info in leaves:
            total_stats.merge(info.stats)

        return WorkspaceDatasetInfo(
            name="Workspace total",
            path=self.workspace_dir,
            note="",
            stats=total_stats,
        )