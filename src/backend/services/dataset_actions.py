"""High-level dataset operations wired to menu actions and context menus.

Handles import/export of calibration data, delete/restore flows, and dataset maintenance actions.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from common.yaml_utils import load_yaml, save_yaml
from PyQt6.QtCore import Qt, QRunnable, QThreadPool, pyqtSignal, QObject
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QInputDialog

from backend.dataset_loader import DatasetLoader
from backend.services.cache_service import is_distortion_vector, is_matrix3x3, is_vector3
from frontend.utils.ui_messages import (
    calibration_removed_message,
    format_move_confirmation,
    format_move_failure,
    format_move_success,
    no_restore_items_message,
    restore_prompt_message,
    restored_pairs_message,
)
from common.reasons import (
    REASON_BLURRY,
    REASON_DUPLICATE,
    REASON_MOTION,
    REASON_SYNC,
)

if TYPE_CHECKING:  # pragma: no cover - import guard
    from image_viewer import ImageViewer


class _DeleteByReasonSignals(QObject):
    finished = pyqtSignal(str, list, list)
    progress = pyqtSignal(int, int)


class _DeleteByReasonWorker(QRunnable):
    def __init__(self, loader: Any, bases: List[str], reason: str, auto: bool = False) -> None:
        super().__init__()
        self.loader = loader
        self.bases = bases
        self.reason = reason
        self.auto = auto
        self.signals = _DeleteByReasonSignals()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:  # pragma: no cover - thread worker
        moved: List[str] = []
        failed: List[str] = []
        total = len(self.bases)
        for idx, base in enumerate(self.bases, start=1):
            if self._cancelled:
                break
            try:
                if self.loader.delete_entry(base, self.reason, auto=self.auto):
                    moved.append(base)
                else:
                    failed.append(base)
            except Exception:  # noqa: BLE001
                failed.append(base)
            self.signals.progress.emit(idx, total)
        self.signals.finished.emit(self.reason, moved, failed)


class DatasetActions:
    def __init__(self, viewer: "ImageViewer", default_dataset_dir: str) -> None:
        self.viewer = viewer
        self.default_dataset_dir = default_dataset_dir
        self.thread_pool: QThreadPool = getattr(viewer, "thread_pool", QThreadPool.globalInstance())
        self._delete_job_active = False
        self._active_delete_worker: Optional[_DeleteByReasonWorker] = None
        self._active_delete_loader: Any = None
        self._ignore_delete_results = False
        self._teardown_cursor_reset = getattr(viewer, "unsetCursor", lambda: None)
        self._delete_task_id = "delete-by-reason"

    # ------------------------------------------------------------------
    # Public entry points wired from ImageViewer
    # ------------------------------------------------------------------
    def import_calibration_data(self) -> None:
        viewer = self.viewer
        session = viewer.session
        if not session.loader:
            QMessageBox.information(viewer, "Calibration data", "Load a dataset first.")
            return
        file_path, _ = QFileDialog.getOpenFileName(
            viewer,
            "Select calibration YAML",
            str(Path.home()),
            "YAML files (*.yml *.yaml)",
        )
        if not file_path:
            return
        payload = load_yaml(file_path)
        if payload == {}:
            QMessageBox.warning(viewer, "Calibration data", f"Could not read file, empty payload: {file_path}")
            return
        matrices, extrinsic = self._parse_calibration_payload(payload)
        if not any(matrices.values()):
            QMessageBox.warning(viewer, "Calibration data", "No valid matrices found in file.")
            return
        for key, data in matrices.items():
            if data:
                viewer.state.cache_data["_matrices"][key] = data
        if extrinsic:
            viewer.state.calibration_extrinsic = extrinsic
        viewer.invalidate_overlay_cache()
        viewer.statusBar().showMessage("Calibration parameters imported", 4000)
        viewer._mark_cache_dirty()
        if viewer.view_rectified and session.has_images():
            viewer.load_current()

    def clear_empty_datasets(self) -> None:
        viewer = self.viewer
        root_dir = QFileDialog.getExistingDirectory(
            viewer,
            "Select datasets root",
            self.default_dataset_dir,
        )
        if not root_dir:
            return
        root_path = Path(root_dir)
        if not root_path.exists():
            QMessageBox.warning(viewer, "Cleanup", "Selected directory does not exist anymore.")
            return
        empty_dirs = self._find_empty_dataset_dirs(root_path)
        if not empty_dirs:
            QMessageBox.information(viewer, "Cleanup", "No empty dataset folders found.")
            return

        def _relative_label(path: Path) -> str:
            try:
                return str(path.relative_to(root_path))
            except ValueError:
                return str(path)

        preview_lines = [f"- {_relative_label(path)}" for path in empty_dirs[:10]]
        if len(empty_dirs) > 10:
            preview_lines.append(f"... and {len(empty_dirs) - 10} more")
        message = "\n".join(
            [
                f"Found {len(empty_dirs)} dataset folder(s) without images:",
                *preview_lines,
                "",
                "Delete these folders permanently?",
            ]
        )
        reply = QMessageBox.question(
            viewer,
            "Cleanup",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        removed: List[Path] = []
        failures: List[str] = []
        for dataset_dir in empty_dirs:
            try:
                shutil.rmtree(dataset_dir)
                removed.append(dataset_dir)
                if viewer.session.dataset_path and dataset_dir == viewer.session.dataset_path:
                    viewer.session.loader = None
                    viewer.session.dataset_path = None
                    viewer._reset_runtime_state()
                    viewer._show_empty_state()
            except OSError as exc:
                failures.append(f"{dataset_dir.name}: {exc}")
        summary = f"Removed {len(removed)} empty dataset folder(s)."
        if failures:
            summary += f" Failed to remove {len(failures)} folder(s)."
            QMessageBox.warning(viewer, "Cleanup", summary + "\n" + "\n".join(failures[:5]))
        else:
            QMessageBox.information(viewer, "Cleanup", summary)
        viewer.statusBar().showMessage(summary, 6000)

    def reset_dataset(self) -> None:
        viewer = self.viewer
        session = viewer.session
        if not session.loader or not session.dataset_path:
            QMessageBox.information(viewer, "Reset dataset", "Load a dataset first.")
            return
        dataset_path = Path(session.dataset_path)
        dataset_name = dataset_path.name
        reply = QMessageBox.question(
            viewer,
            "Reset dataset (dangerous)",
            "This will restore trashed images, clear all labels/marks/calibration, and wipe cached state.\n"
            "Re-run sweeps after reset. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        text, ok = QInputDialog.getText(
            viewer,
            "Confirm reset",
            f"Type the dataset name to confirm reset:\n{dataset_name}",
        )
        if not ok or text.strip() != dataset_name:
            QMessageBox.information(viewer, "Reset dataset", "Confirmation text did not match. Reset cancelled.")
            return
        if not session.reset_dataset_pristine():
            QMessageBox.warning(viewer, "Reset dataset", "Could not reset dataset.")
            return
        viewer._load_dataset(dataset_path)
        viewer.statusBar().showMessage("Dataset reset to pristine state.", 6000)

    def delete_marked_images(self) -> None:
        viewer = self.viewer
        session = viewer.session
        state = viewer.state
        if not session.loader or not state.cache_data["marks"]:
            return
        count = len(state.cache_data["marks"])
        reply = QMessageBox.question(
            viewer,
            "Move to to_delete",
            format_move_confirmation(count),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        task_id = "delete-selected"
        total = len(state.cache_data["marks"])
        updater = viewer.progress_tracker.start_task(task_id, f"Deleting selected images ({total})", total)

        outcome = session.delete_marked_entries(progress_cb=lambda done, overall: updater(done, overall))
        viewer.progress_tracker.finish(task_id)
        viewer.invalidate_overlay_cache()
        viewer._bump_signature_epoch()
        viewer._update_delete_button()
        if outcome.dataset_available and session.loader:
            total = session.total_pairs()
            if total:
                viewer.current_index = min(viewer.current_index, total - 1)
                viewer.current_index = max(viewer.current_index, 0)
                viewer.load_current()
            else:
                viewer.current_index = 0
                viewer._show_empty_state()
        else:
            viewer.current_index = 0
            viewer._show_empty_state()
        if outcome.failed:
            QMessageBox.warning(
                viewer,
                "Move failed",
                format_move_failure(outcome.failed),
            )
        elif outcome.moved:
            destination = session.loader.to_delete_dir if session.loader else "to_delete"
            QMessageBox.information(
                viewer,
                "Moved successfully",
                format_move_success(outcome.moved, str(destination)),
            )
        viewer._update_restore_menu()
        viewer._mark_cache_dirty()

    def delete_by_reason(self, reason: str) -> None:
        viewer = self.viewer
        session = viewer.session
        state = viewer.state
        if self._delete_job_active:
            viewer.statusBar().showMessage("Deletion already in progress", 3000)
            return
        if not session.loader:
            return
        marks = state.cache_data["marks"]
        targets = []
        for base, entry in marks.items():
            if isinstance(entry, dict):
                if entry.get("reason") == reason:
                    targets.append(base)
            else:
                # Legacy format
                if entry == reason:
                    targets.append(base)
        if not targets:
            QMessageBox.information(viewer, "Delete", "No images marked with that reason.")
            return
        reply = QMessageBox.question(
            viewer,
            "Move to to_delete",
            format_move_confirmation(len(targets)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        total = len(targets)
        # Check if any of these marks are auto (unified format)
        is_auto = any(
            isinstance(marks.get(base), dict) and marks[base].get("auto", False)
            for base in targets
        )
        self._ignore_delete_results = False
        self._delete_job_active = True
        viewer.setCursor(Qt.CursorShape.BusyCursor)
        viewer.statusBar().showMessage(f"Deleting {total} marked image(s)...", 0)
        updater = viewer.progress_tracker.start_task(self._delete_task_id, f"Deleting {total} images ({reason})", total)
        worker = _DeleteByReasonWorker(session.loader, targets, reason, auto=is_auto)
        worker.signals.finished.connect(self._handle_delete_by_reason_finished)
        worker.signals.progress.connect(lambda done, overall: updater(done, overall))
        self._active_delete_worker = worker
        self._active_delete_loader = session.loader
        self.thread_pool.start(worker)

    def _handle_delete_by_reason_finished(self, reason: str, moved: List[str], failed: List[str]) -> None:
        viewer = self.viewer
        session = viewer.session
        state = viewer.state
        if getattr(viewer, "_is_closing", False) or self._ignore_delete_results:
            self._teardown_delete_job()
            return
        loader_ref = self._active_delete_loader
        self._teardown_delete_job()
        if not session.loader or loader_ref is not session.loader:
            viewer.statusBar().showMessage("Dataset changed before deletion completed", 6000)
            return
        calib = state.cache_data.setdefault("calibration", {})
        for base in moved:
            state.set_mark_reason(base, None, reason)
            # Clear calibration data by removing from dict (presence = marked)
            calib.pop(base, None)
            viewer.invalidate_overlay_cache(base)
            viewer._evict_pixmap_cache_entry(base)
        viewer._bump_signature_epoch()
        viewer._update_delete_button()
        total = session.total_pairs()
        if total:
            viewer.current_index = min(viewer.current_index, total - 1)
            viewer.current_index = max(viewer.current_index, 0)
            viewer.load_current()
        else:
            viewer.current_index = 0
            viewer._show_empty_state()
        if failed:
            QMessageBox.warning(viewer, "Move failed", format_move_failure(failed))
            viewer.statusBar().showMessage(f"Moved {len(moved)}; failed {len(failed)}", 8000)
        elif moved:
            destination = session.loader.to_delete_dir if session.loader else "to_delete"
            QMessageBox.information(viewer, "Moved successfully", format_move_success(len(moved), str(destination)))
            viewer.statusBar().showMessage(f"Moved {len(moved)} marked image(s) to {destination}", 5000)
        else:
            viewer.statusBar().showMessage("No images moved", 5000)
        viewer._update_restore_menu()
        viewer._mark_cache_dirty()

    def cancel_background_jobs(self) -> None:
        """Best-effort cancellation when the UI is closing."""
        self._ignore_delete_results = True
        if self._active_delete_worker:
            try:
                self._active_delete_worker.cancel()
            except Exception:
                pass
        self._teardown_delete_job()

    def _teardown_delete_job(self) -> None:
        self._delete_job_active = False
        self._active_delete_loader = None
        self._active_delete_worker = None
        try:
            self.viewer.progress_tracker.finish(self._delete_task_id)
        except Exception:
            pass
        self._teardown_cursor_reset()

    def restore_images(self) -> None:
        viewer = self.viewer
        session = viewer.session
        if not session.loader:
            QMessageBox.information(viewer, "Restore", restore_prompt_message())
            return
        restored_pairs = session.restore_from_trash()
        if restored_pairs == 0:
            QMessageBox.information(viewer, "Restore", no_restore_items_message())
            viewer._update_restore_menu()
            return
        viewer.invalidate_overlay_cache()
        viewer._bump_signature_epoch()
        total = session.total_pairs()
        if total:
            viewer.current_index = min(viewer.current_index, total - 1)
            viewer.current_index = max(viewer.current_index, 0)
            viewer.load_current()
        else:
            viewer.current_index = 0
            viewer._show_empty_state()
        viewer._update_delete_button()
        viewer._update_restore_menu()
        QMessageBox.information(viewer, "Restore", restored_pairs_message(restored_pairs))
        viewer._mark_cache_dirty()

    def set_calibration_mark(self, base: str, enabled: bool, *, auto: Optional[bool] = None) -> bool:
        """Set or clear calibration mark for a base.

        Args:
            base: Image base name
            enabled: True to mark, False to unmark
            auto: True if marking from auto-detection sweep, False for manual,
                  None to preserve existing auto flag (default for UI toggle)

        Returns:
            True if state changed, False otherwise
        """
        viewer = self.viewer
        state = viewer.state
        if enabled:
            if base in state.calibration_marked:
                return False
            # Use the state method that modifies cache_data directly
            # Pass auto=False for manual marking (new mark), None preserves existing
            state.set_calibration_mark(base, marked=True, auto=auto if auto is not None else False)
            viewer.defer_calibration_analysis(base, force=True)
            viewer.invalidate_overlay_cache(base)
            state.refresh_calibration_entry(base)
            return True
        if base not in state.calibration_marked:
            return False
        viewer.cancel_deferred_calibration(base)
        # Use the state method that modifies cache_data directly
        state.set_calibration_mark(base, marked=False)
        # Delete corner file from disk when unmarking
        viewer.session.delete_corners(base)
        state.remove_calibration_entry(base)
        viewer.statusBar().showMessage(calibration_removed_message(base), 3000)
        viewer.invalidate_overlay_cache(base)
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _find_empty_dataset_dirs(self, root: Path) -> List[Path]:
        empty_dirs: List[Path] = []
        stack: List[Path] = [root]
        visited: Set[Path] = set()
        while stack:
            current = stack.pop()
            if current in visited or not current.exists():
                continue
            visited.add(current)
            has_structure = any((current / name).is_dir() for name in ("lwir", "visible"))
            if has_structure:
                loader = DatasetLoader(str(current))
                if not loader.load_dataset():
                    empty_dirs.append(current)
                continue
            try:
                for child in current.iterdir():
                    if child.is_dir():
                        stack.append(child)
            except OSError:
                continue
        empty_dirs.sort()
        return empty_dirs

    def _parse_calibration_payload(self, payload: Any) -> Tuple[Dict[str, Optional[Dict[str, Any]]], Optional[Dict[str, Any]]]:
        matrices = {"lwir": None, "visible": None}
        extrinsic: Optional[Dict[str, Any]] = None
        if not isinstance(payload, dict):
            return matrices, extrinsic
        for key in matrices.keys():
            section = payload.get(key)
            if not isinstance(section, dict):
                continue
            camera = section.get("camera_matrix")
            distortion = section.get("distortion")
            if not is_matrix3x3(camera) or not is_distortion_vector(distortion):
                continue

            # Validated, now check None explicitly
            if camera is None or distortion is None:
                continue

            matrices[key] = {
                "camera_matrix": [
                    [float(num) for num in row]
                    for row in camera
                ],
                "distortion": [float(num) for num in distortion],
            }
        extrinsic_section = payload.get("extrinsic")
        if isinstance(extrinsic_section, dict):
            rotation = extrinsic_section.get("rotation")
            translation = extrinsic_section.get("translation")
            if is_matrix3x3(rotation) and is_vector3(translation):
                # Check None explicitly
                if translation is None:
                    pass
                else:
                    extrinsic_data: Dict[str, Any] = {
                        "rotation": rotation,
                        "translation": [float(num) for num in translation],
                    }
                for key in ("essential_matrix", "fundamental_matrix"):
                    matrix = extrinsic_section.get(key)
                    if is_matrix3x3(matrix) and extrinsic_data is not None:
                        extrinsic_data[key] = matrix
                baseline = extrinsic_section.get("baseline") or extrinsic_section.get("baseline_mm")
                if isinstance(baseline, (int, float)) and extrinsic_data is not None:
                    extrinsic_data["baseline"] = float(baseline)
                samples = extrinsic_section.get("samples")
                if isinstance(samples, int) and extrinsic_data is not None:
                    extrinsic_data["samples"] = samples
                reproj = extrinsic_section.get("reprojection_error")
                if isinstance(reproj, (int, float)) and extrinsic_data is not None:
                    extrinsic_data["reprojection_error"] = float(reproj)
                extrinsic = extrinsic_data
        return matrices, extrinsic
