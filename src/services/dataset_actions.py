from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import yaml
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from dataset_loader import DatasetLoader
from services.cache_service import is_distortion_vector, is_matrix3x3, is_vector3
from utils.ui_messages import (
    calibration_removed_message,
    format_move_confirmation,
    format_move_failure,
    format_move_success,
    no_restore_items_message,
    restore_prompt_message,
    restored_pairs_message,
)

if TYPE_CHECKING:  # pragma: no cover - import guard
    from image_viewer import ImageViewer


class DatasetActions:
    def __init__(self, viewer: "ImageViewer", default_dataset_dir: str) -> None:
        self.viewer = viewer
        self.default_dataset_dir = default_dataset_dir

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
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle)
        except (OSError, yaml.YAMLError) as exc:  # noqa: BLE001
            QMessageBox.warning(viewer, "Calibration data", f"Could not read file: {exc}")
            return
        matrices, extrinsic = self._parse_calibration_payload(payload)
        if not any(matrices.values()):
            QMessageBox.warning(viewer, "Calibration data", "No valid matrices found in file.")
            return
        for key, data in matrices.items():
            if data:
                viewer.state.calibration_matrices[key] = data
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

    def delete_marked_images(self) -> None:
        viewer = self.viewer
        session = viewer.session
        state = viewer.state
        if not session.loader or not state.marked_for_delete:
            return
        count = len(state.marked_for_delete)
        reply = QMessageBox.question(
            viewer,
            "Move to to_delete",
            format_move_confirmation(count),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        outcome = session.delete_marked_entries()
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

    def set_calibration_mark(self, base: str, enabled: bool) -> bool:
        viewer = self.viewer
        state = viewer.state
        if enabled:
            if base in state.calibration_marked:
                return False
            state.calibration_marked.add(base)
            viewer.defer_calibration_analysis(base, force=True)
            viewer.invalidate_overlay_cache(base)
            state.refresh_calibration_entry(base)
            return True
        if base not in state.calibration_marked:
            return False
        viewer.cancel_deferred_calibration(base)
        state.calibration_marked.remove(base)
        state.calibration_results.pop(base, None)
        state.calibration_corners.pop(base, None)
        state.calibration_warnings.pop(base, None)
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
                extrinsic = {
                    "rotation": rotation,
                    "translation": [float(num) for num in translation],
                }
                for key in ("essential_matrix", "fundamental_matrix"):
                    matrix = extrinsic_section.get(key)
                    if is_matrix3x3(matrix):
                        extrinsic[key] = matrix
                baseline = extrinsic_section.get("baseline") or extrinsic_section.get("baseline_mm")
                if isinstance(baseline, (int, float)):
                    extrinsic["baseline"] = float(baseline)
                samples = extrinsic_section.get("samples")
                if isinstance(samples, int):
                    extrinsic["samples"] = samples
                reproj = extrinsic_section.get("reprojection_error")
                if isinstance(reproj, (int, float)):
                    extrinsic["reprojection_error"] = float(reproj)
        return matrices, extrinsic
