"""Stereo calibration solver to recover the rigid transform between LWIR and visible cameras.

Runs a background task that aggregates per-pair chessboard samples, computes extrinsics, and persists
results back into the dataset calibration file with detailed per-pair errors.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

from services.calibration.calibration_solver import CALIBRATION_RESULTS_FILENAME
from services.dataset_session import DatasetSession

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore
    np = None  # type: ignore


@dataclass(frozen=True)
class CalibrationExtrinsicSample:
    base: str
    lwir_path: Path
    visible_path: Path
    lwir_corners: Sequence[Tuple[float, float]]
    visible_corners: Sequence[Tuple[float, float]]


class _ExtrinsicTaskSignals(QObject):
    completed = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()


class _CalibrationExtrinsicTask(QRunnable):
    def __init__(
        self,
        dataset_path: Path,
        pattern_size: Tuple[int, int],
        samples: List[CalibrationExtrinsicSample],
        lwir_intrinsic: dict,
        visible_intrinsic: dict,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.pattern_size = pattern_size
        self.samples = samples
        self.lwir_intrinsic = lwir_intrinsic
        self.visible_intrinsic = visible_intrinsic
        self.signals = _ExtrinsicTaskSignals()
        self._cancel_event = Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def _ensure_not_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise RuntimeError("Extrinsic solve cancelled")

    def _object_points(self) -> Any:
        cols, rows = self.pattern_size
        objp = np.zeros((rows * cols, 3), dtype=np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        return objp

    def _load_gray(self, path: Path) -> Optional[Any]:
        self._ensure_not_cancelled()
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        return image

    def _normalized_to_pixels(self, corners: Sequence[Tuple[float, float]], width: int, height: int) -> Any:
        return np.array(
            [[float(u) * width, float(v) * height] for (u, v) in corners],
            dtype=np.float32,
        )

    def _prepare_samples(self) -> Tuple[List[Any], List[Any], List[Any], Tuple[int, int]]:
        self._ensure_not_cancelled()
        obj_points: List[Any] = []
        lwir_points: List[Any] = []
        visible_points: List[Any] = []
        obj_pattern = self._object_points()
        expected = obj_pattern.shape[0]
        lwir_size: Optional[Tuple[int, int]] = None
        visible_size: Optional[Tuple[int, int]] = None
        for sample in self.samples:
            self._ensure_not_cancelled()
            lwir = self._load_gray(sample.lwir_path)
            visible = self._load_gray(sample.visible_path)
            if lwir is None or visible is None:
                continue
            lwir_height, lwir_width = lwir.shape[:2]
            vis_height, vis_width = visible.shape[:2]
            if len(sample.lwir_corners) != expected or len(sample.visible_corners) != expected:
                continue
            if lwir_size and lwir_size != (lwir_width, lwir_height):
                continue
            if visible_size and visible_size != (vis_width, vis_height):
                continue
            obj_points.append(obj_pattern.copy())
            lwir_points.append(self._normalized_to_pixels(sample.lwir_corners, lwir_width, lwir_height))
            visible_points.append(self._normalized_to_pixels(sample.visible_corners, vis_width, vis_height))
            lwir_size = lwir_size or (lwir_width, lwir_height)
            visible_size = visible_size or (vis_width, vis_height)
        image_size = visible_size or lwir_size
        if len(obj_points) < 3 or image_size is None:
            raise RuntimeError(
                f"Could not prepare enough valid samples for extrinsic calibration ({len(obj_points)} usable). "
                "Ensure both channels have detections with consistent resolutions."
            )
        return obj_points, lwir_points, visible_points, image_size

    def _convert_intrinsics(self, source: dict) -> Tuple[Any, Any]:
        camera = np.array(source.get("camera_matrix"), dtype=np.float64)
        distortion = np.array(source.get("distortion"), dtype=np.float64)
        return camera, distortion.reshape(-1, 1)

    def _persist_results(self, payload: dict) -> dict:
        self._ensure_not_cancelled()
        output_path = self.dataset_path / CALIBRATION_RESULTS_FILENAME
        existing: dict = {}
        if output_path.exists():
            try:
                self._ensure_not_cancelled()
                with open(output_path, "r", encoding="utf-8") as handle:
                    existing = yaml.safe_load(handle) or {}
            except (OSError, yaml.YAMLError):  # noqa: PERF203
                existing = {}
        existing.setdefault("channels", {})
        existing["pattern_size"] = list(self.pattern_size)
        existing["updated_at"] = datetime.utcnow().isoformat() + "Z"
        existing["extrinsic"] = payload
        try:
            self._ensure_not_cancelled()
            with open(output_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(existing, handle, sort_keys=False)
        except OSError as exc:  # noqa: BLE001
            raise RuntimeError(f"Could not write calibration file: {exc}") from exc
        payload_with_path = dict(payload)
        payload_with_path["file_path"] = str(output_path)
        return payload_with_path

    def run(self) -> None:  # noqa: D401
        try:
            self._ensure_not_cancelled()
            obj_points, lwir_points, visible_points, image_size = self._prepare_samples()
            camera_lwir, dist_lwir = self._convert_intrinsics(self.lwir_intrinsic)
            camera_visible, dist_visible = self._convert_intrinsics(self.visible_intrinsic)
            criteria = (
                cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
                100,
                1e-5,
            )
            flags = cv2.CALIB_FIX_INTRINSIC
            self._ensure_not_cancelled()
            retval, _, _, _, _, rotation, translation, essential, fundamental = cv2.stereoCalibrate(
                obj_points,
                lwir_points,
                visible_points,
                camera_lwir,
                dist_lwir,
                camera_visible,
                dist_visible,
                image_size,
                criteria=criteria,
                flags=flags,
            )
            per_pair_errors: List[Dict[str, float]] = []
            for idx, sample in enumerate(self.samples):
                if idx >= len(obj_points):
                    break
                # Estimate per-view poses via solvePnP to gauge pair consistency.
                try:
                    rvec_lwir, tvec_lwir = cv2.solvePnP(
                        obj_points[idx],
                        lwir_points[idx],
                        camera_lwir,
                        dist_lwir,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )[1:3]
                    rvec_vis, tvec_vis = cv2.solvePnP(
                        obj_points[idx],
                        visible_points[idx],
                        camera_visible,
                        dist_visible,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )[1:3]
                    r_lwir, _ = cv2.Rodrigues(rvec_lwir)
                    r_vis, _ = cv2.Rodrigues(rvec_vis)
                    r_lv = r_vis @ r_lwir.T
                    t_lv = tvec_vis - r_lv @ tvec_lwir
                    rot_delta, _ = cv2.Rodrigues(r_lv @ rotation.T)
                    rot_err = float(np.linalg.norm(rot_delta) * 180.0 / np.pi)
                    trans_err = float(np.linalg.norm(t_lv.reshape(-1) - translation.reshape(-1)))
                    per_pair_errors.append(
                        {
                            "base": sample.base,
                            "translation_error": trans_err,
                            "rotation_error_deg": rot_err,
                        }
                    )
                except Exception:
                    continue
            payload = {
                "rotation": rotation.tolist(),
                "translation": translation.reshape(-1).tolist(),
                "essential_matrix": essential.tolist(),
                "fundamental_matrix": fundamental.tolist(),
                "baseline": float(np.linalg.norm(translation)),
                "samples": len(obj_points),
                "reprojection_error": float(retval),
                "per_pair_errors": per_pair_errors,
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
            enriched = self._persist_results(payload)
            self.signals.completed.emit(enriched)
        except Exception as exc:  # noqa: BLE001
            self.signals.failed.emit(str(exc))


class CalibrationExtrinsicSolver(QObject):
    """Background runner for stereo calibration tasks."""

    extrinsicSolved = pyqtSignal(dict)
    extrinsicFailed = pyqtSignal(str)

    def __init__(
        self,
        session: DatasetSession,
        pattern_size: Tuple[int, int],
        thread_pool: Optional[QThreadPool] = None,
    ) -> None:
        super().__init__()
        self.session = session
        self.pattern_size = pattern_size
        self.thread_pool = thread_pool or QThreadPool.globalInstance()
        self._active_task: Optional[_CalibrationExtrinsicTask] = None

    def solve(self, samples: Iterable[CalibrationExtrinsicSample]) -> bool:
        if cv2 is None or np is None:
            self.extrinsicFailed.emit("OpenCV is required for stereo calibration")
            return False
        dataset_path = self.session.dataset_path
        if not dataset_path:
            self.extrinsicFailed.emit("Dataset path unavailable")
            return False
        if self._active_task is not None:
            self.extrinsicFailed.emit("Stereo calibration is already running")
            return False
        lwir_intrinsic = self.session.state.calibration_matrices.get("lwir")
        visible_intrinsic = self.session.state.calibration_matrices.get("visible")
        if not (lwir_intrinsic and visible_intrinsic):
            self.extrinsicFailed.emit("Compute individual camera matrices before running extrinsic calibration")
            return False
        sample_list = list(samples)
        if len(sample_list) < 3:
            self.extrinsicFailed.emit("Need at least 3 paired calibration samples for extrinsic solve")
            return False
        task = _CalibrationExtrinsicTask(
            dataset_path,
            self.pattern_size,
            sample_list,
            lwir_intrinsic,
            visible_intrinsic,
        )
        task.signals.completed.connect(self._handle_task_completed)
        task.signals.failed.connect(self._handle_task_failed)
        self._active_task = task
        self.thread_pool.start(task)
        return True

    def cancel(self) -> bool:
        if not self._active_task:
            return False
        self._active_task.cancel()
        return True

    def _handle_task_completed(self, payload: dict) -> None:
        self._active_task = None
        self.extrinsicSolved.emit(payload)

    def _handle_task_failed(self, message: str) -> None:
        self._active_task = None
        self.extrinsicFailed.emit(message)
