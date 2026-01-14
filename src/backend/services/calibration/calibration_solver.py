"""Camera calibration solver and persistence."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

from backend.services.dataset_session import DatasetSession
from config import get_config

try:  # Optional import to keep tests lightweight
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - handled at runtime
    cv2 = None
    np = None

@dataclass(frozen=True)
class CalibrationSample:
    base: str
    channel: str
    image_path: Path
    corners: Sequence[List[float]]


class _SolverTaskSignals(QObject):
    completed = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()


class _CalibrationSolverTask(QRunnable):
    def __init__(
        self,
        dataset_path: Path,
        pattern_size: Tuple[int, int],
        samples: List[CalibrationSample],
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.pattern_size = pattern_size
        self.samples = samples
        self.signals = _SolverTaskSignals()
        self._cancel_event = Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def _ensure_not_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise RuntimeError("Calibration solve cancelled")

    def _object_points(self) -> Any:
        cols, rows = self.pattern_size
        objp = np.zeros((rows * cols, 3), dtype=np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        return objp

    def _channel_payload(self, channel_samples: List[CalibrationSample]) -> Optional[dict]:
        self._ensure_not_cancelled()
        if len(channel_samples) < 3:
            return None
        obj_points: List[Any] = []
        img_points: List[Any] = []
        view_ids: List[str] = []
        image_size: Optional[Tuple[int, int]] = None
        obj_pattern = self._object_points()
        expected = obj_pattern.shape[0]
        for sample in channel_samples:
            self._ensure_not_cancelled()
            image = cv2.imread(str(sample.image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            height, width = image.shape[:2]
            if len(sample.corners) != expected:
                continue
            if image_size is None:
                image_size = (width, height)
            obj_points.append(obj_pattern.copy())
            pts = np.array(
                [[float(u) * width, float(v) * height] for (u, v) in sample.corners],
                dtype=np.float32,
            )
            img_points.append(pts)
            view_ids.append(sample.base)
        if not obj_points or not img_points or image_size is None:
            return None
        self._ensure_not_cancelled()
        retval, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
            obj_points,
            img_points,
            image_size,
            None,
            None,
        )
        per_view_errors: Dict[str, float] = {}
        for idx, base in enumerate(view_ids):
            if idx >= len(rvecs) or idx >= len(tvecs):
                continue
            projected, _ = cv2.projectPoints(
                obj_points[idx],
                rvecs[idx],
                tvecs[idx],
                camera_matrix,
                distortion,
            )
            reproj = np.linalg.norm(projected.reshape(-1, 2) - img_points[idx], axis=1)
            if reproj.size:
                per_view_errors[base] = float(np.sqrt(np.mean(reproj ** 2)))
        return {
            "camera_matrix": camera_matrix.tolist(),
            "distortion": distortion.reshape(-1).tolist(),
            "image_size": list(image_size),
            "samples": len(obj_points),
            "reprojection_error": float(retval),
            "per_view_errors": per_view_errors,
        }

    def run(self) -> None:  # noqa: D401
        try:
            self._ensure_not_cancelled()
            channel_map: Dict[str, List[CalibrationSample]] = {"lwir": [], "visible": []}
            for sample in self.samples:
                self._ensure_not_cancelled()
                channel_map.setdefault(sample.channel, []).append(sample)
            payload = {
                "pattern_size": list(self.pattern_size),
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "channels": {},
            }
            for channel, channel_samples in channel_map.items():
                self._ensure_not_cancelled()
                solved = self._channel_payload(channel_samples)
                if solved:
                    payload["channels"][channel] = solved
            if not payload["channels"]:
                raise RuntimeError("Not enough samples to solve calibration")
            config = get_config()
            output_path = self.dataset_path / config.calibration_intrinsic_filename
            final_payload: Dict[str, Any] = {}
            if output_path.exists():
                try:
                    with open(output_path, "r", encoding="utf-8") as handle:
                        self._ensure_not_cancelled()
                        final_payload = yaml.safe_load(handle) or {}
                except (OSError, yaml.YAMLError):  # noqa: PERF203
                    final_payload = {}
            final_payload["channels"] = payload["channels"]
            final_payload["pattern_size"] = payload["pattern_size"]
            final_payload["updated_at"] = payload["updated_at"]
            self._ensure_not_cancelled()
            with open(output_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(final_payload, handle, sort_keys=False)
            final_payload["file_path"] = str(output_path)
            try:
                self.signals.completed.emit(final_payload)
            except RuntimeError:
                # Signals object may be deleted if GUI closed
                pass
        except Exception as exc:  # noqa: BLE001
            try:
                self.signals.failed.emit(str(exc))
            except RuntimeError:
                # Signals object may be deleted if GUI closed
                pass


class CalibrationSolver(QObject):
    """Wrap the QRunnable solver with Qt friendly signals."""

    calibrationSolved = pyqtSignal(dict)
    calibrationFailed = pyqtSignal(str)

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
        self._active_task: Optional[_CalibrationSolverTask] = None

    def solve(self, samples: Iterable[CalibrationSample]) -> bool:
        if cv2 is None or np is None:
            self.calibrationFailed.emit("OpenCV is required for solving calibration")
            return False
        dataset_path = self.session.dataset_path
        if not dataset_path:
            self.calibrationFailed.emit("Dataset path is not available")
            return False
        if self._active_task is not None:
            self.calibrationFailed.emit("Calibration solve is already running")
            return False
        sample_list = list(samples)
        if not sample_list:
            self.calibrationFailed.emit("No calibration samples available")
            return False
        task = _CalibrationSolverTask(dataset_path, self.pattern_size, sample_list)
        task.signals.completed.connect(self._handle_task_completed)
        task.signals.failed.connect(self._handle_task_failed)
        self._active_task = task
        if self.thread_pool is not None:
            self.thread_pool.start(task)
        return True

    def cancel(self) -> bool:
        if not self._active_task:
            return False
        self._active_task.cancel()
        return True

    def _handle_task_completed(self, payload: dict) -> None:
        self._active_task = None
        self.calibrationSolved.emit(payload)

    def _handle_task_failed(self, message: str) -> None:
        self._active_task = None
        self.calibrationFailed.emit(message)
