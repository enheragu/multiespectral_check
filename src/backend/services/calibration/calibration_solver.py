"""Camera calibration solver and persistence."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

from backend.services.dataset_session import DatasetSession
from common.yaml_utils import load_yaml, save_yaml
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
    corners: Sequence[List[float]]
    image_size: Tuple[int, int]  # (width, height)


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
        """Compute calibration for a single channel using pre-loaded corners and sizes."""
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
            # Use image_size from sample (no image loading needed!)
            width, height = sample.image_size
            if len(sample.corners) != expected:
                continue
            if image_size is None:
                image_size = (width, height)
            elif image_size != (width, height):
                # Skip samples with different image sizes
                continue
            obj_points.append(obj_pattern.copy())
            pts = np.array(
                [[float(u) * width, float(v) * height] for (u, v) in sample.corners],
                dtype=np.float32,
            )
            img_points.append(pts)
            view_ids.append(sample.base)

        if not obj_points or not img_points or image_size is None:
            return None

        # Iteratively calibrate and drop views whose per-view reprojection error is a
        # robust (median + k*MAD) outlier, refitting after each round. The solve runs at
        # the top of each round, so the final intrinsics always match the surviving views.
        config = get_config()
        active = list(range(len(obj_points)))
        rejected_ids: List[str] = []
        camera_matrix = distortion = None
        retval = 0.0
        kept_errors: Dict[str, float] = {}
        iteration = 0
        while True:
            self._ensure_not_cancelled()
            retval, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
                [obj_points[i] for i in active],
                [img_points[i] for i in active],
                image_size,
                None,
                None,
            )
            view_err: List[float] = []
            for pos, i in enumerate(active):
                projected, _ = cv2.projectPoints(obj_points[i], rvecs[pos], tvecs[pos], camera_matrix, distortion)
                reproj = np.linalg.norm(projected.reshape(-1, 2) - img_points[i], axis=1)
                view_err.append(float(np.sqrt(np.mean(reproj ** 2))) if reproj.size else float("inf"))
            kept_errors = {view_ids[i]: e for i, e in zip(active, view_err)}
            finite = [e for e in view_err if np.isfinite(e)]
            if not finite:
                break
            med = float(np.median(finite))
            mad = float(np.median([abs(e - med) for e in finite]))
            threshold = max(
                med + config.intrinsic_reject_k_mad * 1.4826 * mad,
                config.intrinsic_reject_floor_px,
            )
            keep = [i for i, e in zip(active, view_err) if e <= threshold]
            if (
                len(keep) == len(active)
                or len(keep) < config.intrinsic_reject_min_views
                or iteration >= config.intrinsic_reject_max_iters - 1
            ):
                break
            rejected_ids.extend(view_ids[i] for i in active if i not in keep)
            active = keep
            iteration += 1

        if camera_matrix is None:
            return None

        # Kept views report their joint-calibration residual; rejected views report their
        # error against the final intrinsics (solvePnP pose) so the outlier dialog shows why
        # they were dropped.
        per_view_errors: Dict[str, float] = dict(kept_errors)
        for i in range(len(obj_points)):
            base = view_ids[i]
            if base in per_view_errors:
                continue
            ok, rvec, tvec = cv2.solvePnP(obj_points[i], img_points[i], camera_matrix, distortion)
            if not ok:
                continue
            projected, _ = cv2.projectPoints(obj_points[i], rvec, tvec, camera_matrix, distortion)
            reproj = np.linalg.norm(projected.reshape(-1, 2) - img_points[i], axis=1)
            if reproj.size:
                per_view_errors[base] = float(np.sqrt(np.mean(reproj ** 2)))

        # Return clean calibration data + errors separately
        return {
            "calibration": {
                "camera_matrix": camera_matrix.tolist(),
                "distortion": distortion.reshape(-1).tolist(),
                "image_size": list(image_size),
                "samples": len(active),
                "rejected_views": len(rejected_ids),
                "reprojection_error": float(retval),
            },
            "per_view_errors": per_view_errors,
            "rejected_views": rejected_ids,
        }

    def run(self) -> None:  # noqa: D401
        try:
            self._ensure_not_cancelled()
            channel_map: Dict[str, List[CalibrationSample]] = {"lwir": [], "visible": []}
            for sample in self.samples:
                self._ensure_not_cancelled()
                channel_map.setdefault(sample.channel, []).append(sample)

            calibration_payload: Dict[str, Any] = {
                "# source": f"Calibration computed for {self.dataset_path.name}",
                "dataset": self.dataset_path.name,
                "dataset_path": str(self.dataset_path),
                "pattern_size": list(self.pattern_size),
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "channels": {},
            }
            errors_payload: Dict[str, Any] = {
                "# source": f"Calibration errors cache for {self.dataset_path.name}",
                "dataset": self.dataset_path.name,
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "channels": {},
            }

            # Parallelize LWIR and Visible calibration (they are independent)
            channels_to_solve = [(ch, samples) for ch, samples in channel_map.items() if samples]
            if len(channels_to_solve) >= 2:
                # Run both channels in parallel
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = {
                        executor.submit(self._channel_payload, samples): channel
                        for channel, samples in channels_to_solve
                    }
                    for future in as_completed(futures):
                        self._ensure_not_cancelled()
                        channel = futures[future]
                        try:
                            solved = future.result()
                            if solved:
                                calibration_payload["channels"][channel] = solved["calibration"]
                                errors_payload["channels"][channel] = {
                                    "per_view_errors": solved["per_view_errors"],
                                    "reprojection_error": solved["calibration"]["reprojection_error"],
                                    "rejected_views": solved.get("rejected_views", []),
                                }
                        except Exception:
                            # Channel failed, continue with other
                            pass
            else:
                # Single channel - run sequentially
                for channel, channel_samples in channels_to_solve:
                    self._ensure_not_cancelled()
                    solved = self._channel_payload(channel_samples)
                    if solved:
                        calibration_payload["channels"][channel] = solved["calibration"]
                        errors_payload["channels"][channel] = {
                            "per_view_errors": solved["per_view_errors"],
                            "reprojection_error": solved["calibration"]["reprojection_error"],
                            "rejected_views": solved.get("rejected_views", []),
                        }

            if not calibration_payload["channels"]:
                raise RuntimeError("Not enough samples to solve calibration")

            config = get_config()

            # Save clean calibration file (exportable)
            output_path = self.dataset_path / config.calibration_intrinsic_filename
            final_payload: Dict[str, Any] = {}
            if output_path.exists():
                self._ensure_not_cancelled()
                final_payload = load_yaml(output_path) or {}
                self._ensure_not_cancelled()
            final_payload["channels"] = calibration_payload["channels"]
            final_payload["pattern_size"] = calibration_payload["pattern_size"]
            final_payload["updated_at"] = calibration_payload["updated_at"]
            if config.chessboard_square_size_mm > 0:
                final_payload["square_size"] = {
                    "value": config.chessboard_square_size_mm,
                    "unit": "mm",
                }
            self._ensure_not_cancelled()
            save_yaml(output_path, final_payload, sort_keys=False)

            # Pre-compute and cache report data (quads + poses for plot charts)
            try:
                from backend.services.calibration.calibration_report_cache import build_report_cache
                build_report_cache(
                    dataset_path=self.dataset_path,
                    matrices={ch: v for ch, v in final_payload.get("channels", {}).items()},
                    file_metadata=final_payload,
                )
            except Exception:  # noqa: BLE001
                pass

            # Save errors to separate cache file (hidden)
            errors_path = self.dataset_path / config.calibration_errors_filename
            self._ensure_not_cancelled()
            save_yaml(errors_path, errors_payload, sort_keys=False)

            # Return combined payload for GUI with per_view_errors for reproj display
            result_payload = dict(final_payload)
            result_payload["file_path"] = str(output_path)
            # Inject per_view_errors + rejected_views back into channels for GUI consumption
            for channel, err_data in errors_payload["channels"].items():
                if channel in result_payload["channels"]:
                    result_payload["channels"][channel]["per_view_errors"] = err_data["per_view_errors"]
                    result_payload["channels"][channel]["rejected_views"] = err_data.get("rejected_views", [])

            try:
                self.signals.completed.emit(result_payload)
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
