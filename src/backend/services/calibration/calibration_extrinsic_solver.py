"""Stereo calibration solver to recover the rigid transform between LWIR and visible cameras.

Runs a background task that aggregates per-pair chessboard samples, computes extrinsics, and persists
results back into the dataset calibration file with detailed per-pair errors.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

from backend.services.dataset_session import DatasetSession
from common.log_utils import log_info
from common.yaml_utils import load_yaml, save_yaml
from config import get_config

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
    lwir_corners: Sequence[List[float]]
    visible_corners: Sequence[List[float]]


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

    def _normalized_to_pixels(self, corners: Sequence[List[float]], width: int, height: int) -> Any:
        return np.array(
            [[float(pt[0]) * width, float(pt[1]) * height] for pt in corners],
            dtype=np.float32,
        )

    def _extract_image_size(self, intrinsic: dict) -> Optional[Tuple[int, int]]:
        """Extract image size from intrinsic calibration dict."""
        size = intrinsic.get("image_size")
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            return (int(size[0]), int(size[1]))
        return None

    def _prepare_samples(self) -> Tuple[List[Any], List[Any], List[Any], List[CalibrationExtrinsicSample], Tuple[int, int]]:
        """Prepare calibration samples using corners from cache (no image loading needed).

        Returns the per-pair object/image points alongside the list of samples that
        produced them (aligned by index, so a skipped sample does not desync later
        per-pair bookkeeping).
        """
        self._ensure_not_cancelled()
        obj_pattern = self._object_points()
        expected = obj_pattern.shape[0]

        # Get image sizes from intrinsic calibration (no need to load images!)
        lwir_size = self._extract_image_size(self.lwir_intrinsic)
        visible_size = self._extract_image_size(self.visible_intrinsic)
        if not lwir_size or not visible_size:
            raise RuntimeError(
                "Could not determine image sizes from intrinsic calibration. "
                "Ensure calibration files contain 'image_size' field."
            )

        obj_points: List[Any] = []
        lwir_points: List[Any] = []
        visible_points: List[Any] = []
        valid_samples: List[CalibrationExtrinsicSample] = []

        for sample in self.samples:
            self._ensure_not_cancelled()
            # Validate corner counts
            if len(sample.lwir_corners) != expected or len(sample.visible_corners) != expected:
                continue
            obj_points.append(obj_pattern.copy())
            lwir_points.append(
                self._normalized_to_pixels(sample.lwir_corners, lwir_size[0], lwir_size[1])
            )
            visible_points.append(
                self._normalized_to_pixels(sample.visible_corners, visible_size[0], visible_size[1])
            )
            valid_samples.append(sample)

        image_size = visible_size
        if len(obj_points) < 3:
            raise RuntimeError(
                f"Could not prepare enough valid samples for extrinsic calibration ({len(obj_points)} usable). "
                "Ensure both channels have detections with consistent corner counts."
            )
        return obj_points, lwir_points, visible_points, valid_samples, image_size

    def _convert_intrinsics(self, source: dict) -> Tuple[Any, Any]:
        camera = np.array(source.get("camera_matrix"), dtype=np.float64)
        distortion = np.array(source.get("distortion"), dtype=np.float64)
        return camera, distortion.reshape(-1, 1)

    def _stereo_calibrate(
        self,
        obj_points: List[Any],
        lwir_points: List[Any],
        visible_points: List[Any],
        camera_lwir: Any,
        dist_lwir: Any,
        camera_visible: Any,
        dist_visible: Any,
        image_size: Tuple[int, int],
    ) -> Tuple[float, Any, Any, Any, Any]:
        """Run stereoCalibrate (intrinsics fixed) on a subset and return (rms, R, T, E, F)."""
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
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
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
        return float(retval), rotation, translation, essential, fundamental

    def _pose_for(self, objp: Any, image_pts: Any, camera: Any, distortion: Any) -> Optional[Tuple[Any, Any]]:
        """Recover the board pose (rotation matrix, tvec) in a camera via solvePnP.

        The pose depends only on that camera's detection, not on the stereo (R, T), so it
        is computed once per pair and reused across the iterative solve and diagnostics.
        Returns None on failure.
        """
        obj = objp.reshape(-1, 1, 3).astype(np.float32)
        pts = image_pts.reshape(-1, 1, 2).astype(np.float32)
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, pts, camera, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            return None
        if not ok:
            return None
        rmat, _ = cv2.Rodrigues(rvec)
        return rmat, tvec

    def _pair_reproj_error(
        self,
        objp: Any,
        lwir_pts: Any,
        visible_pts: Any,
        pose_lwir: Optional[Tuple[Any, Any]],
        pose_visible: Optional[Tuple[Any, Any]],
        camera_lwir: Any,
        dist_lwir: Any,
        camera_visible: Any,
        dist_visible: Any,
        rotation: Any,
        translation: Any,
    ) -> Optional[float]:
        """Per-pair stereo reprojection error (px) from precomputed board poses.

        Maps each camera's board pose across the rigid (R, T) transform and measures the
        pixel residual in the other camera, averaged over both directions. Depth- and
        baseline-independent: a time-desynchronised pair (board moved between captures)
        cannot be explained by a single rigid transform and yields a large residual, so
        it surfaces as an outlier.
        """
        if pose_lwir is None or pose_visible is None:
            return None
        r_lwir, tvec_l = pose_lwir
        r_vis, tvec_v = pose_visible
        obj = objp.reshape(-1, 1, 3).astype(np.float32)
        lwir_pts = lwir_pts.reshape(-1, 1, 2).astype(np.float32)
        visible_pts = visible_pts.reshape(-1, 1, 2).astype(np.float32)
        try:
            # lwir pose -> visible: X_vis = R @ X_lwir + T
            rvec_lv, _ = cv2.Rodrigues(rotation @ r_lwir)
            proj_v, _ = cv2.projectPoints(obj, rvec_lv, rotation @ tvec_l + translation, camera_visible, dist_visible)
            err_v = float(np.sqrt(np.mean(np.sum((proj_v - visible_pts) ** 2, axis=2))))
            # visible pose -> lwir: X_lwir = R^T @ (X_vis - T)
            rvec_vl, _ = cv2.Rodrigues(rotation.T @ r_vis)
            proj_l, _ = cv2.projectPoints(obj, rvec_vl, rotation.T @ (tvec_v - translation), camera_lwir, dist_lwir)
            err_l = float(np.sqrt(np.mean(np.sum((proj_l - lwir_pts) ** 2, axis=2))))
            return 0.5 * (err_v + err_l)
        except cv2.error:
            return None

    def _persist_results(self, payload: dict) -> dict:
        self._ensure_not_cancelled()
        # Save extrinsic calibration to separate file
        config = get_config()
        output_path = self.dataset_path / config.calibration_extrinsic_filename
        try:
            self._ensure_not_cancelled()
            save_yaml(output_path, payload, sort_keys=False)
        except OSError as exc:  # noqa: BLE001
            raise RuntimeError(f"Could not write extrinsic calibration file: {exc}") from exc
        payload_with_path = dict(payload)
        payload_with_path["file_path"] = str(output_path)
        return payload_with_path

    def run(self) -> None:  # noqa: D401
        try:
            self._ensure_not_cancelled()
            obj_points, lwir_points, visible_points, valid_samples, image_size = self._prepare_samples()
            camera_lwir, dist_lwir = self._convert_intrinsics(self.lwir_intrinsic)
            camera_visible, dist_visible = self._convert_intrinsics(self.visible_intrinsic)

            # Board pose per pair is independent of (R, T), so recover it once and reuse it
            # across the iterative solve and the final per-pair diagnostics.
            poses_lwir = [
                self._pose_for(obj_points[i], lwir_points[i], camera_lwir, dist_lwir)
                for i in range(len(obj_points))
            ]
            poses_visible = [
                self._pose_for(obj_points[i], visible_points[i], camera_visible, dist_visible)
                for i in range(len(obj_points))
            ]

            def reproj_for(indices: List[int], rotation: Any, translation: Any) -> List[float]:
                out: List[float] = []
                for i in indices:
                    err = self._pair_reproj_error(
                        obj_points[i], lwir_points[i], visible_points[i],
                        poses_lwir[i], poses_visible[i],
                        camera_lwir, dist_lwir, camera_visible, dist_visible,
                        rotation, translation,
                    )
                    out.append(err if err is not None else float("inf"))
                return out

            # Iteratively solve and drop pairs whose stereo reprojection error is a robust
            # (median + k*MAD) outlier, refitting after each drop. A single rigid (R, T)
            # cannot explain a time-desynchronised pair, so those are isolated here instead
            # of biasing the final solve. The solve runs at the top of each round, so the
            # final (R, T) always matches the surviving `active` set.
            config = get_config()
            active = list(range(len(obj_points)))
            rejected_bases: List[str] = []
            rotation = translation = essential = fundamental = None
            retval = 0.0
            iteration = 0
            while True:
                self._ensure_not_cancelled()
                retval, rotation, translation, essential, fundamental = self._stereo_calibrate(
                    [obj_points[i] for i in active],
                    [lwir_points[i] for i in active],
                    [visible_points[i] for i in active],
                    camera_lwir, dist_lwir, camera_visible, dist_visible, image_size,
                )
                errs = reproj_for(active, rotation, translation)
                finite = [e for e in errs if np.isfinite(e)]
                if not finite:
                    break
                med = float(np.median(finite))
                mad = float(np.median([abs(e - med) for e in finite]))
                threshold = max(
                    med + config.extrinsic_reject_k_mad * 1.4826 * mad,
                    config.extrinsic_reject_floor_px,
                )
                keep = [i for i, e in zip(active, errs) if e <= threshold]
                if (
                    len(keep) == len(active)
                    or len(keep) < config.extrinsic_reject_min_pairs
                    or iteration >= config.extrinsic_reject_max_iters - 1
                ):
                    break
                dropped = [valid_samples[i].base for i in active if i not in keep]
                rejected_bases.extend(dropped)
                log_info(
                    f"Extrinsic outlier rejection iter {iteration}: dropped {len(dropped)} pairs "
                    f"(reproj > {threshold:.2f}px), {len(keep)} remaining, rms {retval:.3f}px",
                    "CALIB",
                )
                active = keep
                iteration += 1

            if rotation is None or translation is None or essential is None or fundamental is None:
                raise RuntimeError("Stereo calibration did not produce a solution")

            # Per-pair diagnostics against the final transform, flagging auto-rejected pairs.
            # stereo_reproj_error is the discriminating metric; translation/rotation deltas
            # are consumed by the calibration check dialog.
            active_set = set(active)
            final_errs = reproj_for(list(range(len(obj_points))), rotation, translation)
            per_pair_errors: List[Dict[str, Any]] = []
            for idx, sample in enumerate(valid_samples):
                entry: Dict[str, Any] = {
                    "base": sample.base,
                    "stereo_reproj_error": (
                        float(final_errs[idx]) if np.isfinite(final_errs[idx]) else None
                    ),
                    "rejected": idx not in active_set,
                }
                pose_l, pose_v = poses_lwir[idx], poses_visible[idx]
                if pose_l is not None and pose_v is not None:
                    r_lwir, tvec_lwir = pose_l
                    r_vis, tvec_vis = pose_v
                    r_lv = r_vis @ r_lwir.T
                    t_lv = tvec_vis - r_lv @ tvec_lwir
                    rot_delta, _ = cv2.Rodrigues(r_lv @ rotation.T)
                    entry["rotation_error_deg"] = float(np.linalg.norm(rot_delta) * 180.0 / np.pi)
                    entry["translation_error"] = float(
                        np.linalg.norm(t_lv.reshape(-1) - translation.reshape(-1))
                    )
                per_pair_errors.append(entry)

            # Single source of truth for the reported error: RMS of the kept pairs' per-pair
            # stereo reprojection error (the same values shown per-pair in the outlier panel).
            # stereoCalibrate's own retval is kept only for the log (different granularity:
            # joint corner optimisation vs per-pair pose mapping).
            kept_errs = [
                e["stereo_reproj_error"] for e in per_pair_errors
                if not e["rejected"] and isinstance(e["stereo_reproj_error"], float)
            ]
            reproj_rms = float(np.sqrt(np.mean(np.square(kept_errs)))) if kept_errs else float(retval)

            log_info(
                f"Extrinsic calibration: {len(rejected_bases)} outlier pairs rejected, "
                f"per-pair RMS {reproj_rms:.3f}px over {len(active)} pairs "
                f"(stereoCalibrate retval {retval:.3f}px)",
                "CALIB",
            )

            # Clean calibration payload (exportable)
            calibration_payload = {
                "# source": f"Stereo calibration computed for {self.dataset_path.name}",
                "dataset": self.dataset_path.name,
                "dataset_path": str(self.dataset_path),
                "rotation": rotation.tolist(),
                "translation": translation.reshape(-1).tolist(),
                "essential_matrix": essential.tolist(),
                "fundamental_matrix": fundamental.tolist(),
                "baseline": float(np.linalg.norm(translation)),
                "samples": len(active),
                "rejected_pairs": len(rejected_bases),
                "reprojection_error": reproj_rms,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            # Save clean extrinsic file
            enriched = self._persist_results(calibration_payload)

            # Save errors to cache file
            errors_path = self.dataset_path / config.calibration_errors_filename
            errors_data: Dict[str, Any] = {}
            if errors_path.exists():
                errors_data = load_yaml(errors_path) or {}
            errors_data["stereo"] = {
                "per_pair_errors": per_pair_errors,
                "reprojection_error": reproj_rms,
                "rejected_pairs": len(rejected_bases),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            save_yaml(errors_path, errors_data, sort_keys=False)

            # Add per_pair_errors to result for GUI consumption
            enriched["per_pair_errors"] = per_pair_errors

            try:
                self.signals.completed.emit(enriched)
            except RuntimeError:
                # Signals object may be deleted if GUI closed
                pass
        except Exception as exc:  # noqa: BLE001
            try:
                self.signals.failed.emit(str(exc))
            except RuntimeError:
                # Signals object may be deleted if GUI closed
                pass


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
        lwir_intrinsic = self.session.state.cache_data["_matrices"].get("lwir")
        visible_intrinsic = self.session.state.cache_data["_matrices"].get("visible")
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
        self.extrinsicSolved.emit(payload)

    def _handle_task_failed(self, message: str) -> None:
        self._active_task = None
        self.extrinsicFailed.emit(message)
