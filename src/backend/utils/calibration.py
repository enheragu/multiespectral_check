"""Utilities for calibration tagging, chessboard detection, and rectification."""
from __future__ import annotations

import atexit
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple

from PyQt6.QtGui import QImage, QPixmap

from common.log_utils import log_debug
from config import get_config

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore
    np = None  # type: ignore


CalibrationResult = Dict[str, Optional[bool]]
CalibrationCorners = Dict[str, Optional[List[List[float]]]]
CalibrationDebugPayload = Dict[str, Any]
CalibrationDebugBundle = Dict[str, Optional[CalibrationDebugPayload]]

EnhancerFunc = Callable[[Any, Optional["_ColorSpaceCache"]], Optional[Any]]


def _default_enhancement_worker_cap() -> int:
    cpu_count = os.cpu_count() or 2
    return max(1, min(2, max(1, cpu_count // 2)))


def _resolve_enhancement_worker_cap() -> int:
    override = os.environ.get("CALIBRATION_ENHANCEMENT_MAX_WORKERS")
    if override:
        try:
            value = int(override)
            if value > 0:
                return value
        except ValueError:
            pass
    return _default_enhancement_worker_cap()


CALIBRATION_ENHANCEMENT_MAX_WORKERS = _resolve_enhancement_worker_cap()

_ENHANCEMENT_EXECUTOR: Optional[ThreadPoolExecutor] = None
_ENHANCEMENT_EXECUTOR_LOCK = Lock()


def _get_enhancement_executor() -> Optional[ThreadPoolExecutor]:
    global _ENHANCEMENT_EXECUTOR
    if CALIBRATION_ENHANCEMENT_MAX_WORKERS <= 0:
        return None
    if _ENHANCEMENT_EXECUTOR is None:
        with _ENHANCEMENT_EXECUTOR_LOCK:
            if _ENHANCEMENT_EXECUTOR is None:
                _ENHANCEMENT_EXECUTOR = ThreadPoolExecutor(
                    max_workers=CALIBRATION_ENHANCEMENT_MAX_WORKERS,
                    thread_name_prefix="calib-enh",
                )
                atexit.register(_ENHANCEMENT_EXECUTOR.shutdown, wait=False)
    return _ENHANCEMENT_EXECUTOR


class _ColorSpaceCache:
    """Thread-safe lazy cache for derived color spaces."""

    def __init__(self, array: Any) -> None:
        self._array = array
        self._gray: Optional[Any] = None
        self._lab: Optional[Any] = None
        self._hsv: Optional[Any] = None
        self._lock = Lock()

    def rgb(self) -> Any:
        return self._array

    def _compute(self, attr: str, converter: Callable[[Any], Any]) -> Optional[Any]:
        value = getattr(self, attr)
        if value is not None:
            return value
        with self._lock:
            value = getattr(self, attr)
            if value is None:
                try:
                    value = converter(self._array)
                except Exception:  # noqa: BLE001
                    value = None
                setattr(self, attr, value)
        return value

    def gray(self) -> Optional[Any]:
        if cv2 is None:
            return None
        return self._compute("_gray", lambda arr: cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY))

    def lab(self) -> Optional[Any]:
        if cv2 is None:
            return None
        return self._compute("_lab", lambda arr: cv2.cvtColor(arr, cv2.COLOR_RGB2LAB))

    def hsv(self) -> Optional[Any]:
        if cv2 is None:
            return None
        return self._compute("_hsv", lambda arr: cv2.cvtColor(arr, cv2.COLOR_RGB2HSV))


def _run_enhanced_detection(
    label: str,
    enhancer: EnhancerFunc,
    cache: _ColorSpaceCache,
    pattern_size: Tuple[int, int],
    debug: bool,
) -> Tuple[str, Tuple[Optional[bool], Optional[List[List[float]]], Optional[CalibrationDebugPayload]]]:
    source = cache.rgb()
    try:
        enhanced = enhancer(source, cache)
    except TypeError:
        # Fallback for enhancers that don't accept cache parameter
        enhanced = enhancer(source, None)  # type: ignore[call-arg]
    except Exception:  # noqa: BLE001
        return label, (False, None, None)
    if enhanced is None:
        return label, (False, None, None)
    return label, _run_chessboard_detection(enhanced, pattern_size, debug)


def _monotonic_increasing(values: List[float], tolerance: float) -> bool:
    if not values:
        return True
    prev = values[0]
    for current in values[1:]:
        if current + tolerance <= prev:
            return False
        prev = max(prev, current)
    return True


def _span(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(max(values) - min(values))


def _stddev(values: List[float]) -> float:
    count = len(values)
    if count < 2:
        return 0.0
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    return float(max(0.0, variance) ** 0.5)


def _pixmap_to_rgb_array(pixmap: Optional[QPixmap]) -> Optional[Any]:
    if not cv2 or not np or not pixmap or pixmap.isNull():
        return None
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
    width = image.width()
    height = image.height()
    buffer = image.constBits()
    if buffer is None:
        return None
    buffer.setsize(image.sizeInBytes())
    try:
        array = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 3))
    except ValueError:
        return None
    return array.copy()


def _array_to_pixmap(array: Any) -> Optional[QPixmap]:
    if array.ndim != 3 or array.shape[2] not in (1, 3):
        return None
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    height, width, _ = array.shape
    bytes_per_line = width * 3
    image = QImage(array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


def _downscale_for_detection(array: Any, max_edge: int) -> Any:
    if cv2 is None or max_edge <= 0:
        return array
    height, width = array.shape[:2]
    longest = max(height, width)
    if longest <= max_edge:
        return array
    scale = max_edge / float(longest)
    new_size = (
        max(1, int(width * scale)),
        max(1, int(height * scale)),
    )
    return cv2.resize(array, new_size, interpolation=cv2.INTER_AREA)


def _detect_chessboard_from_array(
    array: Any,
    pattern_size: Tuple[int, int],
    debug: bool = False,
) -> Tuple[Optional[bool], Optional[List[List[float]]], Optional[CalibrationDebugPayload]]:
    found, corners, debug_payload = _run_chessboard_detection(array, pattern_size, debug)
    if found:
        return found, corners, debug_payload
    fallback_debug: Optional[CalibrationDebugPayload] = debug_payload
    if not _ENHANCEMENT_PIPELINE:
        return False, None, fallback_debug
    cache = _ColorSpaceCache(array)
    executor = _get_enhancement_executor()
    if executor is None:
        return False, None, fallback_debug
    futures = []
    results: Optional[Tuple[Optional[bool], Optional[List[List[float]]], Optional[CalibrationDebugPayload], str]] = None
    for enhancer_name, enhancer in _ENHANCEMENT_PIPELINE:
        futures.append(
            executor.submit(
                _run_enhanced_detection,
                enhancer_name,
                enhancer,
                cache,
                pattern_size,
                debug,
            )
        )
    for future in as_completed(futures):
        enh_label, enh_result = future.result()
        enh_found, enh_corners, enh_debug = enh_result
        if enh_found:
            for pending in futures:
                if pending is not future:
                    pending.cancel()
            results = (enh_found, enh_corners, enh_debug, enh_label)
            break
        if debug and fallback_debug is None and enh_debug is not None:
            fallback_debug = enh_debug
    if results:
        enh_found, enh_corners, enh_debug, enh_label = results
        if debug and enh_debug is not None and enh_label != "base":
            enh_debug.setdefault("notes", []).append(f"Enhanced via {enh_label}")
        return enh_found, enh_corners, enh_debug
    return False, None, fallback_debug


def _run_chessboard_detection(
    array: Any,
    pattern_size: Tuple[int, int],
    debug: bool = False,
) -> Tuple[Optional[bool], Optional[List[List[float]]], Optional[CalibrationDebugPayload]]:
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape[:2]
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    corners = None
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags)
    else:
        legacy_flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FAST_CHECK
        )
        found, corners = cv2.findChessboardCorners(gray, pattern_size, legacy_flags)
    normalized: Optional[List[List[float]]] = None
    if found and corners is not None:
        normalized = [
            [float(point[0][0]) / width, float(point[0][1]) / height]
            for point in corners
        ]
    debug_payload: Optional[CalibrationDebugPayload] = None
    if debug:
        expected = pattern_size[0] * pattern_size[1]
        overlay = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        text_color = (0, 180, 0) if found else (40, 40, 255)
        if corners is not None:
            cv2.drawChessboardCorners(overlay, pattern_size, corners, bool(found))
        message = "detected" if found else "missing"
        cv2.putText(
            overlay,
            f"{message} ({len(corners) if corners is not None else 0}/{expected})",
            (16, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            text_color,
            2,
            cv2.LINE_AA,
        )
        debug_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        debug_payload = {
            "detected": bool(found) if found is not None else None,
            "corners_found": len(corners) if corners is not None else 0,
            "expected_corners": expected,
            "pixmap": _array_to_pixmap(debug_rgb),
        }
    return bool(found), normalized, debug_payload


def detect_chessboard(
    pixmap: Optional[QPixmap],
    pattern_size: Tuple[int, int],
    debug: bool = False,
) -> Tuple[Optional[bool], Optional[List[List[float]]], Optional[CalibrationDebugPayload]]:
    """Return detection result plus normalized corners and optional debug payload."""
    array = _pixmap_to_rgb_array(pixmap)
    if array is None:
        return None, None, None
    return _detect_chessboard_from_array(array, pattern_size, debug)


def _array_from_path(image_path: Optional[Path]) -> Optional[Any]:
    if not cv2 or not np or not image_path:
        return None
    if not image_path.exists():
        return None
    data = cv2.imread(str(image_path))
    if data is None:
        return None
    return cv2.cvtColor(data, cv2.COLOR_BGR2RGB)


def detect_chessboard_from_path(
    image_path: Optional[Path],
    pattern_size: Tuple[int, int],
) -> Tuple[Optional[bool], Optional[List[List[float]]]]:
    """Detect chessboard corners loading the image directly from disk."""
    array = _array_from_path(image_path)
    if array is None:
        return None, None
    config = get_config()
    array = _downscale_for_detection(array, config.calibration_detection_max_edge)
    found, corners, _ = _detect_chessboard_from_array(array, pattern_size, debug=False)
    return found, corners


def _apply_clahe(array: Any, cache: Optional[_ColorSpaceCache] = None) -> Optional[Any]:
    if cv2 is None:
        return None
    try:
        lab = cache.lab() if cache else cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
        if lab is None:
            return None
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    except Exception:  # noqa: BLE001
        return None


def _apply_bilateral(array: Any, cache: Optional[_ColorSpaceCache] = None) -> Optional[Any]:
    if cv2 is None:
        return None
    try:
        filtered = cv2.bilateralFilter(array, d=5, sigmaColor=75, sigmaSpace=75)
        return filtered
    except Exception:  # noqa: BLE001
        return None


def _apply_unsharp_mask(array: Any, cache: Optional[_ColorSpaceCache] = None) -> Optional[Any]:
    if cv2 is None or np is None:
        return None
    try:
        blurred = cv2.GaussianBlur(array, (0, 0), sigmaX=1.2)
        sharpened = cv2.addWeighted(array, 1.6, blurred, -0.6, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    except Exception:  # noqa: BLE001
        return None


def _apply_visible_boost(array: Any, cache: Optional[_ColorSpaceCache] = None) -> Optional[Any]:
    if cv2 is None:
        return None
    try:
        hsv = cache.hsv() if cache else cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
        if hsv is None:
            return None
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
        v = clahe.apply(v)
        s = cv2.equalizeHist(s)
        boosted = cv2.merge((h, s, v))
        return cv2.cvtColor(boosted, cv2.COLOR_HSV2RGB)
    except Exception:  # noqa: BLE001
        return None


def _apply_gray_equalize(array: Any, cache: Optional[_ColorSpaceCache] = None) -> Optional[Any]:
    if cv2 is None:
        return None
    try:
        gray = cache.gray() if cache else cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        if gray is None:
            return None
        eq = cv2.equalizeHist(gray)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
    except Exception:  # noqa: BLE001
        return None


def _apply_contrast_stretch(array: Any, cache: Optional[_ColorSpaceCache] = None) -> Optional[Any]:
    if cv2 is None:
        return None
    try:
        lab = cache.lab() if cache else cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
        if lab is None:
            return None
        l, a, b = cv2.split(lab)
        stretched = cv2.normalize(l, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        merged = cv2.merge((stretched, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    except Exception:  # noqa: BLE001
        return None


def _apply_gamma_boost(array: Any, cache: Optional[_ColorSpaceCache] = None, gamma: float = 0.7) -> Optional[Any]:
    if cv2 is None or np is None:
        return None
    try:
        gamma = max(0.05, float(gamma))
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(array, table)
    except Exception:  # noqa: BLE001
        return None


def _apply_invert(array: Any, cache: Optional[_ColorSpaceCache] = None) -> Optional[Any]:
    if cv2 is None:
        return None
    try:
        return cv2.bitwise_not(array)
    except Exception:  # noqa: BLE001
        return None


def _apply_adaptive_binary(array: Any, cache: Optional[_ColorSpaceCache] = None) -> Optional[Any]:
    if cv2 is None:
        return None
    try:
        gray = cache.gray() if cache else cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        if gray is None:
            return None
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        binary = cv2.adaptiveThreshold(
            norm,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            5,
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    except Exception:  # noqa: BLE001
        return None


_ENHANCEMENT_PIPELINE = (
    ("clahe", _apply_clahe),
    ("gray_equalize", _apply_gray_equalize),
    ("contrast_stretch", _apply_contrast_stretch),
    ("gamma_boost", _apply_gamma_boost),
    ("bilateral", _apply_bilateral),
    ("unsharp", _apply_unsharp_mask),
    ("visible_boost", _apply_visible_boost),
    ("adaptive_binary", _apply_adaptive_binary),
    ("invert", _apply_invert),
)


def refine_corners_from_path(
    image_path: Optional[Path],
    pattern_size: Tuple[int, int],
    normalized_corners: Optional[List[List[float]]],
) -> Optional[List[List[float]]]:
    """Run OpenCV cornerSubPix on disk-backed images using stored normalized corners."""
    if not normalized_corners:
        return None
    array = _array_from_path(image_path)
    if array is None:
        return None
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape[:2]
    expected = pattern_size[0] * pattern_size[1]
    if len(normalized_corners) != expected:
        return None

    # Keep original corners for validation
    original_corners = np.array(
        [[[float(u) * width, float(v) * height]] for (u, v) in normalized_corners],
        dtype=np.float32,
    )

    # cornerSubPix requires shape (N, 1, 2)
    corners = original_corners.copy()

    # Use very strict criteria for subpixel refinement:
    # - Small window (3x3) to stay local
    # - High precision (0.0001 epsilon)
    # - Limited iterations to prevent drift
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.0001)
    cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)

    # Validate refinement: reject corners that moved too much
    # For TRUE subpixel refinement, corners should move < 0.5 pixels
    # If they move more, the original detection was on a different feature
    MAX_DISPLACEMENT_PIXELS = 0.5
    displacements = np.sqrt(np.sum((corners - original_corners) ** 2, axis=2))

    # Count how many corners moved too much
    bad_count = int(np.sum(displacements > MAX_DISPLACEMENT_PIXELS))
    total = len(corners)

    if bad_count > 0:
        # Selective refinement: keep original for corners that moved too much
        for i in range(len(corners)):
            if displacements[i, 0] > MAX_DISPLACEMENT_PIXELS:
                corners[i] = original_corners[i]
        log_debug(f"Corner refinement: {bad_count}/{total} corners reverted (moved > {MAX_DISPLACEMENT_PIXELS}px)", "CALIB")

    return [[float(point[0][0]) / width, float(point[0][1]) / height] for point in corners]


def evaluate_corner_layout(
    points: Optional[List[List[float]]],
    pattern_size: Tuple[int, int],
    *,
    min_span: float = 0.05,
    tolerance: float = 1e-3,
    jitter_tolerance: float = 0.012,
    min_cell_ratio: float = 0.6,
    max_step_ratio: float = 2.5,
) -> Tuple[bool, str]:
    """Validate ordering, span, and shape of detected chessboard corners."""
    if not points:
        return True, ""
    cols, rows = pattern_size
    expected = cols * rows
    if len(points) != expected:
        return False, f"Corner count mismatch ({len(points)}/{expected})"
    xs = [float(pt[0]) for pt in points]
    ys = [float(pt[1]) for pt in points]
    width_span = _span(xs)
    height_span = _span(ys)
    if width_span < min_span or height_span < min_span:
        return False, "Corner spread too small"
    if width_span * height_span < min_span * min_span:
        return False, "Corner coverage too small"
    ordered_rows = [points[idx * cols : (idx + 1) * cols] for idx in range(rows)]
    min_horizontal_step = (width_span / max(1, cols - 1)) * min_cell_ratio
    for row_idx, row_points in enumerate(ordered_rows):
        row_xs = [pt[0] for pt in row_points]
        if not _monotonic_increasing(row_xs, tolerance):
            return False, f"Row {row_idx + 1} has reversed points"
        row_jitter = _stddev([pt[1] for pt in row_points])
        if row_jitter > jitter_tolerance:
            return False, f"Row {row_idx + 1} is too wavy"
        row_steps = [abs(a - b) for a, b in zip(row_xs, row_xs[1:])]
        if row_steps:
            min_step = min(row_steps)
            max_step = max(row_steps)
            if min_step < max(1e-6, min_horizontal_step):
                return False, f"Row {row_idx + 1} has collapsed spacing"
            if max_step / max(min_step, 1e-6) > max_step_ratio:
                return False, f"Row {row_idx + 1} spacing skewed"
    row_centers = [sum(pt[1] for pt in row) / cols for row in ordered_rows]
    if not _monotonic_increasing(row_centers, tolerance):
        return False, "Rows overlap vertically"
    column_blocks = [[ordered_rows[row_idx][col_idx] for row_idx in range(rows)] for col_idx in range(cols)]
    min_vertical_step = (height_span / max(1, rows - 1)) * min_cell_ratio
    for col_idx, column_points in enumerate(column_blocks):
        column_ys = [pt[1] for pt in column_points]
        if not _monotonic_increasing(column_ys, tolerance):
            return False, f"Column {col_idx + 1} has reversed points"
        column_jitter = _stddev([pt[0] for pt in column_points])
        if column_jitter > jitter_tolerance:
            return False, f"Column {col_idx + 1} is too wavy"
        column_steps = [abs(a - b) for a, b in zip(column_ys, column_ys[1:])]
        if column_steps:
            min_step = min(column_steps)
            max_step = max(column_steps)
            if min_step < max(1e-6, min_vertical_step):
                return False, f"Column {col_idx + 1} has collapsed spacing"
            if max_step / max(min_step, 1e-6) > max_step_ratio:
                return False, f"Column {col_idx + 1} spacing skewed"
    return True, ""


def analyze_pair(
    base: str,
    lwir_pixmap: Optional[QPixmap],
    vis_pixmap: Optional[QPixmap],
    pattern_size: Tuple[int, int],
    debug: bool = False,
) -> Tuple[CalibrationResult, CalibrationCorners, str, CalibrationDebugBundle]:
    """Return calibration detection results, status summary, and optional debug data."""
    lwir_result, lwir_corners, lwir_debug = detect_chessboard(lwir_pixmap, pattern_size, debug=debug)
    vis_result, vis_corners, vis_debug = detect_chessboard(vis_pixmap, pattern_size, debug=debug)
    status_parts: List[str] = []
    expected = pattern_size[0] * pattern_size[1]
    if cv2 is None or np is None:
        status_parts.append("Install opencv-python to enable chessboard detection")
    else:
        specs = (
            ("LWIR", lwir_result, lwir_corners),
            ("Visible", vis_result, vis_corners),
        )
        for label, result, corner_list in specs:
            if result is True:
                entry = f"{label}: detected"
            elif result is False:
                entry = f"{label}: not found"
            else:
                continue
            if debug:
                count = len(corner_list or [])
                entry += f" ({count}/{expected} corners)"
            status_parts.append(entry)
        if not status_parts:
            status_parts.append("Chessboard detection skipped")
    status_msg = f"Calibration tagged for {base} ({'; '.join(status_parts)})"
    corners: CalibrationCorners = {"lwir": lwir_corners, "visible": vis_corners}
    debug_bundle: CalibrationDebugBundle = {"lwir": lwir_debug, "visible": vis_debug}
    return {"lwir": lwir_result, "visible": vis_result}, corners, status_msg, debug_bundle


def analyze_pair_from_paths(
    base: str,
    lwir_path: Optional[Path],
    vis_path: Optional[Path],
    pattern_size: Tuple[int, int],
) -> Tuple[CalibrationResult, CalibrationCorners]:
    """Detect chessboards directly from image files (no QPixmap dependency)."""
    lwir_result, lwir_corners = detect_chessboard_from_path(lwir_path, pattern_size)
    vis_result, vis_corners = detect_chessboard_from_path(vis_path, pattern_size)
    return (
        {"lwir": lwir_result, "visible": vis_result},
        {"lwir": lwir_corners, "visible": vis_corners},
    )


def undistort_pixmap(
    pixmap: Optional[QPixmap],
    camera_matrix: Optional[Any],
    distortion: Optional[Any],
) -> Optional[QPixmap]:
    """Return an undistorted pixmap using the provided calibration parameters."""
    if not pixmap or pixmap.isNull() or camera_matrix is None or distortion is None:
        return pixmap
    if cv2 is None or np is None:
        return pixmap
    array = _pixmap_to_rgb_array(pixmap)
    if array is None:
        return pixmap
    camera = np.array(camera_matrix, dtype=np.float32)
    dist = np.array(distortion, dtype=np.float32).reshape(-1)
    corrected = cv2.undistort(array, camera, dist)
    result = _array_to_pixmap(corrected)
    return result or pixmap
