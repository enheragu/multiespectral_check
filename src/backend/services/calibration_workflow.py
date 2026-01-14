"""Calibration workflow controller managing calibration targets, queuing, and debug bundles.

Collects calibration targets, manages queuing, emits debug bundles, and keeps cached results
normalized before refreshing overlays and UI state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QMessageBox

from backend.services.calibration.calibration_extrinsic_solver import CalibrationExtrinsicSample
from backend.services.calibration.calibration_solver import CalibrationSample
from common.log_utils import log_debug

if TYPE_CHECKING:
    from backend.services.dataset_session import DatasetSession
    from backend.services.viewer_state import ViewerState
    from backend.services.calibration.calibration_debugger import CalibrationDebugger
    from backend.services.calibration.calibration_controller import CalibrationController


class CalibrationWorkflow(QObject):
    """Manages calibration workflow including target collection, queuing, and result handling."""

    calibrationUpdated = pyqtSignal(str)
    calibrationDetectionCompleted = pyqtSignal(str, bool)  # base, has_detection
    statusMessage = pyqtSignal(str, int)

    def __init__(
        self,
        session: DatasetSession,
        state: ViewerState,
        calibration_debugger: CalibrationDebugger,
        calibration_controller: CalibrationController,
        parent: Optional[QObject] = None,
        # Callbacks
        invalidate_overlay_cache: Optional[Callable[[str], None]] = None,
        load_image_pair: Optional[Callable[[str], None]] = None,
        get_current_base: Optional[Callable[[], Optional[str]]] = None,
        update_stats_panel: Optional[Callable[[], None]] = None,
        mark_cache_dirty: Optional[Callable[[], None]] = None,
    ):
        super().__init__(parent)
        self.session = session
        self.state = state
        self.calibration_debugger = calibration_debugger
        self.calibration_controller = calibration_controller

        self._invalidate_overlay_cache = invalidate_overlay_cache or (lambda base: None)
        self._load_image_pair = load_image_pair or (lambda base: None)
        self._get_current_base = get_current_base or (lambda: None)
        self._update_stats_panel = update_stats_panel or (lambda: None)
        self._mark_cache_dirty = mark_cache_dirty or (lambda: None)

    def collect_calibration_targets(self, *, force: bool = False) -> List[str]:
        """Collect bases that need calibration analysis."""
        # Get image bases from loader or collection
        if self.session.collection:
            image_bases = self.session.collection.image_bases
        elif self.session.loader:
            image_bases = self.session.loader.image_bases
        else:
            return []

        pending: List[str] = []
        for base in image_bases:
            if base not in self.state.calibration_marked:
                continue
            results = self.state.calibration_results.get(base)
            # Don't load corners here - needs_corner_refresh only checks results
            # Corners will be lazy-loaded when detection is actually run
            if force or self.calibration_debugger.needs_corner_refresh(results, None):
                pending.append(base)
        return pending

    def prime_calibration_jobs(self, limit: Optional[int] = None) -> None:
        """Prefetch calibration jobs up to a limit."""
        pending = self.collect_calibration_targets()
        if limit is not None and limit > 0:
            pending = pending[:limit]
        if not pending:
            return
        queued = self.calibration_controller.prefetch(pending, force=False)
        if queued:
            self.statusMessage.emit(f"Preparando {queued} overlay(s) de calibración…", 3000)

    def handle_run_calibration_action(self, parent_widget: QObject) -> None:
        """Handle user request to run calibration detection on marked images."""
        if not self.session.loader and not self.session.collection:
            QMessageBox.information(parent_widget, "Calibration", "Load a dataset before running calibration detection.")
            return
        pending = self.collect_calibration_targets(force=True)
        if not pending:
            # No images marked for calibration - suggest using auto-search
            reply = QMessageBox.question(
                parent_widget,
                "Detect chessboards",
                "No images are marked as calibration candidates.\n\n"
                "Would you like to auto-search for calibration patterns in all images?\n"
                "(This will scan images not marked for deletion)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                # Trigger auto-search - parent_widget should be ImageViewer
                if hasattr(parent_widget, "_handle_auto_calibration_search"):
                    parent_widget._handle_auto_calibration_search()
            return
        reply = QMessageBox.question(
            parent_widget,
            "Detect chessboards",
            f"Run chessboard detection on {len(pending)} calibration candidate(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        queued = self.calibration_controller.prefetch(pending, force=True)
        self.statusMessage.emit(f"Queued chessboard detection for {queued} image(s)", 6000)

    def ensure_calibration_analysis(self, base: str) -> None:
        """Ensure calibration analysis is scheduled for a base if needed."""
        if base not in self.state.calibration_marked:
            return
        results = self.state.calibration_results.get(base)
        corners = self.session.get_corners(base)  # Lazy load from disk
        if self.calibration_debugger.needs_corner_refresh(results, corners):
            if self.schedule_calibration_job(base, priority=True):
                self.statusMessage.emit(f"Analyzing calibration for {base}…", 2000)

    def emit_calibration_debug(
        self,
        base: str,
        results: Optional[Dict[str, Optional[bool]]],
        corners: Optional[Dict[str, Optional[List[List[float]]]]],
        bundle: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit calibration debug information and save debug bundle."""
        if not bundle:
            bundle = self.calibration_debugger.build_cached_debug_bundle(base, results, corners)
            if not bundle:
                self.statusMessage.emit(
                    f"No debug frames available for {base}. Run calibration detection first.",
                    4000,
                )
                return
        status_msg = self.calibration_debugger.format_status(
            base,
            results,
            corners,
            include_counts=True,
        )
        saved_path, log_lines = self.calibration_debugger.save_debug_bundle(base, bundle)
        if log_lines:
            for line in log_lines:
                log_debug(line, "CALIB_WORKFLOW")
        if saved_path:
            status_msg = f"{status_msg} | Debug saved to {saved_path.name}"
        self.statusMessage.emit(status_msg, 6000)

    def schedule_calibration_job(self, base: str, *, force: bool = False, priority: bool = False) -> bool:
        """Schedule a calibration job for a specific base."""
        # Support both datasets (loader) and collections
        if not (self.session.loader or self.session.collection) or not base or base not in self.state.calibration_marked:
            return False
        results = self.state.calibration_results.get(base)
        corners = self.session.get_corners(base)  # Lazy load from disk
        if not force and not self.calibration_debugger.needs_corner_refresh(results, corners):
            return False
        scheduled = self.calibration_controller.schedule(
            base,
            force=force,
            priority="high" if priority else "normal",
        )
        return scheduled

    def handle_calibration_ready(
        self,
        base: str,
        results: Dict[str, Optional[bool]],
        corners: Dict[str, Optional[List[List[float]]]],
    ) -> None:
        """Handle calibration results when ready."""
        normalized_results = dict(results or {})
        normalized_corners = dict(corners or {})
        self._sanitize_calibration_payload(base, normalized_results, normalized_corners)

        # Check if at least one channel detected a pattern
        has_detection = any(v is True for v in normalized_results.values())

        # Auto-mark as calibration if pattern detected and not already marked
        was_unmarked = base not in self.state.calibration_marked
        if was_unmarked:
            if has_detection:
                # Auto-mark this image as calibration candidate
                calib_dict = self.state.cache_data.setdefault("calibration", {})
                if base not in calib_dict:
                    calib_dict[base] = {}
                calib_dict[base]["marked"] = True
                calib_dict[base]["auto"] = True  # Track that this was auto-detected
                log_debug(f"Auto-marked {base} as calibration candidate (pattern detected)", "CALIB_WORKFLOW")
            else:
                # No pattern detected and not marked - emit signal and skip storing
                self.calibrationDetectionCompleted.emit(base, False)
                return

        # Use the state method to update results (not property assignment)
        self.state.set_calibration_results(base, normalized_results)
        self.session.set_corners(base, normalized_corners)  # Save corners to disk immediately
        self.state.refresh_calibration_entry(base)
        self._invalidate_overlay_cache(base)
        current = self._get_current_base()
        if current == base:
            status_msg = self.calibration_debugger.format_status(
                base,
                normalized_results,
                normalized_corners,
                include_counts=True,
            )
            self._load_image_pair(base)
            self.statusMessage.emit(status_msg, 6000)
        self._update_stats_panel()
        self._mark_cache_dirty()
        self.calibrationUpdated.emit(base)

        # Emit detection completed signal for progress tracking
        self.calibrationDetectionCompleted.emit(base, has_detection)

    def handle_calibration_failed(self, base: str, message: str) -> None:
        """Handle calibration failure."""
        self.statusMessage.emit(f"Calibration analysis failed for {base}: {message}", 6000)

    def _sanitize_calibration_payload(
        self,
        base: str,
        results: Dict[str, Optional[bool]],
        corners: Dict[str, Optional[List[List[float]]]],
    ) -> None:
        """Sanitize calibration payload (currently no-op, kept for future use)."""
        return

    def count_calibration_samples(self) -> Dict[str, int]:
        """Quick count of calibration samples WITHOUT loading corners from disk.

        Returns dict with channel counts based on cached results.
        """
        counts: Dict[str, int] = {"lwir": 0, "visible": 0}
        image_bases = self.session.get_all_bases()
        if not image_bases:
            return counts

        out_intrinsic = self.state.calibration_outliers_intrinsic
        for base in image_bases:
            if base not in self.state.calibration_marked:
                continue
            # Use cached results instead of loading corners
            results = self.state.calibration_results.get(base, {})
            for channel in ("lwir", "visible"):
                if base in out_intrinsic.get(channel, set()):
                    continue
                # Check if detection was successful
                if results.get(channel) is True:
                    counts[channel] += 1
        return counts

    def collect_calibration_samples(self) -> List[CalibrationSample]:
        """Collect calibration samples for intrinsic calibration."""
        samples: List[CalibrationSample] = []

        # Get image bases from loader or collection
        image_bases = self.session.get_all_bases()
        if not image_bases:
            return samples

        out_intrinsic = self.state.calibration_outliers_intrinsic
        for base in image_bases:
            if base not in self.state.calibration_marked:
                continue
            bucket = self.session.get_corners(base) or {}  # Lazy load corners
            for channel in ("lwir", "visible"):
                if base in out_intrinsic.get(channel, set()):
                    continue
                points = bucket.get(channel)
                if not points:
                    continue
                image_path = self.session.get_image_path(base, channel)
                if not image_path or not image_path.exists():
                    continue
                samples.append(
                    CalibrationSample(
                        base=base,
                        channel=channel,
                        image_path=image_path,
                        corners=points,
                    )
                )
        return samples

    def count_extrinsic_samples(self) -> int:
        """Quick count of extrinsic samples WITHOUT loading corners.

        Counts images where BOTH channels have successful detection.
        """
        count = 0
        image_bases = self.session.get_all_bases()
        if not image_bases:
            return count

        out_extrinsic = self.state.calibration_outliers_extrinsic
        for base in image_bases:
            if base not in self.state.calibration_marked or base in out_extrinsic:
                continue
            results = self.state.calibration_results.get(base, {})
            # Need both channels detected
            if results.get("lwir") is True and results.get("visible") is True:
                count += 1
        return count

    def collect_extrinsic_samples(self) -> List[CalibrationExtrinsicSample]:
        """Collect samples for extrinsic calibration."""
        samples: List[CalibrationExtrinsicSample] = []

        # Get image bases from loader or collection
        image_bases = self.session.get_all_bases()
        if not image_bases:
            return samples

        out_extrinsic = self.state.calibration_outliers_extrinsic
        for base in image_bases:
            if base not in self.state.calibration_marked or base in out_extrinsic:
                continue
            bucket = self.session.get_corners(base) or {}  # Lazy load corners
            lwir_points = bucket.get("lwir")
            visible_points = bucket.get("visible")
            if not lwir_points or not visible_points:
                continue
            lwir_path = self.session.get_image_path(base, "lwir")
            vis_path = self.session.get_image_path(base, "visible")
            if not lwir_path or not vis_path or not lwir_path.exists() or not vis_path.exists():
                continue
            samples.append(
                CalibrationExtrinsicSample(
                    base=base,
                    lwir_path=lwir_path,
                    visible_path=vis_path,
                    lwir_corners=lwir_points,
                    visible_corners=visible_points,
                )
            )
        return samples
