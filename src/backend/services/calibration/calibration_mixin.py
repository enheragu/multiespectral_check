"""Shared calibration workflow helpers used by the main window.

Collects calibration targets, manages queuing, emits debug bundles, and keeps cached results
normalized before refreshing overlays and UI state.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from PyQt6.QtWidgets import QMessageBox

from backend.services.calibration.calibration_extrinsic_solver import CalibrationExtrinsicSample
from backend.services.calibration.calibration_solver import CalibrationSample
from common.log_utils import log_debug

if TYPE_CHECKING:
    from tqdm import tqdm

# Check if tqdm is available
try:
    from tqdm import tqdm as tqdm_class
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm_class = None  # type: ignore


class CalibrationWorkflowMixin:
    """Mixin for calibration workflow - expects host class to provide these attributes."""

    # Type hints for attributes provided by host class (MainWindow)
    session: Any
    state: Any
    calibration_controller: Any
    calibration_debugger: Any
    progress_tracker: Any
    _calib_search_task_id: Optional[str]
    _calib_search_total: int
    _calib_search_completed: int
    _calib_search_found: int
    _calib_search_tqdm: Optional["tqdm"]

    def statusBar(self) -> Any:
        """Host must provide statusBar() method."""

    def invalidate_overlay_cache(self, base: str) -> None:
        """Host must provide overlay cache invalidation."""

    def _current_base(self) -> Optional[str]:
        """Host must provide current base getter."""

    def load_image_pair(self, base: str) -> None:
        """Host must provide image pair loading."""

    def _update_stats_panel(self) -> None:
        """Host must provide stats panel update."""

    def _mark_cache_dirty(self) -> None:
        """Host must provide cache dirty marking."""

    def _safe_status_message(self, message: str, timeout: int = 3000) -> None:
        """Host must provide safe status message."""

    def _handle_auto_calibration_search(self) -> None:
        """Host must provide auto calibration search handler."""

    def _collect_calibration_targets(self, *, force: bool = False) -> List[str]:
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
            corners = self.state.calibration_corners.get(base)
            if force or self.calibration_debugger.needs_corner_refresh(results, corners):
                pending.append(base)
        return pending

    def _prime_calibration_jobs(self, limit: Optional[int] = None) -> None:
        pending = self._collect_calibration_targets()
        if limit is not None and limit > 0:
            pending = pending[:limit]
        if not pending:
            return
        queued = self.calibration_controller.prefetch(pending, force=False)
        if queued:
            self.statusBar().showMessage(
                f"Preparando {queued} overlay(s) de calibración…",
                3000,
            )

    def _handle_run_calibration_action(self) -> None:
        """Handle 'Detect chessboards' action - detect patterns in calibration-marked images."""
        if not self.session.loader and not self.session.collection:
            QMessageBox.information(self, "Calibration", "Load a dataset before running calibration detection.")
            return

        # Get images marked as calibration candidates
        pending = self._collect_calibration_targets(force=True)

        if not pending:
            # No images marked for calibration - suggest using auto-search
            reply = QMessageBox.question(
                self,
                "Detect chessboards",
                "No images are marked as calibration candidates.\n\n"
                "Would you like to auto-search for calibration patterns in all images?\n"
                "(This will scan images not marked for deletion)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                # Trigger auto-search instead
                self._handle_auto_calibration_search()
            return

        reply = QMessageBox.question(
            self,
            "Detect chessboards",
            f"Run chessboard detection on {len(pending)} calibration candidate(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        queued = self.calibration_controller.prefetch(pending, force=True)
        self.statusBar().showMessage(f"Queued chessboard detection for {queued} image(s)", 6000)

    def _ensure_calibration_analysis(self, base: str) -> None:
        if base not in self.state.calibration_marked:
            return
        results = self.state.calibration_results.get(base)
        corners = self.state.calibration_corners.get(base)
        if self.calibration_debugger.needs_corner_refresh(results, corners):
            if self._schedule_calibration_job(base, priority=True):
                self.statusBar().showMessage(f"Analyzing calibration for {base}…", 2000)

    def _emit_calibration_debug(
        self,
        base: str,
        results: Optional[Dict[str, Optional[bool]]],
        corners: Optional[Dict[str, Optional[List[List[float]]]]],
        bundle: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not bundle:
            bundle = self.calibration_debugger.build_cached_debug_bundle(base, results, corners)
            if not bundle:
                self.statusBar().showMessage(
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
                log_debug(line, "CALIB_MIXIN")
        if saved_path:
            status_msg = f"{status_msg} | Debug saved to {saved_path.name}"
        self.statusBar().showMessage(status_msg, 6000)

    def _schedule_calibration_job(self, base: str, *, force: bool = False, priority: bool = False) -> bool:
        # Need loader or collection
        if not self.session.loader and not self.session.collection:
            return False
        if not base or base not in self.state.calibration_marked:
            return False
        results = self.state.calibration_results.get(base)
        corners = self.state.calibration_corners.get(base)
        if not force and not self.calibration_debugger.needs_corner_refresh(results, corners):
            return False
        scheduled = self.calibration_controller.schedule(
            base,
            force=force,
            priority="high" if priority else "normal",
        )
        return bool(scheduled)

    def _handle_calibration_ready(
        self,
        base: str,
        results: Dict[str, Optional[bool]],
        corners: Dict[str, Optional[List[List[float]]]],
    ) -> None:
        normalized_results = dict(results or {})
        normalized_corners = dict(corners or {})
        self._sanitize_calibration_payload(base, normalized_results, normalized_corners)

        # Check if at least one channel detected a pattern
        has_detection = any(v is True for v in normalized_results.values())

        # Auto-mark as calibration if pattern detected and not already marked
        if base not in self.state.calibration_marked:
            if has_detection:
                # Auto-mark this image as calibration candidate
                self.state.set_calibration_mark(base, marked=True, auto=True)
                log_debug(f"Auto-marked {base} as calibration candidate (pattern detected)", "CALIB_MIXIN")
                # Persist corners to YAML file
                self._persist_auto_detected_corners(base, normalized_corners)
            else:
                # No pattern detected and not marked - update progress and skip storing
                self._update_calibration_search_progress(found=False)
                return

        # Store results and corners in calibration dict
        calib_dict = self.state.cache_data.setdefault("calibration", {})
        if base not in calib_dict:
            calib_dict[base] = {}
        calib_dict[base]["results"] = normalized_results
        calib_dict[base]["corners"] = normalized_corners

        self.state.refresh_calibration_entry(base)
        self.invalidate_overlay_cache(base)
        current = self._current_base()
        if current == base:
            status_msg = self.calibration_debugger.format_status(
                base,
                normalized_results,
                normalized_corners,
                include_counts=True,
            )
            self.load_image_pair(base)
            self.statusBar().showMessage(status_msg, 6000)
        self._update_stats_panel()
        self._mark_cache_dirty()

        # Update calibration search progress if active
        self._update_calibration_search_progress(has_detection)

    def _update_calibration_search_progress(self, found: bool) -> None:
        """Update progress for calibration search operation."""
        task_id = getattr(self, "_calib_search_task_id", None)
        if not task_id:
            return

        total = getattr(self, "_calib_search_total", 0)
        completed = getattr(self, "_calib_search_completed", 0) + 1
        found_count = getattr(self, "_calib_search_found", 0) + (1 if found else 0)

        self._calib_search_completed = completed
        self._calib_search_found = found_count

        # Update progress
        if hasattr(self, "progress_tracker"):
            self.progress_tracker.update(
                task_id,
                completed,
                f"Searching calibration patterns ({completed}/{total}, found: {found_count})"
            )

        # Update tqdm bar if available
        tqdm_bar = getattr(self, "_calib_search_tqdm", None)
        if tqdm_bar:
            tqdm_bar.update(1)
            tqdm_bar.set_postfix_str(f"found: {found_count}")

        # Finish when all done
        if completed >= total:
            if hasattr(self, "progress_tracker"):
                self.progress_tracker.finish(task_id)
            # Close tqdm bar
            if tqdm_bar:
                tqdm_bar.close()
                self._calib_search_tqdm = None
            self._safe_status_message(
                f"Calibration search complete: found {found_count} patterns in {total} images",
                5000
            )
            self._calib_search_task_id = None

    def _persist_auto_detected_corners(
        self,
        base: str,
        corners: Dict[str, Optional[List[List[float]]]],
    ) -> None:
        """Persist auto-detected corners to YAML file.

        Called when auto-search finds a chessboard pattern in an unmarked image.
        This ensures the corners are saved to disk and available for calibration.
        """
        if not hasattr(self.session, "set_corners"):
            log_debug(f"Cannot persist corners for {base}: session.set_corners not available", "CALIB_MIXIN")
            return

        # Only persist if we have actual corners
        has_corners = False
        for channel in ("lwir", "visible"):
            channel_corners = corners.get(channel)
            if channel_corners and len(channel_corners) > 0:
                has_corners = True
                break

        if not has_corners:
            log_debug(f"Skip persisting corners for {base}: no valid corners", "CALIB_MIXIN")
            return

        log_debug(f"Persisting auto-detected corners for {base}", "CALIB_MIXIN")
        self.session.set_corners(base, corners)

    def _handle_calibration_failed(self, base: str, message: str) -> None:
        self.statusBar().showMessage(f"Calibration analysis failed for {base}: {message}", 6000)
        # Update progress even on failure
        self._update_calibration_search_progress(found=False)

    def _sanitize_calibration_payload(
        self,
        base: str,
        results: Dict[str, Optional[bool]],
        corners: Dict[str, Optional[List[List[float]]]],
    ) -> None:
        # Suspect heuristic removed; keep payload as-is.
        return

    def _collect_calibration_samples(self) -> List[CalibrationSample]:
        samples: List[CalibrationSample] = []
        loader = self.session.loader
        if not loader:
            return samples
        out_intrinsic = self.state.calibration_outliers_intrinsic
        # Check preference: use subpixel corners if enabled
        use_subpixel = self.session.cache_service.get_preference("use_subpixel_corners", False)
        for base in loader.image_bases:
            if base not in self.state.calibration_marked:
                continue
            bucket = self.state.calibration_corners.get(base, {})

            # Get image_size from corners bucket (single source of truth)
            image_sizes = bucket.get("image_size", {})

            for channel in ("lwir", "visible"):
                if base in out_intrinsic.get(channel, set()):
                    continue
                # Use subpixel if enabled AND available, else original
                if use_subpixel:
                    points = bucket.get(f"{channel}_subpixel") or bucket.get(channel)
                else:
                    points = bucket.get(channel)
                if not points:
                    continue
                # Get image_size for this channel
                size = image_sizes.get(channel)
                if not size:
                    # Skip if no image_size available (need to re-run chessboard detection)
                    continue
                samples.append(
                    CalibrationSample(
                        base=base,
                        channel=channel,
                        corners=points,
                        image_size=tuple(size),  # type: ignore[arg-type]
                    )
                )
        return samples

    def _collect_extrinsic_samples(self) -> List[CalibrationExtrinsicSample]:
        samples: List[CalibrationExtrinsicSample] = []
        loader = self.session.loader
        if not loader:
            return samples
        out_extrinsic = self.state.calibration_outliers_extrinsic
        # Check preference: use subpixel corners if enabled
        use_subpixel = self.session.cache_service.get_preference("use_subpixel_corners", False)
        for base in loader.image_bases:
            if base not in self.state.calibration_marked or base in out_extrinsic:
                continue
            bucket = self.state.calibration_corners.get(base, {})
            # Use subpixel if enabled AND available, else original
            if use_subpixel:
                lwir_points = bucket.get("lwir_subpixel") or bucket.get("lwir")
                visible_points = bucket.get("visible_subpixel") or bucket.get("visible")
            else:
                lwir_points = bucket.get("lwir")
                visible_points = bucket.get("visible")
            if not lwir_points or not visible_points:
                continue
            lwir_path = loader.get_image_path(base, "lwir")
            vis_path = loader.get_image_path(base, "visible")
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
