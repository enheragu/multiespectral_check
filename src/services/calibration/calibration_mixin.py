"""Shared calibration workflow helpers used by the main window.

Collects calibration targets, manages queuing, emits debug bundles, and keeps cached results
normalized before refreshing overlays and UI state.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtWidgets import QMessageBox

from services.calibration.calibration_extrinsic_solver import CalibrationExtrinsicSample
from services.calibration.calibration_solver import CalibrationSample
from utils.calibration import evaluate_corner_layout


class CalibrationWorkflowMixin:
    def _collect_calibration_targets(self, *, force: bool = False) -> List[str]:
        if not self.session.loader:
            return []
        pending: List[str] = []
        for base in self.session.loader.image_bases:
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
        if not self.session.loader:
            QMessageBox.information(self, "Calibration", "Load a dataset before re-running calibration.")
            return
        pending = self._collect_calibration_targets(force=True)
        if not pending:
            QMessageBox.information(self, "Calibration", "No calibration candidates require re-analysis.")
            return
        reply = QMessageBox.question(
            self,
            "Re-run calibration",
            f"Queue calibration detection for {len(pending)} tagged image(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        queued = self.calibration_controller.prefetch(pending, force=True)
        self.statusBar().showMessage(f"Queued calibration analysis for {queued} image(s)", 6000)

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
        corners: Optional[Dict[str, Optional[List[Tuple[float, float]]]]],
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
            print("\n".join(log_lines))
        if saved_path:
            status_msg = f"{status_msg} | Debug saved to {saved_path.name}"
        self.statusBar().showMessage(status_msg, 6000)

    def _schedule_calibration_job(self, base: str, *, force: bool = False, priority: bool = False) -> bool:
        if not self.session.loader or not base or base not in self.state.calibration_marked:
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
        return scheduled

    def _handle_calibration_ready(
        self,
        base: str,
        results: Dict[str, Optional[bool]],
        corners: Dict[str, Optional[List[Tuple[float, float]]]],
    ) -> None:
        if base not in self.state.calibration_marked:
            return
        normalized_results = dict(results or {})
        normalized_corners = dict(corners or {})
        self._sanitize_calibration_payload(base, normalized_results, normalized_corners)
        self.state.calibration_results[base] = normalized_results
        self.state.calibration_corners[base] = normalized_corners
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

    def _handle_calibration_failed(self, base: str, message: str) -> None:
        self.statusBar().showMessage(f"Calibration analysis failed for {base}: {message}", 6000)

    def _sanitize_calibration_payload(
        self,
        base: str,
        results: Dict[str, Optional[bool]],
        corners: Dict[str, Optional[List[Tuple[float, float]]]],
    ) -> None:
        if not results:
            self.state.calibration_warnings.pop(base, None)
            return
        warnings: Dict[str, str] = {}
        pattern_size = getattr(self.calibration_debugger, "chessboard_size", (7, 7))
        for channel, label in (("lwir", "LWIR"), ("visible", "Visible")):
            if results.get(channel) is not True:
                continue
            channel_corners = corners.get(channel)
            valid, reason = evaluate_corner_layout(channel_corners, pattern_size)
            if valid:
                continue
            warnings[channel] = reason or "Corner layout inconsistent"
            results[channel] = False
        if warnings:
            self.state.calibration_warnings[base] = warnings
            summary = ", ".join(f"{cam.upper()}: {msg}" for cam, msg in warnings.items())
            self.statusBar().showMessage(
                f"{base}: detección sospechosa descartada ({summary})",
                6000,
            )
        else:
            self.state.calibration_warnings.pop(base, None)

    def _collect_calibration_samples(self) -> List[CalibrationSample]:
        samples: List[CalibrationSample] = []
        loader = self.session.loader
        if not loader:
            return samples
        for base in loader.image_bases:
            if base not in self.state.calibration_marked or base in self.state.calibration_outliers:
                continue
            bucket = self.state.calibration_corners.get(base, {})
            warnings = self.state.calibration_warnings.get(base, {})
            for channel in ("lwir", "visible"):
                points = bucket.get(channel)
                if not points or warnings.get(channel):
                    continue
                image_path = loader.get_image_path(base, channel)
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

    def _collect_extrinsic_samples(self) -> List[CalibrationExtrinsicSample]:
        samples: List[CalibrationExtrinsicSample] = []
        loader = self.session.loader
        if not loader:
            return samples
        for base in loader.image_bases:
            if base not in self.state.calibration_marked or base in self.state.calibration_outliers:
                continue
            bucket = self.state.calibration_corners.get(base, {})
            warnings = self.state.calibration_warnings.get(base, {})
            lwir_points = bucket.get("lwir")
            visible_points = bucket.get("visible")
            if not lwir_points or not visible_points:
                continue
            if warnings.get("lwir") or warnings.get("visible"):
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
