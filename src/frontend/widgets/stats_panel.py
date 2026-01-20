"""Reusable widget to display dataset statistics."""
from __future__ import annotations

from typing import Dict, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

from frontend.widgets import style

from backend.services.viewer_state import ViewerState
from common.reasons import (
    REASON_BLURRY,
    REASON_DUPLICATE,
    REASON_MISSING_PAIR,
    REASON_MOTION,
    REASON_SYNC,
    REASON_USER,
    reason_text,
)
from common.log_utils import log_debug, is_debug_enabled


class StatsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("stats_card")
        self.setStyleSheet(style.card_style("stats_card"))

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(6)

        title = QLabel("Dataset stats:")
        title.setStyleSheet(style.heading_style())
        title.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        root.addWidget(title)

        content = QWidget(self)
        content.setObjectName("stats_panel_content")
        content.setStyleSheet(style.panel_body_style("stats_panel_content"))

        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(8, 6, 8, 8)
        content_layout.setSpacing(6)

        grid = QGridLayout()
        grid.setContentsMargins(8, 6, 8, 8)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(4)
        # Row 0: Incomplete pairs (spans full width)
        self.label_missing_pairs = QLabel("Incomplete pairs: LWIR 0 · Visible 0")
        # Row 1: Manual marks | Manual calibration | Outliers
        self.label_marked = QLabel("Manual marks: 0")
        self.label_calibration_manual = QLabel("Manual calibration: 0 (both: 0, partial: 0, none: 0)")
        self.label_outliers_manual = QLabel("Outliers: 0")
        # Row 2: Auto marks | Auto calibration | Outliers
        self.label_detected = QLabel("Auto marks: 0")
        self.label_calibration_auto = QLabel("Auto calibration: 0 (both: 0, partial: 0, none: 0)")
        self.label_outliers_auto = QLabel("Outliers: 0")
        # Row 3: Intrinsic reprojection | Stereo consistency
        self.label_intrinsic_errors = QLabel("Intrinsic reprojection: LWIR –; Visible –")
        self.label_extrinsic_errors = QLabel("Stereo consistency: –")
        for label in (
            self.label_missing_pairs,
            self.label_marked,
            self.label_calibration_manual,
            self.label_outliers_manual,
            self.label_detected,
            self.label_calibration_auto,
            self.label_outliers_auto,
            self.label_intrinsic_errors,
            self.label_extrinsic_errors,
        ):
            label.setStyleSheet("background: transparent;")
            label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
                | Qt.TextInteractionFlag.TextSelectableByKeyboard
            )
        # Row 0: Incomplete pairs (full width)
        grid.addWidget(self.label_missing_pairs, 0, 0, 1, 3)
        # Row 1: Manual row
        grid.addWidget(self.label_marked, 1, 0)
        grid.addWidget(self.label_calibration_manual, 1, 1)
        grid.addWidget(self.label_outliers_manual, 1, 2)
        # Row 2: Auto row
        grid.addWidget(self.label_detected, 2, 0)
        grid.addWidget(self.label_calibration_auto, 2, 1)
        grid.addWidget(self.label_outliers_auto, 2, 2)
        # Row 3: Errors
        grid.addWidget(self.label_intrinsic_errors, 3, 0, 1, 2)
        grid.addWidget(self.label_extrinsic_errors, 3, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)

        content_layout.addLayout(grid)
        root.addWidget(content)

    def update_from_state(
        self,
        state: ViewerState,
        total_pairs: int,
        missing_counts: Optional[Dict[str, int]] = None,
        intrinsic_errors: Optional[Dict[str, Dict[str, float]]] = None,
        extrinsic_errors: Optional[Dict[str, float]] = None,
    ) -> None:
        breakdown = self._compute_mark_breakdown(state)
        manual_marked = breakdown.get("manual_total", 0)
        manual_delete = breakdown.get("manual_delete", 0)
        manual_blurry = breakdown.get("manual_blurry", 0)
        manual_motion = breakdown.get("manual_motion", 0)
        sync_marked = breakdown.get("sync", 0)
        manual_patterns = max(0, breakdown.get("manual_patterns", 0))
        detected_blurry = breakdown.get("detected_blurry", 0)
        detected_motion = breakdown.get("detected_motion", 0)
        detected_duplicates = breakdown.get("detected_duplicates", 0)
        detected_missing = breakdown.get("detected_missing", 0)
        detected_patterns = breakdown.get("detected_patterns", 0)
        auto_total = detected_duplicates + detected_missing + detected_blurry + detected_motion + detected_patterns

        # Calibration counts
        calibration_auto = state.calibration_count_auto
        calibration_manual = state.calibration_count_manual

        # Compute per-type detection breakdown (both/partial/none) for auto vs manual
        calib_data = state.cache_data.get("calibration", {})
        auto_both = auto_partial = auto_none = 0
        manual_both = manual_partial = manual_none = 0
        for base, data in calib_data.items():
            if not isinstance(data, dict):
                continue
            # Presence in dict = marked (no explicit 'marked' check needed)
            results = data.get("results", {})
            positives = sum(1 for ch in ("lwir", "visible") if results.get(ch) is True)
            is_auto = data.get("auto", False)
            if positives >= 2:
                if is_auto:
                    auto_both += 1
                else:
                    manual_both += 1
            elif positives == 1:
                if is_auto:
                    auto_partial += 1
                else:
                    manual_partial += 1
            else:
                if is_auto:
                    auto_none += 1
                else:
                    manual_none += 1

        # Outliers per channel (for display)
        outliers_lwir = len(state.calibration_outliers_intrinsic.get("lwir", set()))
        outliers_visible = len(state.calibration_outliers_intrinsic.get("visible", set()))
        outliers_stereo = len(state.calibration_outliers_extrinsic)

        # Missing pairs
        missing_lwir = missing_counts.get("lwir", 0) if missing_counts else 0
        missing_visible = missing_counts.get("visible", 0) if missing_counts else 0

        # Update labels (using HTML for bold titles)
        self.label_missing_pairs.setText(
            f"<b>Incomplete pairs:</b> LWIR {missing_lwir} · Visible {missing_visible}"
        )
        self.label_marked.setText(
            f"<b>Manual marks:</b> {manual_marked} "
            f"({reason_text(REASON_USER)}: {manual_delete}, "
            f"{reason_text(REASON_BLURRY)}: {manual_blurry}, "
            f"{reason_text(REASON_MOTION)}: {manual_motion}, "
            f"{reason_text(REASON_SYNC)}: {sync_marked}, "
            f"Patterns: {manual_patterns})"
        )
        self.label_calibration_manual.setText(
            f"<b>Manual calibration:</b> {calibration_manual} (both: {manual_both}, partial: {manual_partial}, none: {manual_none})"
        )
        self.label_outliers_manual.setText(
            f"<b>Outliers:</b> LWIR {outliers_lwir}, Visible {outliers_visible}"
        )
        self.label_detected.setText(
            f"<b>Auto marks:</b> {auto_total} "
            f"({reason_text(REASON_DUPLICATE)}: {detected_duplicates}, "
            f"{reason_text(REASON_MISSING_PAIR)}: {detected_missing}, "
            f"{reason_text(REASON_BLURRY)}: {detected_blurry}, "
            f"{reason_text(REASON_MOTION)}: {detected_motion}, "
            f"Patterns: {detected_patterns})"
        )
        self.label_calibration_auto.setText(
            f"<b>Auto calibration:</b> {calibration_auto} (both: {auto_both}, partial: {auto_partial}, none: {auto_none})"
        )
        self.label_outliers_auto.setText(
            f"<b>Outliers:</b> Stereo {outliers_stereo}"
        )
        self._update_intrinsic_errors(intrinsic_errors)
        self._update_extrinsic_errors(extrinsic_errors)

    def update_stats(self, session, state: ViewerState) -> None:
        """Compatibility wrapper: pull missing counts/errors from session and delegate to update_from_state."""
        missing_counts = getattr(state, "missing_counts", None)
        intrinsic_errors = state.cache_data["reproj_errors"]
        extrinsic_errors = state.cache_data["extrinsic_errors"]
        total_pairs = session.total_pairs() if session else 0
        if is_debug_enabled("stats"):
            # Count auto marks from unified format
            auto_counts: Dict[str, int] = {}
            for entry in state.cache_data["marks"].values():
                if isinstance(entry, dict) and entry.get("auto"):
                    reason = entry.get("reason", "")
                    auto_counts[reason] = auto_counts.get(reason, 0) + 1
            log_debug(
                "stats:update"
                f" total_pairs={total_pairs}"
                f" marks={state.cache_data['reason_counts']}"
                f" auto={auto_counts}",
                "STATS",
            )
        self.update_from_state(state, total_pairs, missing_counts, intrinsic_errors, extrinsic_errors)

    def _compute_mark_breakdown(self, state: ViewerState) -> Dict[str, int]:
        if hasattr(state, "breakdown_marks"):
            return state.breakdown_marks()
        # Fallback (should not be hit)
        return {}

    def reset(self) -> None:
        self.label_missing_pairs.setText("<b>Incomplete pairs:</b> LWIR 0 · Visible 0")
        self.label_marked.setText("<b>Manual marks:</b> 0")
        self.label_calibration_manual.setText("<b>Manual calibration:</b> 0 (both: 0, partial: 0, none: 0)")
        self.label_outliers_manual.setText("<b>Outliers:</b> LWIR 0, Visible 0")
        self.label_detected.setText("<b>Auto marks:</b> 0")
        self.label_calibration_auto.setText("<b>Auto calibration:</b> 0 (both: 0, partial: 0, none: 0)")
        self.label_outliers_auto.setText("<b>Outliers:</b> Stereo 0")
        self.label_intrinsic_errors.setText("<b>Intrinsic reprojection:</b> LWIR –; Visible –")
        self.label_extrinsic_errors.setText("<b>Stereo consistency:</b> –")

    def _update_intrinsic_errors(self, errors: Optional[Dict[str, Dict[str, float]]]) -> None:
        if not errors:
            self.label_intrinsic_errors.setText("<b>Intrinsic reprojection:</b> LWIR –; Visible –")
            return
        parts = []
        for channel in ("lwir", "visible"):
            values = errors.get(channel, {}) if isinstance(errors, dict) else {}
            if values:
                worst_base, worst_err = max(values.items(), key=lambda item: item[1])
                parts.append(f"{channel.upper()} max: {worst_err:.2f} px ({worst_base})")
            else:
                parts.append(f"{channel.upper()}: –")
        self.label_intrinsic_errors.setText("<b>Intrinsic reprojection:</b> " + "; ".join(parts))

    def _update_extrinsic_errors(self, errors: Optional[Dict[str, float]]) -> None:
        if not errors:
            self.label_extrinsic_errors.setText("<b>Stereo consistency:</b> –")
            return
        worst_base, worst_err = max(errors.items(), key=lambda item: item[1])
        self.label_extrinsic_errors.setText(
            f"<b>Stereo consistency:</b> max dT {worst_err:.2f} (squares) at {worst_base}"
        )
