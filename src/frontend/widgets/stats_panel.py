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
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)
        self.label_marked = QLabel("Manual marks: 0")
        self.label_detected = QLabel(
            "Detected marks: 0 (Duplicates 0, Missing pair 0, Blurry 0, Motion blur 0)"
        )
        self.label_missing_pairs = QLabel("Incomplete pairs, missing: LWIR 0 · Visible 0")
        self.label_calibration = QLabel(
            "Calibration images: 0 (Both detected: 0; One detected: 0; None detected: 0; Outliers: 0)"
        )
        self.label_intrinsic_errors = QLabel("Intrinsic reprojection: LWIR –; Visible –")
        self.label_extrinsic_errors = QLabel("Stereo consistency: –")
        for label in (
            self.label_marked,
            self.label_detected,
            self.label_missing_pairs,
            self.label_calibration,
            self.label_intrinsic_errors,
            self.label_extrinsic_errors,
        ):
            label.setStyleSheet("background: transparent;")
            label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
                | Qt.TextInteractionFlag.TextSelectableByKeyboard
            )
        grid.addWidget(self.label_marked, 0, 0)
        grid.addWidget(self.label_missing_pairs, 0, 2)
        grid.addWidget(self.label_detected, 1, 0, 1, 3)
        grid.addWidget(self.label_calibration, 2, 0, 1, 3)
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
        auto_blurry = breakdown.get("auto_blurry", 0)
        manual_motion = breakdown.get("manual_motion", 0)
        auto_motion = breakdown.get("auto_motion", 0)
        sync_marked = breakdown.get("sync", 0)
        detected_blurry = breakdown.get("detected_blurry", 0)
        detected_motion = breakdown.get("detected_motion", 0)
        detected_duplicates = breakdown.get("detected_duplicates", 0)
        detected_missing = breakdown.get("detected_missing", 0)
        detected_patterns = breakdown.get("detected_patterns", 0)
        manual_patterns = max(0, breakdown.get("manual_patterns", 0))
        calibration_tagged = len(state.calibration_marked)
        detection_complete = state.cache_data["_detection_counts"].get("both", 0)
        detection_partial = state.cache_data["_detection_counts"].get("partial", 0)
        detection_missing = state.cache_data["_detection_counts"].get("missing", 0)
        outlier_count = len(
            set(state.calibration_outliers_extrinsic)
            | set(state.calibration_outliers_intrinsic.get("lwir", set()))
            | set(state.calibration_outliers_intrinsic.get("visible", set()))
        )
        self.label_marked.setText(
            "Manual marks: "
            f"{manual_marked} ("
            f"{reason_text(REASON_USER)}: {manual_delete}, "
            f"{reason_text(REASON_BLURRY)}: {manual_blurry}, "
            f"{reason_text(REASON_MOTION)}: {manual_motion}, "
            f"{reason_text(REASON_SYNC)}: {sync_marked}, "
            f"Patterns: {manual_patterns})"
        )
        auto_total = detected_duplicates + detected_missing + detected_blurry + detected_motion + detected_patterns
        self.label_detected.setText(
            "Auto marks: "
            f"{auto_total} ("
            f"{reason_text(REASON_DUPLICATE)}: {detected_duplicates}, "
            f"{reason_text(REASON_MISSING_PAIR)}: {detected_missing}, "
            f"{reason_text(REASON_BLURRY)}: {detected_blurry}, "
            f"{reason_text(REASON_MOTION)}: {detected_motion}, "
            f"Patterns: {detected_patterns})"
        )
        self.label_calibration.setText(
            (
                "Calibration images: "
                f"{calibration_tagged} (Both detected: {detection_complete}; "
                f"One detected: {detection_partial}; None detected: {detection_missing}; "
                f"Outliers: {outlier_count})"
            )
        )
        missing_lwir = missing_counts.get("lwir", 0) if missing_counts else 0
        missing_visible = missing_counts.get("visible", 0) if missing_counts else 0
        self.label_missing_pairs.setText(
            f"Incomplete pairs, missing: LWIR {missing_lwir} · Visible {missing_visible}"
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
            log_debug(
                "stats:update"
                f" total_pairs={total_pairs}"
                f" marks={state.cache_data['reason_counts']}"
                f" auto={ {k: len(v) for k, v in state.cache_data['auto_marks'].items()} }",
                "STATS",
            )
        self.update_from_state(state, total_pairs, missing_counts, intrinsic_errors, extrinsic_errors)

    def _compute_mark_breakdown(self, state: ViewerState) -> Dict[str, int]:
        if hasattr(state, "breakdown_marks"):
            return state.breakdown_marks()
        # Fallback (should not be hit)
        return {}

    def reset(self) -> None:
        self.label_detected.setText(
            "Auto marks: 0 (Duplicates 0, Missing pair 0, Blurry 0, Motion blur 0)"
        )
        self.label_missing_pairs.setText("Incomplete pairs, missing: LWIR 0 · Visible 0")
        self.label_calibration.setText(
            "Calibration images: 0 (Both detected: 0; One detected: 0; None detected: 0; Outliers: 0)"
        )
        self.label_intrinsic_errors.setText("Intrinsic reprojection: LWIR –; Visible –")
        self.label_extrinsic_errors.setText("Stereo consistency: –")

    def _update_intrinsic_errors(self, errors: Optional[Dict[str, Dict[str, float]]]) -> None:
        if not errors:
            self.label_intrinsic_errors.setText("Intrinsic reprojection: LWIR –; Visible –")
            return
        parts = []
        for channel in ("lwir", "visible"):
            values = errors.get(channel, {}) if isinstance(errors, dict) else {}
            if values:
                worst_base, worst_err = max(values.items(), key=lambda item: item[1])
                parts.append(f"{channel.upper()} max: {worst_err:.2f} px ({worst_base})")
            else:
                parts.append(f"{channel.upper()}: –")
        self.label_intrinsic_errors.setText("Intrinsic reprojection: " + "; ".join(parts))

    def _update_extrinsic_errors(self, errors: Optional[Dict[str, float]]) -> None:
        if not errors:
            self.label_extrinsic_errors.setText("Stereo consistency: –")
            return
        worst_base, worst_err = max(errors.items(), key=lambda item: item[1])
        self.label_extrinsic_errors.setText(
            f"Stereo consistency: max dT {worst_err:.2f} (squares) at {worst_base}"
        )
