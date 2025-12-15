"""Reusable widget to display dataset statistics."""
from __future__ import annotations

from typing import Dict, Optional

from PyQt6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

from widgets import style

from services.viewer_state import ViewerState
from utils.reasons import (
    REASON_BLURRY,
    REASON_DUPLICATE,
    REASON_MOTION,
    REASON_SYNC,
)


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
        self.label_duplicates = QLabel("Duplicates: 0")
        self.label_missing_pairs = QLabel("Incomplete pairs, missing: LWIR 0 · Visible 0")
        self.label_calibration = QLabel(
            "Calibration images: 0 (Both detected: 0; One detected: 0; None detected: 0; Suspect: 0; Outliers: 0)"
        )
        self.label_intrinsic_errors = QLabel("Intrinsic reprojection: LWIR –; Visible –")
        self.label_extrinsic_errors = QLabel("Stereo consistency: –")
        for label in (
            self.label_marked,
            self.label_duplicates,
            self.label_missing_pairs,
            self.label_calibration,
            self.label_intrinsic_errors,
            self.label_extrinsic_errors,
        ):
            label.setStyleSheet("background: transparent;")
        grid.addWidget(self.label_marked, 0, 0)
        grid.addWidget(self.label_duplicates, 0, 1)
        grid.addWidget(self.label_missing_pairs, 0, 2)
        grid.addWidget(self.label_calibration, 1, 0, 1, 3)
        grid.addWidget(self.label_intrinsic_errors, 2, 0, 1, 2)
        grid.addWidget(self.label_extrinsic_errors, 2, 2)
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
        reason_counts = state.mark_reason_counts
        total_marked = sum(reason_counts.values())
        duplicate_marked = reason_counts.get(REASON_DUPLICATE, 0)
        manual_marked = max(0, total_marked - duplicate_marked)
        blurry_marked = reason_counts.get(REASON_BLURRY, 0)
        motion_marked = reason_counts.get(REASON_MOTION, 0)
        sync_marked = reason_counts.get(REASON_SYNC, 0)
        calibration_tagged = len(state.calibration_marked)
        detection_complete = state.calibration_detection_counts.get("both", 0)
        detection_partial = state.calibration_detection_counts.get("partial", 0)
        detection_missing = state.calibration_detection_counts.get("missing", 0)
        suspect_count = len(state.calibration_suspect_bases)
        outlier_count = len(getattr(state, "calibration_outliers", set()))
        self.label_marked.setText(
            f"Manual marks: {manual_marked} (Blurry {blurry_marked}, Motion {motion_marked}, Sync {sync_marked})"
        )
        self.label_duplicates.setText(f"Duplicates: {duplicate_marked}")
        self.label_calibration.setText(
            (
                "Calibration images: "
                f"{calibration_tagged} (Both detected: {detection_complete}; "
                f"One detected: {detection_partial}; None detected: {detection_missing}; "
                f"Suspect: {suspect_count}; Outliers: {outlier_count})"
            )
        )
        missing_lwir = missing_counts.get("lwir", 0) if missing_counts else 0
        missing_visible = missing_counts.get("visible", 0) if missing_counts else 0
        self.label_missing_pairs.setText(
            f"Incomplete pairs, missing: LWIR {missing_lwir} · Visible {missing_visible}"
        )
        self._update_intrinsic_errors(intrinsic_errors)
        self._update_extrinsic_errors(extrinsic_errors)

    def reset(self) -> None:
        self.label_marked.setText("Manual marks: 0")
        self.label_duplicates.setText("Duplicates: 0")
        self.label_missing_pairs.setText("Incomplete pairs, missing: LWIR 0 · Visible 0")
        self.label_calibration.setText(
            "Calibration images: 0 (Both detected: 0; One detected: 0; None detected: 0; Suspect: 0; Outliers: 0)"
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
