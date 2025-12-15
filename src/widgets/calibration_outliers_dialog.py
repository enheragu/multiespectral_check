"""Dialog to review calibration errors and mark outliers."""
from __future__ import annotations

from statistics import mean, median
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from widgets import style


# Table color config (adjust here if you want a different palette)
DEFAULT_BG = QColor(232, 235, 239)  # neutral soft gray for missing/metadata cells
VALID_BG = QColor(252, 253, 254)    # near-white for valid numeric values
OUTLIER_BG = QColor(220, 53, 69)    # red for outliers
TEXT_DEFAULT = QColor(36, 40, 44)   # dark gray text
TEXT_INVERT = Qt.GlobalColor.white


def _safe_median(values: List[float]) -> float:
    return median(values) if values else 0.0


def _mad(values: List[float], center: float) -> float:
    if not values:
        return 0.0
    deviations = [abs(v - center) for v in values]
    return median(deviations) if deviations else 0.0


class CalibrationOutliersDialog(QDialog):
    def __init__(
        self,
        rows: List[Dict[str, Optional[float]]],
        parent=None,
        refresh_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calibration outliers")
        self._rows = rows
        self._refresh_callback = refresh_callback
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        card = QWidget(self)
        card.setObjectName("outliers_card")
        card.setStyleSheet(style.card_style("outliers_card") + style.table_widget_style())

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 14, 14, 14)
        card_layout.setSpacing(10)

        table_title = QLabel("Image summary (error and inclusion):")
        table_title.setStyleSheet(style.heading_style())
        card_layout.addWidget(table_title)

        hint = QLabel("Reprojection error for each image (px). Uncheck to exclude from future solves.")
        card_layout.addWidget(hint)

        self.table = QTableWidget(len(rows), 6, card)
        self.table.setHorizontalHeaderLabels(["#", "Image", "LWIR", "Visible", "Stereo", "Include"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(False)
        card_layout.addWidget(self.table)
        self.stats_title = QLabel("Reprojection error summary:")
        self.stats_title.setStyleSheet(style.heading_style())
        card_layout.addWidget(self.stats_title)
        stats_box = QWidget()
        stats_box.setObjectName("outliers_stats_panel")
        stats_box.setStyleSheet(style.panel_body_style("outliers_stats_panel"))
        stats_layout = QVBoxLayout(stats_box)
        stats_layout.setContentsMargins(8, 6, 8, 8)
        stats_layout.setSpacing(6)
        self.stats_label = QLabel()
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        card_layout.addWidget(stats_box)

        self._populate_table(rows)

        buttons_layout = QHBoxLayout()
        if self._refresh_callback is not None:
            self.refresh_btn = QPushButton("Refresh calibration")
            self.refresh_btn.clicked.connect(self._handle_refresh)
            buttons_layout.addWidget(self.refresh_btn)
        self.auto_btn = QPushButton("Auto-outliers")
        self.auto_btn.clicked.connect(self.apply_auto_outliers)
        buttons_layout.addWidget(self.auto_btn)
        buttons_layout.addStretch(1)
        card_layout.addLayout(buttons_layout)

        layout.addWidget(card)


    def _row_score(self, row: Dict[str, Optional[float]]) -> Optional[float]:
        candidates = [row.get("lwir"), row.get("visible"), row.get("stereo")]
        numeric = [float(v) for v in candidates if isinstance(v, (int, float))]
        return max(numeric) if numeric else None

    def _populate_table(self, rows: List[Dict[str, Optional[float]]]) -> None:
        self._rows = rows
        self.table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            index_item = QTableWidgetItem(str(row_idx + 1))
            index_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_idx, 0, index_item)
            base_item = QTableWidgetItem(row.get("base", ""))
            self.table.setItem(row_idx, 1, base_item)
            for col, key, detect_key in (
                (2, "lwir", "detect_lwir"),
                (3, "visible", "detect_visible"),
                (4, "stereo", None),
            ):
                value = row.get(key)
                detect_flag = row.get(detect_key) if detect_key else None
                text = self._format_error_value(value, detect_flag, key)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row_idx, col, item)
            include_item = QTableWidgetItem()
            include_item.setFlags(include_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            include_item.setCheckState(Qt.CheckState.Checked if row.get("included", True) else Qt.CheckState.Unchecked)
            self.table.setItem(row_idx, 5, include_item)
        self.table.resizeColumnsToContents()
        self._update_stats_and_styles()

    def _format_error_value(self, value: Optional[float], detected: Optional[bool], key: str) -> str:
        if isinstance(value, (int, float)):
            return f"{value:.3f}"
        if key == "stereo":
            return "No pair"
        if detected is False or detected is None:
            return "No chessboard"
        return "—"

    def _compute_threshold(self, scores: List[float]) -> Tuple[float, float, float]:
        center = _safe_median(scores)
        mad_value = _mad(scores, center)
        threshold = center + 2.5 * mad_value if mad_value > 0 else center * 1.5
        return center, mad_value, threshold

    def _update_stats_and_styles(self) -> None:
        scores: List[float] = []
        lwir_vals: List[float] = []
        vis_vals: List[float] = []
        stereo_vals: List[float] = []
        for row in self._rows:
            score = self._row_score(row)
            if score is not None:
                scores.append(score)
            for bucket, key in ((lwir_vals, "lwir"), (vis_vals, "visible"), (stereo_vals, "stereo")):
                value = row.get(key)
                if isinstance(value, (int, float)):
                    bucket.append(float(value))
        thresholds = {
            "lwir": self._compute_threshold(lwir_vals)[2] if lwir_vals else 0.0,
            "visible": self._compute_threshold(vis_vals)[2] if vis_vals else 0.0,
            "stereo": self._compute_threshold(stereo_vals)[2] if stereo_vals else 0.0,
        }
        overall_threshold = self._compute_threshold(scores)[2] if scores else 0.0
        self._apply_threshold_styles(thresholds)
        self._update_stats_label(lwir_vals, vis_vals, stereo_vals, thresholds, overall_threshold)

    def _apply_threshold_styles(self, thresholds: Dict[str, float]) -> None:
        default_bg = DEFAULT_BG
        default_fg = TEXT_DEFAULT
        for row_idx, row in enumerate(self._rows):
            for col in (0, 1, 5):
                item = self.table.item(row_idx, col)
                if item:
                    item.setBackground(default_bg)
                    item.setForeground(default_fg)
            for col, key in ((2, "lwir"), (3, "visible"), (4, "stereo")):
                item = self.table.item(row_idx, col)
                if not item:
                    continue
                value = row.get(key)
                threshold = thresholds.get(key, 0.0)
                is_numeric = isinstance(value, (int, float))
                is_outlier = is_numeric and threshold > 0 and float(value) > threshold
                if not is_numeric:
                    item.setBackground(default_bg)
                    item.setForeground(default_fg)
                elif is_outlier:
                    item.setBackground(OUTLIER_BG)
                    item.setForeground(TEXT_INVERT)
                else:
                    item.setBackground(VALID_BG)
                    item.setForeground(default_fg)

    def _update_stats_label(
        self,
        lwir_vals: List[float],
        vis_vals: List[float],
        stereo_vals: List[float],
        thresholds: Dict[str, float],
        overall_threshold: float,
    ) -> None:
        def _mean_or_dash(values: List[float]) -> str:
            return f"{mean(values):.3f} px" if values else "—"

        lwir_count = len(lwir_vals)
        vis_count = len(vis_vals)
        stereo_count = len(stereo_vals)
        parts = [
            f"<b>• LWIR: </b> {_mean_or_dash(lwir_vals)} (threshold {thresholds.get('lwir', 0.0):.3f} px, n={lwir_count})",
            f"<br><b>• Visible:</b> {_mean_or_dash(vis_vals)} (threshold {thresholds.get('visible', 0.0):.3f} px, n={vis_count})",
            f"<br><b>• Stereo:</b> {_mean_or_dash(stereo_vals)} (threshold {thresholds.get('stereo', 0.0):.3f} px, n={stereo_count})",
        ]
        note = (
            "<br><br><b>Note:</b>Thresholds use median + 2.5·MAD when dispersion exists; otherwise 1.5×median. "
            "Red cells exceed their channel threshold and will be unchecked by Auto-outliers."
        )
        self.stats_label.setText("".join(parts) + note)

    def apply_auto_outliers(self) -> None:
        scores: List[Tuple[int, float]] = []
        for idx, row in enumerate(self._rows):
            score = self._row_score(row)
            if score is not None:
                scores.append((idx, score))
        score_values = [s for _, s in scores]
        _, _, threshold = self._compute_threshold(score_values) if score_values else (0.0, 0.0, 0.0)
        for idx, score in scores:
            include_item = self.table.item(idx, 5)
            if include_item is None:
                continue
            include_item.setCheckState(Qt.CheckState.Unchecked if score > threshold else Qt.CheckState.Checked)
        # Refresh styling/stats after toggling include states
        self._update_stats_and_styles()

    def _handle_refresh(self) -> None:
        if not self._refresh_callback:
            return
        self.refresh_btn.setEnabled(False)
        try:
            self._refresh_callback()
        finally:
            self.refresh_btn.setEnabled(True)

    def update_rows(self, rows: List[Dict[str, Optional[float]]]) -> None:
        self._populate_table(rows)

    def selected_bases(self) -> Tuple[List[str], List[str]]:
        include: List[str] = []
        exclude: List[str] = []
        for row in range(self.table.rowCount()):
            base_item = self.table.item(row, 1)
            include_item = self.table.item(row, 5)
            if not base_item or not include_item:
                continue
            base = base_item.text()
            if include_item.checkState() == Qt.CheckState.Checked:
                include.append(base)
            else:
                exclude.append(base)
        return include, exclude
