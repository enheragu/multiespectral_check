"""Dialog to review calibration errors and mark outliers."""
from __future__ import annotations

from statistics import mean, median
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from backend.services.progress_tracker import ProgressSnapshot
from frontend.widgets import style
from frontend.widgets.progress_panel import ProgressPanel


# Table color config (adjust here if you want a different palette)
DEFAULT_BG = QColor(232, 235, 239)  # neutral soft gray for missing/metadata cells
VALID_BG = QColor(252, 253, 254)    # near-white for valid numeric values
OUTLIER_BG = QColor(220, 53, 69)    # red for outliers
TEXT_DEFAULT = QColor(36, 40, 44)   # dark gray text
TEXT_INVERT = Qt.GlobalColor.white

# Sentinel value for non-numeric items (always sorted last)
_NON_NUMERIC_SENTINEL = float("inf")


class NumericTableItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically using UserRole data.

    Non-numeric items (no UserRole set or inf) are always placed at the end,
    regardless of ascending/descending sort order.

    We achieve this by storing float('inf') for non-numeric values in UserRole.
    This way they naturally sort to the end in ascending order.
    For descending order, we need to detect the sort order and adjust.
    """

    def __lt__(self, other: QTableWidgetItem) -> bool:
        self_val = self.data(Qt.ItemDataRole.UserRole)
        other_val = other.data(Qt.ItemDataRole.UserRole)

        # Get numeric values - None or inf means "invalid/no data"
        self_num = float(self_val) if isinstance(self_val, (int, float)) else float("inf")
        other_num = float(other_val) if isinstance(other_val, (int, float)) else float("inf")

        self_is_invalid = self_num == float("inf")
        other_is_invalid = other_num == float("inf")

        # Check if we're sorting descending by looking at header
        table = self.tableWidget()
        descending = False
        if table:
            header = table.horizontalHeader()
            if header:
                sort_col = header.sortIndicatorSection()
                if sort_col == self.column():
                    descending = header.sortIndicatorOrder() == Qt.SortOrder.DescendingOrder

        # Invalid values always go to the end regardless of sort direction
        if self_is_invalid and other_is_invalid:
            # Both invalid: compare by text
            return (self.text() or "") < (other.text() or "")
        if self_is_invalid:
            # self is invalid, other is valid
            # In ascending: invalid goes last (return False = self > other)
            # In descending: invalid still goes last (return True = self < other, but Qt inverts it)
            return descending
        if other_is_invalid:
            # other is invalid, self is valid
            return not descending

        # Both valid: normal numeric comparison
        return self_num < other_num


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
        refresh_intrinsic_callback: Optional[Callable[[Dict[str, Set[str]], Dict[str, Set[str]]], None]] = None,
        refresh_extrinsic_callback: Optional[Callable[[Dict[str, Set[str]], Dict[str, Set[str]]], None]] = None,
        calibration_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calibration outliers")
        self._rows = rows  # full set
        self._display_rows: List[Dict[str, Optional[float]]] = []
        self._row_lookup: Dict[str, Dict[str, Optional[float]]] = {}
        self._include_state: Dict[str, Dict[str, bool]] = {}
        self._updating_table = False
        self._refresh_intrinsic_callback = refresh_intrinsic_callback
        self._refresh_extrinsic_callback = refresh_extrinsic_callback
        self._calibration_info = calibration_info or {}
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
        table_title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        card_layout.addWidget(table_title)

        hint = QLabel(
            "Per-channel reprojection error (px). Uncheck LWIR/Visible/Stereo columns to exclude that view from solves."
        )
        hint.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        card_layout.addWidget(hint)

        self.table = QTableWidget(len(rows), 8, card)
        self.table.setHorizontalHeaderLabels(
            ["#", "Image", "LWIR", "Visible", "Stereo", "Include LWIR", "Include Visible", "Include Stereo"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        # Fail fast: check header exists
        v_header = self.table.verticalHeader()
        h_header = self.table.horizontalHeader()
        if v_header is not None:
            v_header.setVisible(False)
        if h_header is not None:
            h_header.setStretchLastSection(True)
            # Set minimum widths for columns to fit header text
            h_header.setMinimumSectionSize(40)
            # Resize mode: fit content but allow user resize
            h_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

        self.table.setAlternatingRowColors(False)
        self.table.setSortingEnabled(True)
        card_layout.addWidget(self.table)

        # Add progress panel (hidden by default, shown during calibration)
        self.progress_panel = ProgressPanel(card)
        self.progress_panel.hide()
        card_layout.addWidget(self.progress_panel)

        self.stats_title = QLabel("Reprojection error summary:")
        self.stats_title.setStyleSheet(style.heading_style())
        self.stats_title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        card_layout.addWidget(self.stats_title)
        stats_box = QWidget()
        stats_box.setObjectName("outliers_stats_panel")
        stats_box.setStyleSheet(style.panel_body_style("outliers_stats_panel"))
        stats_layout = QVBoxLayout(stats_box)
        stats_layout.setContentsMargins(8, 6, 8, 8)
        stats_layout.setSpacing(6)
        self.stats_label = QLabel()
        self.stats_label.setWordWrap(True)
        self.stats_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        stats_layout.addWidget(self.stats_label)

        # Add calibration info note if available
        if self._calibration_info:
            self.calib_info_label = QLabel()
            self.calib_info_label.setWordWrap(True)
            self.calib_info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
            self.calib_info_label.setStyleSheet("color: #666; font-size: 11px; margin-top: 6px;")
            self._update_calibration_info_label()
            stats_layout.addWidget(self.calib_info_label)
        else:
            self.calib_info_label = None

        card_layout.addWidget(stats_box)

        self._populate_table(rows)
        self.table.itemChanged.connect(self._handle_item_changed)

        buttons_layout = QHBoxLayout()
        self.refresh_intrinsic_btn: Optional[QPushButton] = None
        self.refresh_extrinsic_btn: Optional[QPushButton] = None
        if self._refresh_intrinsic_callback is not None:
            self.refresh_intrinsic_btn = QPushButton("Rerun intrinsics")
            self.refresh_intrinsic_btn.setFixedHeight(style.BUTTON_HEIGHT)
            self.refresh_intrinsic_btn.clicked.connect(self._handle_refresh_intrinsic)
            buttons_layout.addWidget(self.refresh_intrinsic_btn)
        if self._refresh_extrinsic_callback is not None:
            self.refresh_extrinsic_btn = QPushButton("Rerun extrinsics")
            self.refresh_extrinsic_btn.setFixedHeight(style.BUTTON_HEIGHT)
            self.refresh_extrinsic_btn.clicked.connect(self._handle_refresh_extrinsic)
            buttons_layout.addWidget(self.refresh_extrinsic_btn)
        add_all_btn = QPushButton("Add all")
        add_all_btn.setFixedHeight(style.BUTTON_HEIGHT)
        add_all_btn.clicked.connect(self.include_all)
        buttons_layout.addWidget(add_all_btn)
        self.auto_btn = QPushButton("Auto-outliers")
        self.auto_btn.setFixedHeight(style.BUTTON_HEIGHT)
        self.auto_btn.clicked.connect(self.apply_auto_outliers)
        buttons_layout.addWidget(self.auto_btn)
        buttons_layout.addStretch(1)
        card_layout.addLayout(buttons_layout)

        layout.addWidget(card)

    def _update_calibration_info_label(self) -> None:
        """Update the calibration info label with source file information."""
        if not self.calib_info_label or not self._calibration_info:
            return
        parts: List[str] = []
        intrinsic_path = self._calibration_info.get("intrinsic_path")
        intrinsic_date = self._calibration_info.get("intrinsic_date")
        extrinsic_path = self._calibration_info.get("extrinsic_path")
        extrinsic_date = self._calibration_info.get("extrinsic_date")

        if intrinsic_path:
            intrinsic_note = f"Intrinsic: {intrinsic_path}"
            if intrinsic_date:
                intrinsic_note += f" ({intrinsic_date})"
            parts.append(intrinsic_note)
        if extrinsic_path:
            extrinsic_note = f"Extrinsic: {extrinsic_path}"
            if extrinsic_date:
                extrinsic_note += f" ({extrinsic_date})"
            parts.append(extrinsic_note)

        if parts:
            self.calib_info_label.setText("\n".join(parts))
        else:
            self.calib_info_label.setText("No calibration files found")

    def _row_score(self, row: Dict[str, Optional[float]]) -> Optional[float]:
        candidates = [row.get("lwir"), row.get("visible"), row.get("stereo")]
        numeric = [float(v) for v in candidates if isinstance(v, (int, float))]
        return max(numeric) if numeric else None

    def _populate_table(self, rows: List[Dict[str, Optional[float]]]) -> None:
        self._rows = rows
        self._display_rows = list(rows)
        # Asegurar que las keys sean str
        self._row_lookup = {str(r.get("base", f"row-{idx}")): r for idx, r in enumerate(rows)}
        self._updating_table = True
        prev_sort = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            index_item = QTableWidgetItem(str(row_idx + 1))
            index_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_idx, 0, index_item)
            base = str(row.get("base", ""))
            base_item = QTableWidgetItem(base)
            self.table.setItem(row_idx, 1, base_item)
            for col, key, detect_key in (
                (2, "lwir", "detect_lwir"),
                (3, "visible", "detect_visible"),
                (4, "stereo", None),
            ):
                value = row.get(key)
                detect_val = row.get(detect_key) if detect_key else None
                detect_flag = bool(detect_val) if detect_val is not None else None
                text = self._format_error_value(value, detect_flag, key)
                # Use NumericTableItem for proper numeric sorting
                item = NumericTableItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if isinstance(value, (int, float)) and value is not None:
                    item.setData(Qt.ItemDataRole.UserRole, float(value))
                self.table.setItem(row_idx, col, item)
            default_includes = {
                "lwir": bool(row.get("include_lwir", True)),
                "visible": bool(row.get("include_visible", True)),
                "stereo": bool(row.get("include_stereo", True)),
            }
            # Asegurar que base es str para el dict
            base_str = str(base)
            current_state = self._include_state.get(base_str, default_includes)
            self._include_state[base_str] = current_state
            for col, channel in ((5, "lwir"), (6, "visible"), (7, "stereo")):
                include_item = QTableWidgetItem()
                include_item.setFlags(include_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                include_item.setCheckState(
                    Qt.CheckState.Checked if current_state.get(channel, True) else Qt.CheckState.Unchecked
                )
                self.table.setItem(row_idx, col, include_item)
        self.table.resizeColumnsToContents()
        # Ensure minimum widths for header text visibility
        h_header = self.table.horizontalHeader()
        if h_header:
            for col in range(self.table.columnCount()):
                header_text = self.table.horizontalHeaderItem(col)
                if header_text:
                    # Measure header text width + padding
                    fm = self.table.fontMetrics()
                    text_width = fm.horizontalAdvance(header_text.text()) + 20
                    if h_header.sectionSize(col) < text_width:
                        h_header.resizeSection(col, text_width)
        self.table.setSortingEnabled(prev_sort)
        self._updating_table = False
        self._update_stats_and_styles()

    def _format_error_value(self, value: Optional[float], detected: Optional[bool], key: str) -> str:
        if isinstance(value, (int, float)):
            return f"{value:.3f}"
        if key == "stereo":
            # No stereo error = either no pair or calibration not yet computed
            if detected is True:
                return "Not computed"  # Has detection but no error yet
            return "No pair"
        # For LWIR/Visible: check detection status
        if detected is True:
            return "Not computed"  # Chessboard detected but calibration not yet run
        if detected is False:
            return "No chessboard"
        return "—"  # Unknown state

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
        for row in self._display_rows:
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
        thresholds["lwir"] = max(thresholds.get("lwir", 0.0), 0.5)
        thresholds["visible"] = max(thresholds.get("visible", 0.0), 0.5)
        overall_threshold = self._compute_threshold(scores)[2] if scores else 0.0
        self._thresholds = thresholds
        self._apply_threshold_styles(thresholds)
        self._update_stats_label(lwir_vals, vis_vals, stereo_vals, thresholds, overall_threshold)

    def _apply_threshold_styles(self, thresholds: Dict[str, float]) -> None:
        default_bg = DEFAULT_BG
        default_fg = TEXT_DEFAULT
        for row_idx in range(self.table.rowCount()):
            base_item = self.table.item(row_idx, 1)
            row = self._row_lookup.get(base_item.text() if base_item else "")
            if not row:
                continue
            for col in (0, 1, 5, 6, 7):
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
                is_outlier = is_numeric and threshold > 0 and value > threshold  # type: ignore[operator]
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
        def _max_or_dash(values: List[float]) -> str:
            return f"{max(values):.3f} px" if values else "—"
        def _min_or_dash(values: List[float]) -> str:
            return f"{min(values):.3f} px" if values else "—"

        lwir_count = len(lwir_vals)
        vis_count = len(vis_vals)
        stereo_count = len(stereo_vals)
        parts = [
            f"<b>• LWIR: </b> mean: {_mean_or_dash(lwir_vals)} (min,max): ({_min_or_dash(lwir_vals)},{_max_or_dash(lwir_vals)}) (threshold {thresholds.get('lwir', 0.0):.3f} px, n={lwir_count})",
            f"<br><b>• Visible:</b> mean: {_mean_or_dash(vis_vals)} (min,max): ({_min_or_dash(vis_vals)},{_max_or_dash(vis_vals)}) (threshold {thresholds.get('visible', 0.0):.3f} px, n={vis_count})",
            f"<br><b>• Stereo:</b> mean: {_mean_or_dash(stereo_vals)} (min,max): ({_min_or_dash(stereo_vals)},{_max_or_dash(stereo_vals)}) (threshold {thresholds.get('stereo', 0.0):.3f} px, n={stereo_count})",
        ]
        note = (
            "<br><br><b>Note:</b>Thresholds use median + 3,5·MAD when dispersion exists; otherwise 1.5×median. "
            "LWIR/Visible thresholds are floored at 0.5 px so tiny errors never flag as outliers. "
            "Red cells exceed their channel threshold and will be unchecked by Auto-outliers."
        )
        self.stats_label.setText("".join(parts) + note)

    def apply_auto_outliers(self) -> None:
        thresholds = getattr(self, "_thresholds", {})
        self._updating_table = True
        for row in self._display_rows:
            base = row.get("base", "")
            if not base:
                continue
            state = self._include_state.setdefault(base, {"lwir": True, "visible": True, "stereo": True})  # type: ignore[arg-type]
            for key in ("lwir", "visible", "stereo"):
                value = row.get(key)
                threshold = thresholds.get(key, 0.0)
                if isinstance(value, (int, float)) and threshold > 0 and value > threshold:  # type: ignore[operator]
                    state[key] = False
                elif key not in state:
                    state[key] = True

        for row_idx in range(self.table.rowCount()):
            base_item = self.table.item(row_idx, 1)
            if not base_item:
                continue
            base = base_item.text()
            state = self._include_state.get(base, {"lwir": True, "visible": True, "stereo": True})
            for col, key in ((5, "lwir"), (6, "visible"), (7, "stereo")):
                include_item = self.table.item(row_idx, col)
                if not include_item:
                    continue
                include_item.setCheckState(Qt.CheckState.Checked if state.get(key, True) else Qt.CheckState.Unchecked)
        self._updating_table = False
        self._update_stats_and_styles()

    def include_all(self) -> None:
        self._updating_table = True
        for row_idx in range(self.table.rowCount()):
            base_item = self.table.item(row_idx, 1)
            if not base_item:
                continue
            base = base_item.text()
            state = self._include_state.setdefault(base, {"lwir": True, "visible": True, "stereo": True})
            for col, key in ((5, "lwir"), (6, "visible"), (7, "stereo")):
                include_item = self.table.item(row_idx, col)
                if not include_item:
                    continue
                state[key] = True
                include_item.setCheckState(Qt.CheckState.Checked)
        self._updating_table = False
        self._update_stats_and_styles()

    def _invoke_refresh(self, callback: Optional[Callable[[Dict[str, Set[str]], Dict[str, Set[str]]], None]], button: Optional[QPushButton]) -> None:
        if not callback:
            return
        if button:
            button.setEnabled(False)
        try:
            include, exclude = self.selected_channel_sets()
            callback(include, exclude)
        finally:
            if button:
                button.setEnabled(True)

    def _handle_refresh_intrinsic(self) -> None:
        self._invoke_refresh(self._refresh_intrinsic_callback, self.refresh_intrinsic_btn)

    def _handle_refresh_extrinsic(self) -> None:
        self._invoke_refresh(self._refresh_extrinsic_callback, self.refresh_extrinsic_btn)

    def update_rows(self, rows: List[Dict[str, Optional[float]]]) -> None:
        self._populate_table(rows)

    def selected_channel_sets(self) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        include: Dict[str, Set[str]] = {"lwir": set(), "visible": set(), "stereo": set()}
        exclude: Dict[str, Set[str]] = {"lwir": set(), "visible": set(), "stereo": set()}
        for row in range(self.table.rowCount()):
            base_item = self.table.item(row, 1)
            if not base_item:
                continue
            base = base_item.text()
            for col, key in ((5, "lwir"), (6, "visible"), (7, "stereo")):
                include_item = self.table.item(row, col)
                if not include_item:
                    continue
                target = include if include_item.checkState() == Qt.CheckState.Checked else exclude
                target[key].add(base)
        return include, exclude

    def _handle_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_table or item.column() not in (5, 6, 7):
            return
        base_item = self.table.item(item.row(), 1)
        if not base_item:
            return
        channel = {5: "lwir", 6: "visible", 7: "stereo"}.get(item.column())
        if not channel:
            return
        state = self._include_state.setdefault(base_item.text(), {"lwir": True, "visible": True, "stereo": True})
        state[channel] = item.checkState() == Qt.CheckState.Checked

    # ----------------------------------------------------------------
    # Progress panel interface
    # ----------------------------------------------------------------
    def set_progress(self, snapshot: Optional[ProgressSnapshot]) -> None:
        """Update progress panel with a snapshot (or hide if None)."""
        self.progress_panel.set_snapshot(snapshot)

    def set_cancel_state(self, enabled: bool, tooltip: str = "") -> None:
        """Enable/disable the cancel button in the progress panel."""
        self.progress_panel.set_cancel_state(enabled, tooltip)

    def connect_cancel(self, slot: Callable[[], None]) -> None:
        """Connect a slot to the cancel button."""
        self.progress_panel.cancelRequested.connect(slot)