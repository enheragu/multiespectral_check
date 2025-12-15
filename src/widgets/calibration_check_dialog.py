"""Simple dialog to inspect calibration matrices and residual errors."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QDialog,
    QFormLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from widgets import style


class CalibrationCheckDialog(QDialog):
    def __init__(
        self,
        parent,
        matrices: Optional[Dict[str, dict]] = None,
        extrinsic: Optional[Dict[str, dict]] = None,
        results_path: Optional[Path] = None,
    ) -> None:
        super().__init__(parent)
        self.matrices = matrices or {}
        self.extrinsic = extrinsic or {}
        self.results_path = results_path
        self.file_metadata = self._load_file_metadata()
        self.setWindowTitle("Calibration report")
        self.setMinimumWidth(420)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(0)

        card = QWidget(self)
        card.setObjectName("calib_report_card")
        card.setStyleSheet(style.card_style("calib_report_card"))

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 14, 14, 14)
        card_layout.setSpacing(12)

        header = QLabel("Latest computed calibration matrices and reprojection errors.")
        header.setWordWrap(True)
        # header.setStyleSheet("font-weight: 700; font-size: 13px; color: #0f1115;")
        card_layout.addWidget(header)

        metadata_group = self._file_info_group()
        if metadata_group:
            card_layout.addWidget(self._section_heading("Calibration file:"))
            card_layout.addWidget(metadata_group)

        card_layout.addWidget(self._section_heading("Intrinsic calibration:"))
        card_layout.addWidget(self._intrinsic_group())
        card_layout.addWidget(self._section_heading("Extrinsic calibration:"))
        card_layout.addWidget(self._extrinsic_group())

        card_layout.addStretch(1)

        if self.results_path and self.results_path.exists():
            open_button = QPushButton("Open calibration file")
            open_button.setFixedHeight(style.BUTTON_HEIGHT)
            open_button.clicked.connect(self._open_results_file)
            card_layout.addWidget(open_button, alignment=Qt.AlignmentFlag.AlignRight)

        layout.addWidget(card)

    def _section_heading(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(style.heading_style())
        return label

    def _file_info_group(self) -> Optional[QWidget]:
        if not self.file_metadata:
            return None
        panel = QWidget()
        panel.setObjectName("file_info_panel")
        panel.setStyleSheet(style.panel_body_style("file_info_panel"))
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        file_name = QLabel(self.file_metadata.get("file_name", "—"))
        file_name.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._field_label("File"), file_name)
        file_path = QLabel(self.file_metadata.get("file_path", "—"))
        file_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._field_label("Location"), file_path)
        timestamp = QLabel(self._format_timestamp(self.file_metadata.get("updated_at")))
        timestamp.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._field_label("Computed"), timestamp)
        pattern = QLabel(self._format_pattern_label(self.file_metadata.get("pattern_size")))
        pattern.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._field_label("Pattern"), pattern)
        square = QLabel(self._format_square_label(self.file_metadata.get("square_size")))
        square.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._field_label("Square side"), square)
        layout.addLayout(form)
        return panel

    def _intrinsic_group(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("intrinsic_panel")
        panel.setStyleSheet(style.panel_body_style("intrinsic_panel"))
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.addWidget(self._channel_block("visible", "Visible camera"), 0, 0)
        grid.addWidget(self._channel_block("lwir", "LWIR camera"), 0, 1)
        layout.addLayout(grid)
        return panel

    def _extrinsic_group(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("extrinsic_panel")
        panel.setStyleSheet(style.panel_body_style("extrinsic_panel"))
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        payload = self.extrinsic or {}
        translation = payload.get("translation")
        rotation = payload.get("rotation")
        if not translation or not rotation:
            message = QLabel("Not available. Compute the stereo extrinsic transform first.")
            message.setWordWrap(True)
            form.addRow(message)
            layout.addLayout(form)
            return panel

        samples = payload.get("samples", 0)
        rms = payload.get("reprojection_error")
        summary = f"Samples: {samples}"
        if rms is not None:
            summary += f" | RMS error: {rms:.4f}"
        form.addRow(self._field_label("Summary"), QLabel(summary))

        baseline = payload.get("baseline")
        baseline_label = QLabel(f"{baseline:.4f}" if baseline is not None else "—")
        form.addRow(self._field_label("Baseline"), baseline_label)

        updated = payload.get("updated_at")
        updated_label = QLabel(self._format_timestamp(updated))
        updated_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._field_label("Computed"), updated_label)

        translation_label = QLabel(self._format_vector(translation))
        translation_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._field_label("Translation"), translation_label)

        rotation_label = QLabel(self._format_matrix(rotation))
        rotation_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._field_label("Rotation matrix"), rotation_label)

        per_pair = payload.get("per_pair_errors") or []
        per_pair_label = QLabel(self._format_per_pair_errors(per_pair))
        per_pair_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        per_pair_label.setWordWrap(True)
        form.addRow(self._field_label("Per-pair trans | rot (deg)"), per_pair_label)
        layout.addLayout(form)
        return panel

    def _channel_block(self, key: str, title: str) -> QWidget:
        panel = QWidget()
        panel.setObjectName(f"channel_panel_{key}")
        panel.setStyleSheet(style.panel_body_style(panel.objectName()))
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        heading = QLabel(f"{title}:")
        heading.setStyleSheet(style.heading_style())
        layout.addWidget(heading)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        payload = self.matrices.get(key)
        if not payload or not payload.get("camera_matrix"):
            message = QLabel("Not available. Compute calibration first.")
            message.setWordWrap(True)
            form.addRow(message)
            layout.addLayout(form)
            return panel
        samples = payload.get("samples", 0)
        error = payload.get("reprojection_error")
        header = QLabel(
            f"Samples: {samples} | RMS error: {error:.4f}" if error is not None else f"Samples: {samples}"
        )
        form.addRow(self._field_label("Summary"), header)
        matrix = QLabel(self._format_matrix(payload.get("camera_matrix")))
        matrix.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._field_label("Camera matrix"), matrix)
        distortion = QLabel(self._format_vector(payload.get("distortion")))
        distortion.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._field_label("Distortion"), distortion)
        layout.addLayout(form)
        return panel

    def _field_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("font-weight: 700; color: #111;")
        return label

    def _format_matrix(self, matrix) -> str:
        if not matrix:
            return "—"
        return "\n".join(
            "[ " + ", ".join(f"{value:.4f}" for value in row) + " ]" for row in matrix
        )

    def _format_vector(self, vector) -> str:
        if not vector:
            return "—"
        return "[ " + ", ".join(f"{value:.5f}" for value in vector) + " ]"

    def _format_per_pair_errors(self, rows) -> str:
        if not rows:
            return "—"
        lines = []
        for row in rows:
            base = row.get("base", "?")
            trans = row.get("translation_error")
            rot = row.get("rotation_error_deg")
            if trans is None or rot is None:
                continue
            lines.append(f"{base}: {trans:.4f} | {rot:.2f} deg")
        return "\n".join(lines) if lines else "—"

    def _open_results_file(self) -> None:
        if not self.results_path:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.results_path).resolve())))

    def _load_file_metadata(self) -> Dict[str, Any]:
        if not self.results_path or not self.results_path.exists():
            return {}
        try:
            with open(self.results_path, "r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
        except Exception:  # noqa: BLE001
            return {}
        metadata = {
            "file_name": self.results_path.name,
            "file_path": str(self.results_path),
            "pattern_size": payload.get("pattern_size"),
            "square_size": payload.get("square_size") or payload.get("square_length"),
            "updated_at": payload.get("updated_at"),
        }
        return metadata

    def _format_timestamp(self, raw: Optional[str]) -> str:
        if not raw or not isinstance(raw, str):
            return "Not specified"
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            local_dt = parsed.astimezone()
            return local_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:  # noqa: BLE001
            return raw

    def _format_pattern_label(self, pattern: Optional[Any]) -> str:
        if isinstance(pattern, (list, tuple)) and len(pattern) == 2:
            cols, rows = pattern
            return f"{cols} × {rows} corners"
        return "Not specified"

    def _format_square_label(self, value: Optional[Any]) -> str:
        if isinstance(value, dict):
            magnitude = value.get("value")
            unit = value.get("unit", "units")
            if isinstance(magnitude, (int, float)):
                return f"{magnitude:.3f} {unit}"
        if isinstance(value, (int, float)):
            return f"{value:.3f} units"
        return "Not specified"
