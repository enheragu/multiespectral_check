"""Simple dialog to inspect calibration matrices and residual errors."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from frontend.widgets import style


class CalibrationCheckDialog(QDialog):
    def __init__(
        self,
        parent,
        matrices: Optional[Dict[str, dict]] = None,
        extrinsic: Optional[Dict[str, dict]] = None,
        intrinsic_path: Optional[Path] = None,
        extrinsic_path: Optional[Path] = None,
        dataset_paths: Optional[List[str]] = None,
    ) -> None:
        super().__init__(parent)
        self.matrices = matrices or {}
        self.extrinsic = extrinsic or {}
        self.intrinsic_path = intrinsic_path
        self.extrinsic_path = extrinsic_path
        self.dataset_paths = dataset_paths or []
        self.file_metadata = self._load_file_metadata()
        self.setWindowTitle("Calibration report")
        self.setMinimumWidth(720)
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
        header.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        # header.setStyleSheet("font-weight: 700; font-size: 13px; color: #0f1115;")
        card_layout.addWidget(header)

        metadata_group = self._file_info_group()
        if metadata_group:
            card_layout.addWidget(self._section_heading("Calibration information:"))
            card_layout.addWidget(metadata_group)

        card_layout.addWidget(self._section_heading("Intrinsic calibration:"))
        card_layout.addWidget(self._intrinsic_group())
        card_layout.addWidget(self._section_heading("Extrinsic calibration:"))
        card_layout.addWidget(self._extrinsic_group())

        card_layout.addStretch(1)

        # Add button to open calibration files (both if available)
        has_intrinsic = self.intrinsic_path and self.intrinsic_path.exists()
        has_extrinsic = self.extrinsic_path and self.extrinsic_path.exists()
        if has_intrinsic or has_extrinsic:
            open_button = QPushButton("Open calibration files")
            open_button.setFixedHeight(style.BUTTON_HEIGHT)
            open_button.clicked.connect(self._open_calibration_files)
            card_layout.addWidget(open_button, alignment=Qt.AlignmentFlag.AlignRight)

        scroll = QScrollArea(self)
        scroll.setWidget(card)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        layout.addWidget(scroll)
        self.resize(max(self.minimumWidth(), 900), 720)

    def refresh_data(
        self,
        matrices: Optional[Dict[str, dict]],
        extrinsic: Optional[Dict[str, dict]],
        intrinsic_path: Optional[Path],
        extrinsic_path: Optional[Path] = None,
        dataset_paths: Optional[List[str]] = None,
    ) -> None:
        self.matrices = matrices or {}
        self.extrinsic = extrinsic or {}
        self.intrinsic_path = intrinsic_path
        self.extrinsic_path = extrinsic_path
        self.dataset_paths = dataset_paths or []
        self.file_metadata = self._load_file_metadata()
        self._rebuild_ui()

    def _rebuild_ui(self) -> None:
        layout = self.layout()
        if layout:
            while layout.count():
                item = layout.takeAt(0)
                if item is None:
                    break
                widget = item.widget()
                if widget:
                    widget.deleteLater()
        self._build_ui()

    def _section_heading(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(style.heading_style())
        return label

    def _file_info_group(self) -> Optional[QWidget]:
        # Show info if we have any metadata or dataset paths
        if not self.file_metadata and not self.dataset_paths:
            return None
        panel = QWidget()
        panel.setObjectName("file_info_panel")
        panel.setStyleSheet(style.panel_body_style("file_info_panel"))
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)

        # Intrinsic file
        intrinsic_name = self.file_metadata.get("intrinsic_file", "—")
        intrinsic_label = QLabel(intrinsic_name)
        intrinsic_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._field_label("Intrinsic file"), intrinsic_label)

        # Extrinsic file
        extrinsic_name = self.file_metadata.get("extrinsic_file", "—")
        extrinsic_label = QLabel(extrinsic_name)
        extrinsic_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._field_label("Extrinsic file"), extrinsic_label)

        # Location (parent directory)
        file_path = self.file_metadata.get("file_path", "—")
        path_label = QLabel(file_path)
        path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._field_label("Location"), path_label)

        # Timestamp
        timestamp = QLabel(self._format_timestamp(self.file_metadata.get("updated_at")))
        timestamp.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._field_label("Computed"), timestamp)

        # Pattern and square
        pattern = QLabel(self._format_pattern_label(self.file_metadata.get("pattern_size")))
        pattern.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._field_label("Pattern"), pattern)
        square = QLabel(self._format_square_label(self.file_metadata.get("square_size")))
        square.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._field_label("Square side"), square)

        # Datasets used (now inside calibration info section)
        if self.dataset_paths:
            datasets_text = ", ".join(Path(p).name for p in self.dataset_paths)
            datasets_label = QLabel(datasets_text)
            datasets_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            datasets_label.setWordWrap(True)
            form.addRow(self._field_label("Datasets used"), datasets_label)

        layout.addLayout(form)
        return panel

    def _dataset_list_group(self) -> Optional[QWidget]:
        if not self.dataset_paths:
            return None
        panel = QWidget()
        panel.setObjectName("dataset_list_panel")
        panel.setStyleSheet(style.panel_body_style("dataset_list_panel"))
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        for dataset in self.dataset_paths:
            label = QLabel(dataset)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(label)
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
        layout.setSpacing(8)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(8)
        payload = self.extrinsic or {}
        translation = payload.get("translation")
        rotation = payload.get("rotation")
        if not translation or not rotation:
            message = QLabel("Not available. Compute the stereo extrinsic transform first.")
            message.setWordWrap(True)
            message.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            form.addRow(message)
            layout.addLayout(form)
            return panel

        frame_note = QLabel(
            "Extrinsic transform maps points from the LWIR camera frame into the visible camera frame. "
            "Units follow the chessboard square size used during calibration."
        )
        frame_note.setWordWrap(True)
        frame_note.setStyleSheet("color: #444;")
        frame_note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._field_label("Frame note"), frame_note)

        samples = payload.get("samples", 0)
        rms = payload.get("reprojection_error")
        summary = f"Samples: {samples}"
        if rms is not None:
            summary += f" | RMS error: {rms:.4f}"
        summary_label = QLabel(summary)
        summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._field_label("Summary"), summary_label)

        baseline = payload.get("baseline")
        baseline_label = QLabel(f"{baseline:.4f}" if baseline is not None else "—")
        baseline_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._field_label("Baseline"), baseline_label)

        updated = payload.get("updated_at")
        updated_str = str(updated) if updated and not isinstance(updated, dict) else None
        updated_label = QLabel(self._format_timestamp(updated_str))
        updated_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._field_label("Computed"), updated_label)

        translation_label = QLabel(self._format_vector(translation))
        translation_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        translation_label.setStyleSheet("font-family: monospace;")
        form.addRow(self._field_label("Translation (LWIR → Visible)"), translation_label)

        rotation_label = QLabel(self._format_matrix(rotation))
        rotation_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rotation_label.setWordWrap(True)
        rotation_label.setStyleSheet("font-family: monospace; padding: 2px 0 8px 0;")
        form.addRow(self._field_label("Rotation (LWIR → Visible)"), rotation_label)

        ## Too much information to show in the dialog
        # per_pair = payload.get("per_pair_errors")
        # per_pair_list: List[Any] = per_pair if isinstance(per_pair, list) else []
        # per_pair_label = QLabel(self._format_per_pair_errors(per_pair_list))
        # per_pair_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        # per_pair_label.setWordWrap(True)
        # form.addRow(self._field_label("Per-pair Δtrans | Δrot (deg)"), per_pair_label)
        # per_pair_help = QLabel(
            # "Per-pair deltas compare each calibration image pair against the solved extrinsic: "
            # "translation error is |t_pair - T| (in chessboard units) and rotation error is the angle between R_pair and R."
        # )
        # per_pair_help.setWordWrap(True)
        # per_pair_help.setStyleSheet("color: #444;")
        # per_pair_help.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        # form.addRow(self._field_label("Per-pair note"), per_pair_help)
        layout.addLayout(form)
        return panel

    def _channel_block(self, key: str, title: str) -> QWidget:
        panel = QWidget()
        panel.setObjectName(f"channel_panel_{key}")
        panel.setStyleSheet(style.panel_body_style(panel.objectName()))
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        heading = QLabel(f"{title}:")
        heading.setStyleSheet(style.heading_style())
        layout.addWidget(heading)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(6)
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
        matrix.setWordWrap(True)
        matrix.setStyleSheet("font-family: monospace; padding: 2px 0 8px 0;")
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
            return "— (no per-pair estimates were produced by the solver)"
        lines = []
        for row in rows:
            base = row.get("base", "?")
            trans = row.get("translation_error")
            rot = row.get("rotation_error_deg")
            if trans is None or rot is None:
                continue
            lines.append(f"{base}: {trans:.4f} | {rot:.2f} deg")
        return "\n".join(lines) if lines else "—"

    def _open_calibration_files(self) -> None:
        """Open both calibration files in external editor."""
        if self.intrinsic_path and self.intrinsic_path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.intrinsic_path.resolve())))
        if self.extrinsic_path and self.extrinsic_path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.extrinsic_path.resolve())))

    def _load_file_metadata(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}

        # Load intrinsic metadata
        if self.intrinsic_path and self.intrinsic_path.exists():
            try:
                with open(self.intrinsic_path, "r", encoding="utf-8") as handle:
                    payload = yaml.safe_load(handle) or {}
                metadata["intrinsic_file"] = self.intrinsic_path.name
                metadata["file_path"] = str(self.intrinsic_path.parent)
                metadata["pattern_size"] = payload.get("pattern_size")
                metadata["square_size"] = payload.get("square_size") or payload.get("square_length")
                metadata["updated_at"] = payload.get("updated_at")
            except Exception:  # noqa: BLE001
                metadata["intrinsic_file"] = self.intrinsic_path.name + " (error reading)"

        # Load extrinsic metadata
        if self.extrinsic_path and self.extrinsic_path.exists():
            try:
                with open(self.extrinsic_path, "r", encoding="utf-8") as handle:
                    payload = yaml.safe_load(handle) or {}
                metadata["extrinsic_file"] = self.extrinsic_path.name
                # Use extrinsic timestamp if intrinsic didn't have one
                if not metadata.get("updated_at"):
                    metadata["updated_at"] = payload.get("updated_at")
                if not metadata.get("file_path"):
                    metadata["file_path"] = str(self.extrinsic_path.parent)
            except Exception:  # noqa: BLE001
                metadata["extrinsic_file"] = self.extrinsic_path.name + " (error reading)"

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
