"""Simple dialog to inspect calibration matrices and residual errors."""
from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.yaml_utils import load_yaml, save_yaml
from PyQt6.QtCore import Qt, QRect, QRectF, QSizeF, QUrl, QPointF, QMarginsF
from PyQt6.QtGui import (
    QColor,
    QDesktopServices,
    QPainter,
    QPen,
    QPixmap,
    QPageLayout,
    QPageSize,
    QPolygonF,
    QTextDocument,
)
from PyQt6.QtPrintSupport import QPrinter
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
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
        dataset_path: Optional[Path] = None,
    ) -> None:
        super().__init__(parent)
        self.matrices = matrices or {}
        self.extrinsic = extrinsic or {}
        self.intrinsic_path = intrinsic_path
        self.extrinsic_path = extrinsic_path
        self.dataset_paths = dataset_paths or []
        self.dataset_path = dataset_path
        self.file_metadata = self._load_file_metadata()
        self.setWindowTitle("Calibration report")
        self.setMinimumWidth(style.DIALOG_REPORT_MIN_W)
        self._report_data: dict = {}
        self._build_ui()

    def _load_report_data(self) -> None:
        """Populate self._report_data from cache or by computing from corner files."""
        if not self.dataset_path:
            return
        from config import APP_VERSION
        from backend.services.calibration.calibration_report_cache import (
            build_report_cache, load_report_cache,
        )
        updated_at = self.file_metadata.get("updated_at")
        cached = load_report_cache(self.dataset_path, APP_VERSION, updated_at)
        if cached is not None:
            self._report_data = cached
            return
        # Cache missing or stale — build it now (also writes to disk)
        build_report_cache(
            dataset_path=self.dataset_path,
            matrices=self.matrices,
            file_metadata={**self.file_metadata, "updated_at": updated_at},
        )
        cached = load_report_cache(self.dataset_path, APP_VERSION, updated_at)
        self._report_data = cached or {}

    def _build_ui(self) -> None:
        self._load_report_data()
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
            card_layout.addWidget(style.section_heading_label("Calibration information:"))
            card_layout.addWidget(metadata_group)

        card_layout.addWidget(style.section_heading_label("Intrinsic calibration:"))
        card_layout.addWidget(self._intrinsic_group())
        card_layout.addWidget(style.section_heading_label("Extrinsic calibration:"))
        card_layout.addWidget(self._extrinsic_group())

        # Chessboard coverage charts
        coverage_widget = self._chessboard_coverage_group()
        if coverage_widget:
            card_layout.addWidget(style.section_heading_label("Chessboard coverage:"))
            card_layout.addWidget(coverage_widget)

        # Distortion map
        distortion_widget = self._distortion_map_group()
        if distortion_widget:
            card_layout.addWidget(style.section_heading_label("Distortion map:"))
            card_layout.addWidget(distortion_widget)

        # Pattern pose diversity
        pose_widget = self._pose_diversity_group()
        if pose_widget:
            card_layout.addWidget(style.section_heading_label("Calibration pattern pose diversity:"))
            card_layout.addWidget(pose_widget)

        card_layout.addStretch(1)

        self._card = card
        scroll = QScrollArea(self)
        scroll.setWidget(card)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        layout.addWidget(scroll)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.setContentsMargins(0, 8, 0, 0)
        from config import APP_VERSION
        ver_label = QLabel(f"GUI v{APP_VERSION}")
        ver_label.setStyleSheet("color: #9099a8; font-size: 12px;")
        btn_row.addWidget(ver_label)
        btn_row.addStretch(1)
        has_intrinsic = self.intrinsic_path and self.intrinsic_path.exists()
        has_extrinsic = self.extrinsic_path and self.extrinsic_path.exists()
        if has_intrinsic or has_extrinsic:
            open_button = QPushButton("Open calibration files")
            open_button.clicked.connect(self._open_calibration_files)
            btn_row.addWidget(open_button)
        pdf_button = QPushButton("Export PDF")
        pdf_button.clicked.connect(self._export_pdf)
        btn_row.addWidget(pdf_button)
        layout.addLayout(btn_row)

        self.resize(style.DIALOG_REPORT_MIN_W + 40, 800)

    def refresh_data(
        self,
        matrices: Optional[Dict[str, dict]],
        extrinsic: Optional[Dict[str, dict]],
        intrinsic_path: Optional[Path],
        extrinsic_path: Optional[Path] = None,
        dataset_paths: Optional[List[str]] = None,
        dataset_path: Optional[Path] = None,
    ) -> None:
        self.matrices = matrices or {}
        self.extrinsic = extrinsic or {}
        self.intrinsic_path = intrinsic_path
        self.extrinsic_path = extrinsic_path
        self.dataset_paths = dataset_paths or []
        self.dataset_path = dataset_path
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

    def _file_info_group(self) -> Optional[QWidget]:
        # Show info if we have any metadata or dataset paths
        if not self.file_metadata and not self.dataset_paths:
            return None
        panel, layout = style.make_panel("file_info_panel")
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
        panel, layout = style.make_panel("dataset_list_panel")
        for dataset in self.dataset_paths:
            label = QLabel(dataset)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(label)
        return panel

    def _intrinsic_group(self) -> QWidget:
        panel, layout = style.make_panel("intrinsic_panel")
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
        panel, layout = style.make_panel("extrinsic_panel", spacing=8)
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

        # Resolve square size → meters for unit-aware display (done first so frame_note can reference it)
        sq_meta = self.file_metadata.get("square_size")
        square_size_m: Optional[float] = None
        if isinstance(sq_meta, dict):
            val = sq_meta.get("value")
            unit = (sq_meta.get("unit") or "").lower()
            if isinstance(val, (int, float)):
                if unit == "mm":
                    square_size_m = float(val) / 1000.0
                elif unit == "cm":
                    square_size_m = float(val) / 100.0
                elif unit in ("m", "meters"):
                    square_size_m = float(val)
        elif isinstance(sq_meta, (int, float)):
            square_size_m = float(sq_meta) / 1000.0  # bare number → assume mm

        if square_size_m:
            frame_note_text = (
                "Extrinsic transform maps points from the LWIR camera frame into the visible camera frame. "
                f"Translation converted to meters using the stored square size ({square_size_m * 1000:.1f} mm). "
                "Rotation shown as ZYX Euler angles."
            )
        else:
            frame_note_text = (
                "Extrinsic transform maps points from the LWIR camera frame into the visible camera frame. "
                "Translation in pattern squares (square physical size not available for meter conversion). "
                "Rotation shown as ZYX Euler angles."
            )
        frame_note = QLabel(frame_note_text)
        frame_note.setWordWrap(True)
        frame_note.setStyleSheet(f"color: {style.TEXT_SECONDARY};")
        frame_note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._field_label("Frame note"), frame_note)

        samples = payload.get("samples", 0)
        rms = payload.get("reprojection_error")
        summary = f"Samples: {samples}"
        if rms is not None:
            summary += f" | RMS error: {rms:.4f} px"
        summary_label = QLabel(summary)
        summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._field_label("Summary"), summary_label)

        baseline = payload.get("baseline")
        if baseline is not None and square_size_m:
            baseline_text = f"{baseline * square_size_m:.4f} m"
        elif baseline is not None:
            baseline_text = f"{baseline:.4f} squares"
        else:
            baseline_text = "—"
        baseline_label = QLabel(baseline_text)
        baseline_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._field_label("Baseline (‖T‖)"), baseline_label)

        updated = payload.get("updated_at")
        updated_str = str(updated) if updated and not isinstance(updated, dict) else None
        updated_label = QLabel(self._format_timestamp(updated_str))
        updated_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._field_label("Computed"), updated_label)

        if square_size_m:
            t_m = [v * square_size_m for v in translation]
            translation_text = "[ " + ", ".join(f"{v:.4f}" for v in t_m) + " ]  (m)"
        else:
            translation_text = self._format_vector(translation) + "  (pattern squares)"
        translation_label = QLabel(translation_text)
        translation_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        translation_label.setStyleSheet(style.MONO_TEXT_STYLE)
        form.addRow(self._field_label("Translation (LWIR → Visible)"), translation_label)

        rotation_label = QLabel(self._format_matrix(rotation))
        rotation_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rotation_label.setWordWrap(True)
        rotation_label.setStyleSheet(style.MONO_TEXT_STYLE + " padding: 2px 0 8px 0;")
        form.addRow(self._field_label("Rotation matrix (LWIR → Visible)"), rotation_label)

        try:
            R = rotation
            pitch_rad = -math.asin(max(-1.0, min(1.0, R[2][0])))
            cos_p = math.cos(pitch_rad)
            if abs(cos_p) > 1e-6:
                roll_rad = math.atan2(R[2][1] / cos_p, R[2][2] / cos_p)
                yaw_rad = math.atan2(R[1][0] / cos_p, R[0][0] / cos_p)
            else:
                roll_rad = math.atan2(-R[1][2], R[1][1])
                yaw_rad = 0.0
            euler_text = (
                f"roll {math.degrees(roll_rad):.2f}°   "
                f"pitch {math.degrees(pitch_rad):.2f}°   "
                f"yaw {math.degrees(yaw_rad):.2f}°"
            )
            euler_label = QLabel(euler_text)
            euler_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            euler_label.setStyleSheet(style.MONO_TEXT_STYLE)
            form.addRow(self._field_label("Rotation (roll/pitch/yaw)"), euler_label)
        except Exception:  # noqa: BLE001
            pass

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
        panel, layout = style.make_panel(f"channel_panel_{key}")
        heading = style.section_heading_label(f"{title}:")
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
        _rejected = payload.get("rejected_views", payload.get("rejected_pairs", []))
        rejected = len(_rejected) if isinstance(_rejected, list) else int(_rejected or 0)
        error = payload.get("reprojection_error")
        sample_str = f"Samples: {samples} used" + (f", {rejected} rejected" if rejected else "")
        header = QLabel(
            f"{sample_str} | RMS error: {error:.4f} px" if error is not None else sample_str
        )
        form.addRow(self._field_label("Summary"), header)
        matrix = QLabel(self._format_matrix(payload.get("camera_matrix")))
        matrix.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        matrix.setWordWrap(True)
        matrix.setStyleSheet(style.MONO_TEXT_STYLE + " padding: 2px 0 8px 0;")
        form.addRow(self._field_label("Camera matrix"), matrix)
        distortion = QLabel(self._format_vector(payload.get("distortion")))
        distortion.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._field_label("Distortion"), distortion)
        fov_text = self._format_fov(payload.get("camera_matrix"), payload.get("image_size"))
        if fov_text:
            fov_label = QLabel(fov_text)
            fov_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            fov_label.setStyleSheet(style.MONO_TEXT_STYLE)
            form.addRow(self._field_label("FOV (H × V)"), fov_label)
        layout.addLayout(form)
        return panel

    def _chessboard_coverage_group(self) -> Optional[QWidget]:
        """Build side-by-side chessboard coverage charts (LWIR + Visible)."""
        lwir_quads = self._report_data.get("lwir_quads") or []
        vis_quads = self._report_data.get("vis_quads") or []
        if not lwir_quads and not vis_quads:
            return None

        panel = QWidget()
        panel.setObjectName("coverage_panel")
        panel.setStyleSheet(style.panel_body_style("coverage_panel"))
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(12)

        chart_w, chart_h = style.CHART_W_PAIR, style.CHART_H_PAIR

        # LWIR chart
        lwir_label = QLabel()
        lwir_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lwir_pix = _render_chessboard_coverage(
            lwir_quads, chart_w, chart_h, "LWIR", QColor(220, 60, 60, 50),
        )
        lwir_label.setPixmap(lwir_pix)
        layout.addWidget(lwir_label)

        # Visible chart
        vis_label = QLabel()
        vis_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vis_pix = _render_chessboard_coverage(
            vis_quads, chart_w, chart_h, "Visible", QColor(60, 120, 220, 50),
        )
        vis_label.setPixmap(vis_pix)
        layout.addWidget(vis_label)

        return panel

    def _pose_diversity_group(self) -> Optional[QWidget]:
        """Build tilt-scatter, distance-histogram, and tilt-vs-distance charts."""
        lwir_poses = [tuple(p) for p in self._report_data.get("lwir_poses") or []]
        vis_poses = [tuple(p) for p in self._report_data.get("vis_poses") or []]
        if not lwir_poses and not vis_poses:
            return None
        dist_unit = self._report_data.get("dist_unit", "grid sq.")

        outer, layout = style.make_panel("pose_diversity_panel", spacing=8)

        chart_w, chart_h = style.CHART_W_PAIR, style.CHART_H_GRID
        lwir_color = QColor(220, 60, 60)
        vis_color = QColor(60, 120, 220)

        def _row(lwir_pix: QPixmap, vis_pix: QPixmap) -> QWidget:
            row = QWidget()
            row.setStyleSheet("background: transparent;")
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(12)
            for pix in (lwir_pix, vis_pix):
                lbl = QLabel()
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setPixmap(pix)
                rl.addWidget(lbl)
            return row

        def _row_with_caption(lwir_pix: QPixmap, vis_pix: QPixmap, caption: str) -> QWidget:
            container = QWidget()
            container.setStyleSheet("background: transparent;")
            cl = QVBoxLayout(container)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(2)
            cl.addWidget(_row(lwir_pix, vis_pix))
            cap = QLabel(caption)
            cap.setStyleSheet(f"color: {style.TEXT_CAPTION}; font-size: 10px; font-style: italic;")
            cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cl.addWidget(cap)
            return container

        layout.addWidget(_row_with_caption(
            _render_tilt_scatter(lwir_poses, chart_w, chart_h, "LWIR — Tilt distribution", lwir_color),
            _render_tilt_scatter(vis_poses, chart_w, chart_h, "Visible — Tilt distribution", vis_color),
            "Each dot = one image. Good coverage: cloud spread broadly around the origin, not clustered at zero.",
        ))
        layout.addWidget(_row_with_caption(
            _render_distance_histogram(lwir_poses, chart_w, chart_h, "LWIR — Pattern distance distribution", lwir_color, dist_unit),
            _render_distance_histogram(vis_poses, chart_w, chart_h, "Visible — Pattern distance distribution", vis_color, dist_unit),
            "Good coverage: bars spanning a wide range. All bars at the same distance limits depth diversity.",
        ))
        layout.addWidget(_row_with_caption(
            _render_tilt_vs_distance(lwir_poses, chart_w, chart_h, "LWIR — Tilt magnitude (Y) vs. Distance (X)", lwir_color, dist_unit),
            _render_tilt_vs_distance(vis_poses, chart_w, chart_h, "Visible — Tilt magnitude (Y) vs. Distance (X)", vis_color, dist_unit),
            "Good coverage: dots spread across both axes with no strong diagonal correlation.",
        ))

        return outer

    def _distortion_map_group(self) -> Optional[QWidget]:
        """Build side-by-side distortion warp-grid charts (LWIR + Visible)."""
        lwir_payload = self.matrices.get("lwir")
        vis_payload = self.matrices.get("visible")
        if not lwir_payload and not vis_payload:
            return None

        panel = QWidget()
        panel.setObjectName("distortion_map_panel")
        panel.setStyleSheet(style.panel_body_style("distortion_map_panel"))
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(12)

        chart_w, chart_h = style.CHART_W_PAIR, style.CHART_H_PAIR

        lwir_label = QLabel()
        lwir_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lwir_label.setPixmap(
            _render_distortion_map(lwir_payload, chart_w, chart_h, "LWIR", QColor(220, 60, 60))
        )
        layout.addWidget(lwir_label)

        vis_label = QLabel()
        vis_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vis_label.setPixmap(
            _render_distortion_map(vis_payload, chart_w, chart_h, "Visible", QColor(60, 120, 220))
        )
        layout.addWidget(vis_label)

        return panel

    def _field_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(f"font-weight: 700; color: {style.TEXT_TITLE};")
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

    def _format_fov(self, camera_matrix, image_size) -> str:
        if not camera_matrix or not image_size or len(image_size) < 2:
            return ""
        try:
            fx = float(camera_matrix[0][0])
            fy = float(camera_matrix[1][1])
            w, h = float(image_size[0]), float(image_size[1])
            hfov = math.degrees(2 * math.atan(w / (2 * fx)))
            vfov = math.degrees(2 * math.atan(h / (2 * fy)))
            return f"{hfov:.1f}° × {vfov:.1f}°   (fx={fx:.0f} px,  fy={fy:.0f} px)"
        except Exception:  # noqa: BLE001
            return ""

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

    def _build_extrinsic_html_rows(self) -> str:
        """Return HTML table rows for the extrinsic calibration section."""
        payload = self.extrinsic or {}
        translation = payload.get("translation")
        rotation = payload.get("rotation")
        if not translation or not rotation:
            return '<tr><td colspan="2">Not available. Compute the stereo extrinsic transform first.</td></tr>'

        def _row(label: str, value: str) -> str:
            return f'<tr><td class="lbl" width="33%">{label}</td><td>{value}</td></tr>'

        def _mono(text: str) -> str:
            return f'<span style="font-family:monospace;">{text}</span>'

        sq_meta = self.file_metadata.get("square_size")
        square_size_m: Optional[float] = None
        if isinstance(sq_meta, dict):
            val = sq_meta.get("value")
            unit = (sq_meta.get("unit") or "").lower()
            if isinstance(val, (int, float)):
                if unit == "mm":
                    square_size_m = float(val) / 1000.0
                elif unit == "cm":
                    square_size_m = float(val) / 100.0
                elif unit in ("m", "meters"):
                    square_size_m = float(val)
        elif isinstance(sq_meta, (int, float)):
            square_size_m = float(sq_meta) / 1000.0

        rows = ""
        samples = payload.get("samples", 0)
        _rejected = payload.get("rejected_views", payload.get("rejected_pairs", []))
        rejected = len(_rejected) if isinstance(_rejected, list) else int(_rejected or 0)
        rms = payload.get("reprojection_error")
        summary = f"Samples: {samples} used" + (f", {rejected} rejected" if rejected else "")
        if rms is not None:
            summary += f" | RMS error: {rms:.4f} px"
        rows += _row("Summary", summary)

        updated = payload.get("updated_at")
        updated_str = str(updated) if updated and not isinstance(updated, dict) else None
        rows += _row("Computed", self._format_timestamp(updated_str))

        baseline = payload.get("baseline")
        if baseline is not None and square_size_m:
            baseline_text = f"{baseline * square_size_m:.4f} m"
        elif baseline is not None:
            baseline_text = f"{baseline:.4f} squares"
        else:
            baseline_text = "—"
        rows += _row("Baseline (‖T‖)", baseline_text)

        if square_size_m:
            t_m = [v * square_size_m for v in translation]
            translation_text = "[ " + ", ".join(f"{v:.4f}" for v in t_m) + " ]  (m)"
        else:
            translation_text = self._format_vector(translation) + "  (pattern squares)"
        rows += _row("Translation (LWIR → Visible)", _mono(translation_text))

        mat_str = self._format_matrix(rotation).replace("\n", "<br/>")
        rows += _row("Rotation matrix (LWIR → Visible)", _mono(mat_str))

        try:
            R = rotation
            pitch_rad = -math.asin(max(-1.0, min(1.0, R[2][0])))
            cos_p = math.cos(pitch_rad)
            if abs(cos_p) > 1e-6:
                roll_rad = math.atan2(R[2][1] / cos_p, R[2][2] / cos_p)
                yaw_rad = math.atan2(R[1][0] / cos_p, R[0][0] / cos_p)
            else:
                roll_rad = math.atan2(-R[1][2], R[1][1])
                yaw_rad = 0.0
            euler_text = (
                f"roll {math.degrees(roll_rad):.2f}°   "
                f"pitch {math.degrees(pitch_rad):.2f}°   "
                f"yaw {math.degrees(yaw_rad):.2f}°"
            )
            rows += _row("Rotation (roll/pitch/yaw)", _mono(euler_text))
        except Exception:  # noqa: BLE001
            pass

        note = (
            "Extrinsic transform maps points from the LWIR camera frame into the visible camera frame. "
            + (f"Translation converted to meters using {square_size_m * 1000:.1f} mm square size. "
               if square_size_m else "Translation in pattern squares (square size not available). ")
            + "Rotation shown as ZYX Euler angles."
        )
        rows += _row("Note", f'<span style="color:#444; font-size:8pt;">{note}</span>')
        return rows

    def _build_pdf_html(self) -> str:
        """Build HTML string for QTextDocument-based PDF export."""
        import base64
        from PyQt6.QtCore import QBuffer, QIODevice
        from config import APP_VERSION
        from datetime import date, datetime

        def _pix_url(pix: QPixmap) -> str:
            buf = QBuffer()
            buf.open(QIODevice.OpenModeFlag.WriteOnly)
            pix.save(buf, "PNG")
            buf.close()
            b64 = base64.b64encode(bytes(buf.data())).decode("ascii")
            return f"data:image/png;base64,{b64}"

        def _row(label: str, value: str) -> str:
            return f'<tr><td class="lbl" width="33%">{label}</td><td>{value}</td></tr>'

        def _mono(text: str) -> str:
            return f'<span style="font-family:monospace;">{text}</span>'

        def _panel(content: str) -> str:
            return (
                '<table width="100%" cellspacing="0" cellpadding="0"'
                ' style="border:1px solid #dde1e7; margin-bottom:8pt;">'
                f'<tr><td style="padding:6pt;">{content}</td></tr></table>'
            )

        # ---- header metadata ----
        cal_raw = self.file_metadata.get("updated_at", "")
        try:
            if cal_raw:
                _c = re.sub(r'([+\-]\d{2}:\d{2})Z$', r'\1', cal_raw)
                if _c.endswith('Z'):
                    _c = _c[:-1] + '+00:00'
                cal_date = datetime.fromisoformat(_c).strftime("%Y-%m-%d %H:%M")
            else:
                cal_date = ""
        except Exception:
            cal_date = cal_raw
        pattern = self.file_metadata.get("pattern_size")
        pattern_str = (
            f"{pattern[0]}×{pattern[1]}"
            if isinstance(pattern, (list, tuple)) and len(pattern) == 2 else ""
        )
        sq = self.file_metadata.get("square_size")
        sq_str = (
            f"{sq.get('value')} {sq.get('unit', '')}"
            if isinstance(sq, dict) and sq.get("value") else ""
        )
        file_path = self.file_metadata.get("file_path", "")

        meta_parts = [f"Generated: {date.today()}", f"GUI v{APP_VERSION}"]
        if cal_date:
            meta_parts.append(f"Calibration: {cal_date}")
        if pattern_str:
            meta_parts.append(f"Pattern: {pattern_str}")
        if sq_str:
            meta_parts.append(f"Square: {sq_str}")
        meta_line = " &nbsp;&middot;&nbsp; ".join(meta_parts)
        path_line = f'<br/><span style="font-size:7pt;">{file_path}</span>' if file_path else ""

        # ---- calibration information ----
        info_rows = ""
        info_rows += _row("Intrinsic file", self.file_metadata.get("intrinsic_file", "—"))
        info_rows += _row("Extrinsic file", self.file_metadata.get("extrinsic_file", "—"))
        if file_path:
            info_rows += _row("Location", file_path)
        info_rows += _row("Computed", self._format_timestamp(cal_raw))
        info_rows += _row("Pattern", self._format_pattern_label(pattern))
        info_rows += _row("Square side", self._format_square_label(sq))
        if self.dataset_paths:
            info_rows += _row("Datasets used", ", ".join(Path(p).name for p in self.dataset_paths))

        # ---- intrinsic calibration ----
        def _channel_rows(key: str, title: str) -> str:
            payload = self.matrices.get(key)
            h = f'<tr><td colspan="2"><b>{title}</b></td></tr>'
            if not payload or not payload.get("camera_matrix"):
                return h + _row("", "Not available. Compute calibration first.")
            samples = payload.get("samples", 0)
            _rej = payload.get("rejected_views", payload.get("rejected_pairs", []))
            rejected = len(_rej) if isinstance(_rej, list) else int(_rej or 0)
            err = payload.get("reprojection_error")
            summary = f"Samples: {samples} used" + (f", {rejected} rejected" if rejected else "")
            if err is not None:
                summary += f" | RMS error: {err:.4f} px"
            h += _row("Summary", summary)
            mat_str = self._format_matrix(payload.get("camera_matrix")).replace("\n", "<br/>")
            h += _row("Camera matrix", _mono(mat_str))
            h += _row("Distortion", _mono(self._format_vector(payload.get("distortion"))))
            fov_str = self._format_fov(payload.get("camera_matrix"), payload.get("image_size"))
            if fov_str:
                h += _row("FOV (H × V)", _mono(fov_str))
            return h

        intr_rows = _channel_rows("visible", "Visible camera")
        intr_rows += '<tr><td colspan="2" style="height:6pt;"></td></tr>'
        intr_rows += _channel_rows("lwir", "LWIR camera")

        # ---- charts ----
        disp_w = 238  # display width per chart in 2-column layout (A4 usable ~510pt minus padding)

        # Coverage
        lwir_quads = self._report_data.get("lwir_quads") or []
        vis_quads = self._report_data.get("vis_quads") or []
        coverage_html = ""
        if lwir_quads or vis_quads:
            cw, ch = style.CHART_W_PAIR, style.CHART_H_PAIR
            lwir_pix = _render_chessboard_coverage(lwir_quads, cw, ch, "LWIR", QColor(220, 60, 60, 50))
            vis_pix = _render_chessboard_coverage(vis_quads, cw, ch, "Visible", QColor(60, 120, 220, 50))
            imgs = (
                f'<table width="100%" cellspacing="0"><tr>'
                f'<td align="center"><img src="{_pix_url(lwir_pix)}" width="{disp_w}"/></td>'
                f'<td align="center"><img src="{_pix_url(vis_pix)}" width="{disp_w}"/></td>'
                f'</tr></table>'
            )
            coverage_html = f'<h3>Chessboard coverage</h3>{_panel(imgs)}'

        # Distortion maps
        lwir_payload = self.matrices.get("lwir")
        vis_payload = self.matrices.get("visible")
        distortion_html = ""
        if lwir_payload or vis_payload:
            cw, ch = style.CHART_W_PAIR, style.CHART_H_PAIR
            lwir_dpix = _render_distortion_map(lwir_payload, cw, ch, "LWIR", QColor(220, 60, 60))
            vis_dpix = _render_distortion_map(vis_payload, cw, ch, "Visible", QColor(60, 120, 220))
            imgs = (
                f'<table width="100%" cellspacing="0"><tr>'
                f'<td align="center"><img src="{_pix_url(lwir_dpix)}" width="{disp_w}"/></td>'
                f'<td align="center"><img src="{_pix_url(vis_dpix)}" width="{disp_w}"/></td>'
                f'</tr></table>'
            )
            distortion_html = f'<h3>Distortion map</h3>{_panel(imgs)}'

        # Pose diversity
        lwir_poses = [tuple(p) for p in self._report_data.get("lwir_poses") or []]
        vis_poses = [tuple(p) for p in self._report_data.get("vis_poses") or []]
        dist_unit = self._report_data.get("dist_unit", "grid sq.")
        pose_html = ""
        if lwir_poses or vis_poses:
            cw, ch = style.CHART_W_PAIR, style.CHART_H_GRID

            def _pose_row(lwir_pix: QPixmap, vis_pix: QPixmap, caption: str) -> str:
                return (
                    f'<tr>'
                    f'<td align="center"><img src="{_pix_url(lwir_pix)}" width="{disp_w}"/></td>'
                    f'<td align="center"><img src="{_pix_url(vis_pix)}" width="{disp_w}"/></td>'
                    f'</tr>'
                    f'<tr><td colspan="2" align="center"'
                    f' style="color:#666; font-size:8pt; font-style:italic; padding-bottom:4pt;">'
                    f'{caption}</td></tr>'
                )

            pose_rows = _pose_row(
                _render_tilt_scatter(lwir_poses, cw, ch, "LWIR — Tilt distribution", QColor(220, 60, 60)),
                _render_tilt_scatter(vis_poses, cw, ch, "Visible — Tilt distribution", QColor(60, 120, 220)),
                "Each dot = one image. Good coverage: cloud spread broadly around the origin, not clustered at zero.",
            )
            pose_rows += _pose_row(
                _render_distance_histogram(lwir_poses, cw, ch, "LWIR — Pattern distance distribution", QColor(220, 60, 60), dist_unit),
                _render_distance_histogram(vis_poses, cw, ch, "Visible — Pattern distance distribution", QColor(60, 120, 220), dist_unit),
                "Good coverage: bars spanning a wide range. All bars at the same distance limits depth diversity.",
            )
            pose_rows += _pose_row(
                _render_tilt_vs_distance(lwir_poses, cw, ch, "LWIR — Tilt (Y) vs. Distance (X)", QColor(220, 60, 60), dist_unit),
                _render_tilt_vs_distance(vis_poses, cw, ch, "Visible — Tilt (Y) vs. Distance (X)", QColor(60, 120, 220), dist_unit),
                "Good coverage: dots spread across both axes with no strong diagonal correlation.",
            )
            imgs = f'<table width="100%" cellspacing="0">{pose_rows}</table>'
            pose_html = f'<h3>Calibration pattern pose diversity</h3>{_panel(imgs)}'

        ext_rows = self._build_extrinsic_html_rows()

        css = (
            f"body {{ font-family: Arial, sans-serif; font-size: 9pt; color: {style.TEXT_TITLE}; }}"
            f" h2 {{ font-size: 14pt; font-weight: bold; color: {style.TEXT_TITLE}; margin: 4pt 0 10pt 0; }}"
            " h3 { font-size: 9pt; color: #5c6370; font-weight: bold; margin: 8pt 0 2pt 0; }"
            " table.data td { padding: 2pt 4pt; vertical-align: top; }"
            f" td.lbl {{ font-weight: bold; color: {style.TEXT_TITLE}; }}"
        )

        return (
            f'<!DOCTYPE html><html><head><meta charset="utf-8"><style>{css}</style></head><body>'
            f'<p style="color:#9099a8; font-size:8pt; padding-bottom:3pt; margin-bottom:6pt;'
            f' border-bottom:1px solid #c0c6d0;">{meta_line}{path_line}</p>'
            f'<h2>Calibration Report</h2>'
            f'<h3>Calibration information</h3>'
            f'{_panel(f"<table class=\'data\' width=\'100%\' cellspacing=\'0\'>{info_rows}</table>")}'
            f'<h3>Intrinsic calibration</h3>'
            f'{_panel(f"<table class=\'data\' width=\'100%\' cellspacing=\'0\'>{intr_rows}</table>")}'
            f'<h3>Extrinsic calibration</h3>'
            f'{_panel(f"<table class=\'data\' width=\'100%\' cellspacing=\'0\'>{ext_rows}</table>")}'
            f'{coverage_html}'
            f'{distortion_html}'
            f'{pose_html}'
            f'</body></html>'
        )

    def _export_pdf(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export PDF", "calibration_report.pdf", "PDF Files (*.pdf)")
        if not path:
            return
        if not path.endswith(".pdf"):
            path += ".pdf"

        printer = QPrinter()
        printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        printer.setOutputFileName(path)
        printer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
        printer.setPageOrientation(QPageLayout.Orientation.Portrait)
        printer.setPageMargins(QMarginsF(15, 15, 15, 15), QPageLayout.Unit.Millimeter)

        doc = QTextDocument()
        doc.setHtml(self._build_pdf_html())
        doc.setPageSize(QSizeF(printer.pageRect(QPrinter.Unit.Point).size()))
        getattr(doc, "print")(printer)

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
            payload = load_yaml(self.intrinsic_path)
            if payload:
                metadata["intrinsic_file"] = self.intrinsic_path.name
                metadata["file_path"] = str(self.intrinsic_path.parent)
                metadata["pattern_size"] = payload.get("pattern_size")
                metadata["square_size"] = payload.get("square_size") or payload.get("square_length")
                metadata["updated_at"] = payload.get("updated_at")
            else:
                metadata["intrinsic_file"] = self.intrinsic_path.name + " (error reading)"

        # Load extrinsic metadata
        if self.extrinsic_path and self.extrinsic_path.exists():
            payload = load_yaml(self.extrinsic_path)
            if payload:
                metadata["extrinsic_file"] = self.extrinsic_path.name
                # Use extrinsic timestamp if intrinsic didn't have one
                if not metadata.get("updated_at"):
                    metadata["updated_at"] = payload.get("updated_at")
                if not metadata.get("file_path"):
                    metadata["file_path"] = str(self.extrinsic_path.parent)
            else:
                metadata["extrinsic_file"] = self.extrinsic_path.name + " (error reading)"

        return metadata

    def _format_timestamp(self, raw: Optional[str]) -> str:
        if not raw or not isinstance(raw, str):
            return "Not specified"
        try:
            # Normalize malformed "+00:00Z" (offset + trailing Z) before parsing
            cleaned = re.sub(r'([+\-]\d{2}:\d{2})Z$', r'\1', raw)
            if cleaned.endswith('Z'):
                cleaned = cleaned[:-1] + '+00:00'
            parsed = datetime.fromisoformat(cleaned)
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


# ======================================================================
# Chessboard coverage chart helpers
# ======================================================================



def _begin_chart(width: int, height: int, title: str):
    """Create a chart pixmap with title drawn and plot area pre-filled."""
    pix = QPixmap(width * style.CHART_DPR, height * style.CHART_DPR)
    pix.setDevicePixelRatio(style.CHART_DPR)
    pix.fill(style.CHART_BG_QCOLOR)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    ml, mb, mt, mr = style.CHART_MARGINS
    pw, ph = width - ml - mr, height - mt - mb
    f = painter.font()
    f.setPixelSize(style.CHART_TITLE_FONT_SIZE)
    f.setBold(True)
    painter.setFont(f)
    painter.setPen(style.CHART_AXIS_QCOLOR)
    painter.drawText(
        QRectF(0, 2, width, mt - 2),
        Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
        title,
    )
    painter.fillRect(QRectF(ml, mt, pw, ph), style.CHART_PLOT_BG_QCOLOR)
    return pix, painter, ml, mb, mt, mr, pw, ph


def _chart_no_data(pix: QPixmap, painter: QPainter, ml, mb, mt, mr, pw, ph) -> QPixmap:
    """Finish an empty chart with a grey border and return the pixmap."""
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(style.CHART_PLACEHOLDER_QCOLOR, 1))
    painter.drawRect(QRectF(ml, mt, pw, ph))
    painter.end()
    return pix


def _end_chart(painter: QPainter, ml, mb, mt, mr, pw, ph, n: int) -> None:
    """Draw border + sample-count annotation and end the painter."""
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(style.CHART_AXIS_QCOLOR, 0.8))
    painter.drawRect(QRectF(ml, mt, pw, ph))
    f = painter.font()
    f.setPixelSize(style.CHART_FONT_SIZE - 1)
    painter.setFont(f)
    painter.setPen(QColor("#666"))
    painter.drawText(
        QRectF(ml + 4, mt + 4, pw - 8, 14),
        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
        f"n={n}",
    )
    painter.end()


def _set_tick_font(painter: QPainter) -> None:
    """Switch painter to the small tick-label font."""
    f = painter.font()
    f.setPixelSize(style.CHART_FONT_SIZE)
    f.setBold(False)
    painter.setFont(f)




def _render_chessboard_coverage(
    quads: List[List[List[float]]],
    width: int,
    height: int,
    title: str,
    fill_color: QColor,
) -> QPixmap:
    """Draw chessboard quadrilaterals as semi-transparent polygons on a [0,1]² canvas."""
    pix = QPixmap(width * style.CHART_DPR, height * style.CHART_DPR)
    pix.setDevicePixelRatio(style.CHART_DPR)
    pix.fill(style.CHART_BG_QCOLOR)

    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    f = p.font()
    f.setPixelSize(style.CHART_FONT_SIZE)
    p.setFont(f)

    # Title
    f.setPixelSize(style.CHART_TITLE_FONT_SIZE)
    f.setBold(True)
    p.setFont(f)
    p.setPen(style.CHART_AXIS_QCOLOR)
    ml, mb, mt, mr = style.CHART_MARGINS
    p.drawText(
        QRectF(0, 2, width, mt - 2),
        Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
        title,
    )
    f.setBold(False)
    f.setPixelSize(style.CHART_FONT_SIZE)
    p.setFont(f)

    pw = width - ml - mr
    ph = height - mt - mb

    # Background image area
    p.fillRect(QRectF(ml, mt, pw, ph), style.CHART_PLOT_BG_QCOLOR)

    # Draw axes with ticks (0.0 – 1.0)
    pen = QPen(style.CHART_AXIS_QCOLOR, 1)
    p.setPen(pen)
    p.drawLine(ml, mt, ml, height - mb)
    p.drawLine(ml, height - mb, width - mr, height - mb)
    n_ticks = 6
    for i in range(n_ticks):
        t = i / (n_ticks - 1)
        x = ml + t * pw
        p.drawLine(int(x), height - mb, int(x), height - mb + 3)
        p.drawText(
            QRectF(x - 16, height - mb + 4, 32, 14),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            f"{t:.1f}",
        )
        y = height - mb - t * ph
        p.drawLine(ml - 3, int(y), ml, int(y))
        p.drawText(
            QRectF(0, y - 7, ml - 5, 14),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"{t:.1f}",
        )

    # Axis labels
    p.drawText(
        QRectF(ml, height - 10, pw, 12),
        Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, "x",
    )
    p.save()
    p.translate(8, mt + ph / 2)
    p.rotate(-90)
    p.drawText(
        QRectF(-ph / 2, -8, ph, 16),
        Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, "y",
    )
    p.restore()

    # Draw chessboard quads
    if quads:
        stroke_color = QColor(
            fill_color.red(), fill_color.green(), fill_color.blue(),
            min(fill_color.alpha() + 60, 200),
        )
        pen = QPen(stroke_color, 1.0)
        p.setBrush(fill_color)

        for quad in quads:
            polygon = QPolygonF()
            for pt in quad:
                px = ml + pt[0] * pw
                py = mt + pt[1] * ph
                polygon.append(QPointF(px, py))
            p.setPen(pen)
            p.drawPolygon(polygon)

    # Count label
    if quads:
        p.setPen(style.CHART_AXIS_QCOLOR)
        p.drawText(
            QRectF(ml + 4, mt + 4, pw, 16),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            f"{len(quads)} detections",
        )

    # Border
    p.setPen(QPen(style.CHART_PLACEHOLDER_QCOLOR, 1))
    p.drawRect(QRectF(ml, mt, pw, ph))

    p.end()
    return pix


# ======================================================================
# Distortion map chart helpers
# ======================================================================

def _render_distortion_map(
    payload: Optional[dict],
    width: int,
    height: int,
    title: str,
    main_color: QColor,
) -> QPixmap:
    """Render a warp-grid distortion map for a single camera channel.

    Draws a regular undistorted grid in light gray and the same grid after
    applying the lens distortion model in color, so barrel/pincushion is
    immediately visible.
    """
    pix = QPixmap(width * style.CHART_DPR, height * style.CHART_DPR)
    pix.setDevicePixelRatio(style.CHART_DPR)
    pix.fill(style.CHART_BG_QCOLOR)

    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    ml, mb, mt, mr = style.CHART_MARGINS
    pw = width - ml - mr
    ph = height - mt - mb

    # Title
    f = painter.font()
    f.setPixelSize(style.CHART_TITLE_FONT_SIZE)
    f.setBold(True)
    painter.setFont(f)
    painter.setPen(style.CHART_AXIS_QCOLOR)
    painter.drawText(
        QRectF(0, 2, width, mt - 2),
        Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
        title,
    )

    painter.fillRect(QRectF(ml, mt, pw, ph), style.CHART_PLOT_BG_QCOLOR)

    if not payload:
        painter.setPen(QPen(style.CHART_PLACEHOLDER_QCOLOR, 1))
        painter.drawRect(QRectF(ml, mt, pw, ph))
        painter.end()
        return pix

    cam_mat = payload.get("camera_matrix")
    dist_coeffs = payload.get("distortion")
    img_size = payload.get("image_size")
    if not cam_mat or not dist_coeffs or not img_size or len(img_size) < 2:
        painter.setPen(QPen(style.CHART_PLACEHOLDER_QCOLOR, 1))
        painter.drawRect(QRectF(ml, mt, pw, ph))
        painter.end()
        return pix

    fx = float(cam_mat[0][0])
    fy = float(cam_mat[1][1])
    cx = float(cam_mat[0][2])
    cy = float(cam_mat[1][2])
    img_w = float(img_size[0])
    img_h = float(img_size[1])

    k1 = float(dist_coeffs[0]) if len(dist_coeffs) > 0 else 0.0
    k2 = float(dist_coeffs[1]) if len(dist_coeffs) > 1 else 0.0
    td1 = float(dist_coeffs[2]) if len(dist_coeffs) > 2 else 0.0  # p1 tangential
    td2 = float(dist_coeffs[3]) if len(dist_coeffs) > 3 else 0.0  # p2 tangential
    k3 = float(dist_coeffs[4]) if len(dist_coeffs) > 4 else 0.0

    def to_canvas(px: float, py: float):
        return ml + (px / img_w) * pw, mt + (py / img_h) * ph

    def apply_distortion(px: float, py: float):
        xn = (px - cx) / fx
        yn = (py - cy) / fy
        r2 = xn * xn + yn * yn
        r4 = r2 * r2
        r6 = r4 * r2
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        xd = xn * radial + 2.0 * td1 * xn * yn + td2 * (r2 + 2.0 * xn * xn)
        yd = yn * radial + td1 * (r2 + 2.0 * yn * yn) + 2.0 * td2 * xn * yn
        return xd * fx + cx, yd * fy + cy

    N_GRID = 8   # grid lines per axis
    N_SAMP = 40  # sample points per grid line for smooth curves

    # Undistorted reference grid (light gray)
    painter.setPen(QPen(QColor(190, 190, 190), 0.8))
    for i in range(N_GRID + 1):
        y = img_h * i / N_GRID
        prev = to_canvas(0.0, y)
        for j in range(1, N_SAMP + 1):
            curr = to_canvas(img_w * j / N_SAMP, y)
            painter.drawLine(QPointF(*prev), QPointF(*curr))
            prev = curr

        x = img_w * i / N_GRID
        prev = to_canvas(x, 0.0)
        for j in range(1, N_SAMP + 1):
            curr = to_canvas(x, img_h * j / N_SAMP)
            painter.drawLine(QPointF(*prev), QPointF(*curr))
            prev = curr

    # Distorted grid (colored)
    painter.setPen(QPen(main_color, 1.2))
    for i in range(N_GRID + 1):
        y = img_h * i / N_GRID
        prev = to_canvas(*apply_distortion(0.0, y))
        for j in range(1, N_SAMP + 1):
            curr = to_canvas(*apply_distortion(img_w * j / N_SAMP, y))
            painter.drawLine(QPointF(*prev), QPointF(*curr))
            prev = curr

        x = img_w * i / N_GRID
        prev = to_canvas(*apply_distortion(x, 0.0))
        for j in range(1, N_SAMP + 1):
            curr = to_canvas(*apply_distortion(x, img_h * j / N_SAMP))
            painter.drawLine(QPointF(*prev), QPointF(*curr))
            prev = curr

    # Border
    painter.setPen(QPen(style.CHART_PLACEHOLDER_QCOLOR, 1))
    painter.drawRect(QRectF(ml, mt, pw, ph))

    # Distortion type and magnitude label
    corner_r2 = ((img_w / 2.0) / fx) ** 2 + ((img_h / 2.0) / fy) ** 2
    corner_r4 = corner_r2 ** 2
    corner_r6 = corner_r4 * corner_r2
    max_pct = abs(k1 * corner_r2 + k2 * corner_r4 + k3 * corner_r6) * 100.0
    dist_type = "barrel" if k1 > 0 else "pincushion"
    f2 = painter.font()
    f2.setPixelSize(style.CHART_FONT_SIZE)
    f2.setBold(False)
    painter.setFont(f2)
    painter.setPen(style.CHART_AXIS_QCOLOR)
    painter.drawText(
        QRectF(ml + 4, mt + 4, pw - 4, 16),
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        f"{dist_type}  (radial {max_pct:.1f}% at corner)",
    )

    # Legend
    painter.drawText(
        QRectF(ml, mt + ph + 4, pw, mb - 4),
        Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
        "gray = ideal  |  color = distorted",
    )

    painter.end()
    return pix


# ======================================================================
# Pose diversity chart helpers (tilt scatter, distance histogram, tilt vs distance)
# ======================================================================

def _nice_step(data_range: float, target_ticks: int = 5) -> float:
    """Return a human-friendly tick step for the given data range."""
    if data_range <= 0:
        return 1.0
    raw = data_range / max(target_ticks, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 2.5, 5, 10):
        step = factor * magnitude
        if data_range / step <= target_ticks + 1:
            return step
    return magnitude * 10


def _map(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    """Linear mapping from [src_min, src_max] to [dst_min, dst_max]."""
    span = src_max - src_min
    if span == 0:
        return (dst_min + dst_max) / 2
    return dst_min + (value - src_min) / span * (dst_max - dst_min)




def _render_tilt_scatter(
    poses: List[tuple],
    width: int,
    height: int,
    title: str,
    main_color: QColor,
) -> QPixmap:
    """Scatter plot of tilt_X vs tilt_Y (degrees) for each calibration image."""
    pix, painter, ml, mb, mt, mr, pw, ph = _begin_chart(width, height, title)
    if not poses:
        return _chart_no_data(pix, painter, ml, mb, mt, mr, pw, ph)

    tilts_x = [p[0] for p in poses]
    tilts_y = [p[1] for p in poses]
    max_abs = max(max(abs(v) for v in tilts_x), max(abs(v) for v in tilts_y), 30.0)
    margin = max_abs * 0.15
    ax_min, ax_max = -(max_abs + margin), max_abs + margin
    step = _nice_step(2 * (max_abs + margin), 5)

    _set_tick_font(painter)
    tick = math.ceil(ax_min / step) * step
    while tick <= ax_max + 1e-9:
        cx = _map(tick, ax_min, ax_max, ml, ml + pw)
        cy = _map(tick, ax_min, ax_max, mt + ph, mt)
        is_zero = abs(tick) < 1e-9
        painter.setPen(QPen(QColor("#888888" if is_zero else "#aaaaaa"), 1.2 if is_zero else 0.6))
        painter.drawLine(QPointF(cx, mt), QPointF(cx, mt + ph))
        painter.drawLine(QPointF(ml, cy), QPointF(ml + pw, cy))
        painter.setPen(style.CHART_AXIS_QCOLOR)
        label = f"{tick:.0f}°"
        painter.drawText(QRectF(cx - 16, mt + ph + 2, 32, mb - 4),
                         Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, label)
        painter.drawText(QRectF(0, cy - 8, ml - 3, 16),
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, label)
        tick += step

    painter.setPen(style.CHART_AXIS_QCOLOR)
    painter.drawText(QRectF(ml, mt + ph + 14, pw, 14),
                     Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, "Tilt X (°)")

    dot_r = 4.0
    fill = QColor(main_color)
    fill.setAlpha(180)
    painter.setBrush(fill)
    painter.setPen(QPen(main_color.darker(130), 0.8))
    for tx, ty, _ in poses:
        px = _map(tx, ax_min, ax_max, ml, ml + pw)
        py = _map(ty, ax_min, ax_max, mt + ph, mt)
        painter.drawEllipse(QPointF(px, py), dot_r, dot_r)

    _end_chart(painter, ml, mb, mt, mr, pw, ph, len(poses))
    return pix


def _render_distance_histogram(
    poses: List[tuple],
    width: int,
    height: int,
    title: str,
    main_color: QColor,
    dist_unit: str = "grid sq.",
) -> QPixmap:
    """Histogram of pattern distances showing distance-coverage distribution."""
    pix, painter, ml, mb, mt, mr, pw, ph = _begin_chart(width, height, title)
    if not poses:
        return _chart_no_data(pix, painter, ml, mb, mt, mr, pw, ph)

    distances = [p[2] for p in poses]
    d_max = max(max(distances), 1.0)
    n_bins = max(5, min(20, int(math.sqrt(len(distances)))))
    bin_w = d_max / n_bins
    counts = [0] * n_bins
    for d in distances:
        counts[min(int(d / bin_w), n_bins - 1)] += 1

    max_count = max(counts) if counts else 1
    y_step = _nice_step(max_count, 4)
    y_max = math.ceil(max_count / y_step) * y_step

    _set_tick_font(painter)

    # Y grid
    y_tick = 0.0
    while y_tick <= y_max + 1e-9:
        cy = _map(y_tick, 0, y_max, mt + ph, mt)
        painter.setPen(QPen(QColor("#cccccc"), 0.6))
        painter.drawLine(QPointF(ml, cy), QPointF(ml + pw, cy))
        painter.setPen(style.CHART_AXIS_QCOLOR)
        painter.drawText(QRectF(0, cy - 8, ml - 3, 16),
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                         f"{int(y_tick)}")
        y_tick += y_step

    # X axis labels: 0, midpoint, max — full-width rects with alignment flags avoid clipping
    fmt = ".2f" if dist_unit == "m" else ".1f"
    painter.setPen(style.CHART_AXIS_QCOLOR)
    painter.drawText(QRectF(ml, mt + ph + 2, 48, mb - 4),
                     Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, f"{0:{fmt}}")
    painter.drawText(QRectF(ml, mt + ph + 2, pw, mb - 4),
                     Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, f"{d_max / 2:{fmt}}")
    painter.drawText(QRectF(ml, mt + ph + 2, pw, mb - 4),
                     Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop, f"{d_max:{fmt}}")

    # Bars
    bar_gap = max(1.0, pw / n_bins * 0.08)
    bar_pw = pw / n_bins - bar_gap
    fill = QColor(main_color)
    fill.setAlpha(110)
    for i, cnt in enumerate(counts):
        if cnt == 0:
            continue
        bx = ml + i * (pw / n_bins) + bar_gap / 2
        bar_h = _map(cnt, 0, y_max, 0, ph)
        painter.setBrush(fill)
        painter.setPen(QPen(main_color.darker(120), 0.5))
        painter.drawRect(QRectF(bx, mt + ph - bar_h, bar_pw, bar_h))

    painter.setPen(style.CHART_AXIS_QCOLOR)
    painter.drawText(QRectF(ml, mt + ph + 14, pw, 14),
                     Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                     f"Distance ({dist_unit})")

    _end_chart(painter, ml, mb, mt, mr, pw, ph, len(poses))
    return pix


def _render_tilt_vs_distance(
    poses: List[tuple],
    width: int,
    height: int,
    title: str,
    main_color: QColor,
    dist_unit: str = "grid sq.",
) -> QPixmap:
    """Scatter of tilt magnitude (°) vs. distance to check for coverage correlation."""
    pix, painter, ml, mb, mt, mr, pw, ph = _begin_chart(width, height, title)
    if not poses:
        return _chart_no_data(pix, painter, ml, mb, mt, mr, pw, ph)

    distances = [p[2] for p in poses]
    tilts = [math.sqrt(p[0] ** 2 + p[1] ** 2) for p in poses]

    d_max = max(distances) if distances else 1.0
    t_max = max(tilts) if tilts else 1.0
    ax_d_min, ax_d_max = 0.0, d_max + (d_max * 0.1 if d_max > 0 else 1.0)
    ax_t_min, ax_t_max = 0.0, max(45.0, t_max + (t_max * 0.15 if t_max > 0 else 1.0))

    _set_tick_font(painter)

    # X grid
    d_fmt = ".2f" if dist_unit == "m" else ".1f"
    d_step = _nice_step(ax_d_max - ax_d_min, 5)
    x_tick = math.ceil(ax_d_min / d_step) * d_step
    while x_tick <= ax_d_max + 1e-9:
        cx = _map(x_tick, ax_d_min, ax_d_max, ml, ml + pw)
        painter.setPen(QPen(QColor("#cccccc"), 0.6))
        painter.drawLine(QPointF(cx, mt), QPointF(cx, mt + ph))
        painter.setPen(style.CHART_AXIS_QCOLOR)
        painter.drawText(QRectF(cx - 20, mt + ph + 2, 40, mb - 4),
                         Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                         f"{x_tick:{d_fmt}}")
        x_tick += d_step

    # Y axis label (rotated)
    painter.save()
    painter.translate(10, mt + ph / 2)
    painter.rotate(-90)
    f_ax = painter.font()
    f_ax.setPixelSize(style.CHART_FONT_SIZE)
    painter.setFont(f_ax)
    painter.setPen(style.CHART_AXIS_QCOLOR)
    painter.drawText(QRectF(-ph / 2, -10, ph, 20),
                     Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, "Tilt (°)")
    painter.restore()

    # Y grid
    t_step = _nice_step(ax_t_max - ax_t_min, 4)
    y_tick = 0.0
    while y_tick <= ax_t_max + 1e-9:
        cy = _map(y_tick, ax_t_min, ax_t_max, mt + ph, mt)
        painter.setPen(QPen(QColor("#cccccc"), 0.6))
        painter.drawLine(QPointF(ml, cy), QPointF(ml + pw, cy))
        painter.setPen(style.CHART_AXIS_QCOLOR)
        painter.drawText(QRectF(0, cy - 8, ml - 3, 16),
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                         f"{y_tick:.0f}°")
        y_tick += t_step

    painter.setPen(style.CHART_AXIS_QCOLOR)
    painter.drawText(QRectF(ml, mt + ph + 14, pw, 14),
                     Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                     f"Distance ({dist_unit})")

    dot_r = 4.0
    fill = QColor(main_color)
    fill.setAlpha(180)
    painter.setBrush(fill)
    painter.setPen(QPen(main_color.darker(130), 0.8))
    for (_, _, dist), tilt in zip(poses, tilts):
        px = _map(dist, ax_d_min, ax_d_max, ml, ml + pw)
        py = _map(tilt, ax_t_min, ax_t_max, mt + ph, mt)
        painter.drawEllipse(QPointF(px, py), dot_r, dot_r)

    _end_chart(painter, ml, mb, mt, mr, pw, ph, len(poses))
    return pix
