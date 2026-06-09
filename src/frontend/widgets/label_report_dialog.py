"""Dialog displaying a label summary report for a dataset or workspace.

Includes:
- Overview panel (counts by channel, source)
- Per-class breakdown with attributes
- 2×2 chart grid (YOLO-style):
  Top-left  = class histogram (horizontal bars)
  Top-right = bbox overlay (semi-transparent rectangles)
  Bot-left  = bbox centre heatmap (50×50 grid, axis labels)
  Bot-right = bbox w×h size heatmap (50×50 grid, axis labels)
- Class selector to filter overlay + heatmaps by class
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from PyQt6.QtCore import Qt, QRectF, QSizeF, QMarginsF
from PyQt6.QtGui import QColor, QPainter, QPen, QPageLayout, QPageSize, QPixmap, QTextDocument
from PyQt6.QtPrintSupport import QPrinter
from PyQt6.QtWidgets import (
    QComboBox,
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

# ---------------------------------------------------------------------------
# Chart rendering constants
# ---------------------------------------------------------------------------
_CHART_W = style.CHART_W_GRID
_CHART_H = style.CHART_H_GRID

# Colour palette for per-class items (12 distinguishable colours)
_CLASS_COLORS = [
    QColor("#4e79a7"), QColor("#f28e2b"), QColor("#e15759"), QColor("#76b7b2"),
    QColor("#59a14f"), QColor("#edc948"), QColor("#b07aa1"), QColor("#ff9da7"),
    QColor("#9c755f"), QColor("#bab0ac"), QColor("#af7aa1"), QColor("#86bcb6"),
]


class LabelReportDialog(QDialog):
    """Shows label counts aggregated by class, channel, source, and attributes.

    Includes a 2×2 chart grid (YOLO-style): class histogram, bbox overlay,
    centre heatmap, and w×h size heatmap — all filterable by class.

    Args:
        parent: Parent widget.
        summary: Labels summary dict.
        title: Window / header title.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        summary: Optional[Dict[str, Any]] = None,
        title: str = "Label report",
    ) -> None:
        super().__init__(parent)
        self.summary: Dict[str, Any] = summary or {}
        self._title = title
        self._refresh_callback = None
        self.setWindowTitle(title)
        self.setMinimumWidth(style.DIALOG_REPORT_MIN_W)
        self._build_ui()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:  # noqa: N802
        """Ensure ESC always closes the dialog (even if a child widget has focus)."""
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(event)

    def refresh(self, summary: Dict[str, Any], title: Optional[str] = None) -> None:
        """Refresh the dialog with new summary data."""
        self.summary = summary
        if title:
            self._title = title
            self.setWindowTitle(title)
        self._rebuild_ui()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(0)

        card = QWidget(self)
        card.setObjectName("label_report_card")
        card.setStyleSheet(style.card_style("label_report_card"))
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 14, 14, 14)
        card_layout.setSpacing(12)

        # Header
        header = QLabel(self._title)
        header.setStyleSheet(style.heading_style(14))
        card_layout.addWidget(header)

        if not self.summary or not self.summary.get("total_annotations"):
            empty = QLabel("No labels found.")
            empty.setStyleSheet(f"color: {style.TEXT_CAPTION};")
            card_layout.addWidget(empty)
        else:
            # Overview panel
            card_layout.addWidget(style.section_heading_label("Overview"))
            card_layout.addWidget(self._overview_panel())

            # Charts 2×2 grid with class selector
            card_layout.addWidget(style.section_heading_label("Distribution"))
            card_layout.addWidget(self._charts_section())

            # Per-class panels
            card_layout.addWidget(style.section_heading_label("Classes"))
            card_layout.addWidget(self._classes_panel())

        card_layout.addStretch(1)

        self._card = card
        scroll = QScrollArea(self)
        scroll.setWidget(card)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        layout.addWidget(scroll)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.setContentsMargins(0, 8, 0, 0)
        btn_row.addStretch(1)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._on_refresh_clicked)
        btn_row.addWidget(refresh_btn)
        pdf_btn = QPushButton("Export PDF")
        pdf_btn.clicked.connect(self._export_pdf)
        btn_row.addWidget(pdf_btn)
        layout.addLayout(btn_row)

        self.resize(max(self.minimumWidth(), 740), 820)

    def _rebuild_ui(self) -> None:
        old_layout = self.layout()
        if old_layout:
            QWidget().setLayout(old_layout)
        self._build_ui()

    # ------------------------------------------------------------------
    # Panels
    # ------------------------------------------------------------------

    def _overview_panel(self) -> QWidget:
        s = self.summary
        panel = QWidget()
        panel.setObjectName("label_overview_panel")
        panel.setStyleSheet(style.panel_body_style("label_overview_panel"))
        form = QFormLayout(panel)
        form.setContentsMargins(10, 8, 10, 8)
        form.setVerticalSpacing(6)

        form.addRow(self._field("Total annotations"), self._value(str(s.get("total_annotations", 0))))

        imgs = s.get("images_labeled", {})
        vis_imgs = imgs.get("visible", 0)
        lwir_imgs = imgs.get("lwir", 0)
        form.addRow(self._field("Images labelled"), self._value(f"visible: {vis_imgs}  ·  lwir: {lwir_imgs}"))

        by_ch = s.get("by_channel", {})
        vis_ann = by_ch.get("visible", 0)
        lwir_ann = by_ch.get("lwir", 0)
        form.addRow(self._field("Annotations by channel"), self._value(f"visible: {vis_ann}  ·  lwir: {lwir_ann}"))

        by_src = s.get("by_source", {})
        parts = []
        for key in ("manual", "reviewed", "auto"):
            cnt = by_src.get(key, 0)
            if cnt:
                parts.append(f"{key}: {cnt}")
        form.addRow(self._field("Annotations by source"), self._value("  ·  ".join(parts) if parts else "—"))

        return panel

    # ------------------------------------------------------------------
    # Charts section — 2×2 grid
    # ------------------------------------------------------------------

    def _charts_section(self) -> QWidget:
        """Build charts panel: class selector + 2×2 chart grid."""
        wrapper, vbox = style.make_panel("label_charts_panel", margins=(8, 8, 8, 8), spacing=8)

        # Precompute sorted classes list (used by combo, histogram, colours)
        by_class = self.summary.get("by_class", {})
        self._sorted_classes: List = sorted(
            by_class.items(), key=lambda kv: kv[1].get("total", 0), reverse=True,
        )
        self._class_color_map: Dict[str, QColor] = {}
        for idx, (cid, _) in enumerate(self._sorted_classes):
            self._class_color_map[cid] = _CLASS_COLORS[idx % len(_CLASS_COLORS)]

        # Class selector
        combo_row = QHBoxLayout()
        combo_row.addWidget(self._field("Filter by class:"))
        self._chart_combo = QComboBox()
        self._chart_combo.addItem("All classes", None)
        for cid, cdata in self._sorted_classes:
            cname = cdata.get("name") or cid
            display = f"{cid}: {cname}" if cid != cname else cname
            self._chart_combo.addItem(display, cid)
        self._chart_combo.currentIndexChanged.connect(self._on_chart_filter_changed)
        combo_row.addWidget(self._chart_combo, 1)
        combo_row.addStretch()
        vbox.addLayout(combo_row)

        # 2×2 chart grid
        grid = QGridLayout()
        grid.setSpacing(12)

        # Top-left: class histogram (always shows all classes)
        self._cls_hist_label = QLabel()
        self._cls_hist_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self._cls_hist_label, 0, 0)

        # Top-right: bbox overlay
        self._bbox_overlay_label = QLabel()
        self._bbox_overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self._bbox_overlay_label, 0, 1)

        # Bottom-left: centre heatmap
        self._centre_hm_label = QLabel()
        self._centre_hm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self._centre_hm_label, 1, 0)

        # Bottom-right: w×h size heatmap
        self._size_hm_label = QLabel()
        self._size_hm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self._size_hm_label, 1, 1)

        vbox.addLayout(grid)

        # Render initial charts (all classes)
        self._render_all_charts()

        return wrapper

    def _on_chart_filter_changed(self, _index: int) -> None:
        self._render_all_charts()

    def _render_all_charts(self) -> None:
        """Re-render all four charts with current combo selection."""
        class_id = self._chart_combo.currentData()

        # Class histogram is always global
        self._cls_hist_label.setPixmap(
            _render_class_histogram(
                self._sorted_classes, self._class_color_map, _CHART_W, _CHART_H,
                highlight_class=class_id,
            )
        )

        # Filtered charts
        if class_id is None:
            charts = self.summary.get("charts", {})
            color = QColor(70, 130, 180, 35)
        else:
            cls_data = self.summary.get("by_class", {}).get(str(class_id), {})
            charts = cls_data.get("charts", {})
            base = self._class_color_map.get(str(class_id), _CLASS_COLORS[0])
            color = QColor(base.red(), base.green(), base.blue(), 35)

        samples = charts.get("bbox_samples", [])
        self._bbox_overlay_label.setPixmap(
            _render_bbox_overlay(samples, _CHART_W, _CHART_H, color)
        )

        pos_grid = charts.get("position_grid", [])
        self._centre_hm_label.setPixmap(
            _render_heatmap(pos_grid, _CHART_W, _CHART_H, "x (normalised)", "y (normalised)")
        )

        size_grid = charts.get("size_wh_grid", [])
        self._size_hm_label.setPixmap(
            _render_heatmap(size_grid, _CHART_W, _CHART_H, "width", "height")
        )

    # ------------------------------------------------------------------
    # Classes panel
    # ------------------------------------------------------------------

    def _classes_panel(self) -> QWidget:
        """Build a panel with one sub-section per class, sorted by count."""
        by_class: Dict[str, Any] = self.summary.get("by_class", {})
        sorted_classes = sorted(by_class.items(), key=lambda kv: kv[1].get("total", 0), reverse=True)

        panel, vbox = style.make_panel("label_classes_panel", margins=(10, 8, 10, 8), spacing=10)

        for class_id, cls_data in sorted_classes:
            cls_name = cls_data.get("name") or class_id
            total = cls_data.get("total", 0)

            display = f"{class_id}: {cls_name}" if class_id != cls_name else cls_name
            heading = QLabel(f"<b>{display}</b> — {total} annotations")
            heading.setTextFormat(Qt.TextFormat.RichText)
            heading.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            vbox.addWidget(heading)

            form = QFormLayout()
            form.setContentsMargins(16, 0, 0, 0)
            form.setVerticalSpacing(3)

            by_ch = cls_data.get("by_channel", {})
            ch_parts = []
            for ch in ("visible", "lwir"):
                cnt = by_ch.get(ch, 0)
                if cnt:
                    ch_parts.append(f"{ch}: {cnt}")
            if ch_parts:
                form.addRow(self._field("Channels"), self._value("  ·  ".join(ch_parts)))

            by_src = cls_data.get("by_source", {})
            src_parts = []
            for src in ("manual", "reviewed", "auto"):
                cnt = by_src.get(src, 0)
                if cnt:
                    src_parts.append(f"{src}: {cnt}")
            if src_parts:
                form.addRow(self._field("Source"), self._value("  ·  ".join(src_parts)))

            attrs: Dict[str, Dict[str, int]] = cls_data.get("attributes", {})
            for attr_name, val_counts in sorted(attrs.items()):
                sorted_vals = sorted(val_counts.items(), key=lambda kv: kv[1], reverse=True)
                vals_str = ", ".join(f"{v}: {c}" for v, c in sorted_vals)
                form.addRow(self._field(attr_name), self._value(vals_str))

            vbox.addLayout(form)

        return panel

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _field(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"font-weight: 700; color: {style.TEXT_TITLE};")
        return lbl

    def _value(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lbl.setWordWrap(True)
        return lbl

    def _on_refresh_clicked(self) -> None:
        if self._refresh_callback:
            self._refresh_callback()

    def _build_pdf_html(self) -> str:
        """Build HTML string for QTextDocument-based PDF export."""
        import base64
        from PyQt6.QtCore import QBuffer, QIODevice
        from config import APP_VERSION
        from datetime import date

        def _pix_url(pix: QPixmap) -> str:
            buf = QBuffer()
            buf.open(QIODevice.OpenModeFlag.WriteOnly)
            pix.save(buf, "PNG")
            buf.close()
            b64 = base64.b64encode(bytes(buf.data())).decode("ascii")
            return f"data:image/png;base64,{b64}"

        def _row(label: str, value: str) -> str:
            return f'<tr><td class="lbl" width="33%">{label}</td><td>{value}</td></tr>'

        def _panel(content: str) -> str:
            return (
                '<table width="100%" cellspacing="0" cellpadding="0"'
                ' style="border:1px solid #dde1e7; margin-bottom:8pt;">'
                f'<tr><td style="padding:6pt;">{content}</td></tr></table>'
            )

        # ---- header ----
        total = self.summary.get("total_annotations", 0)
        n_classes = len(self.summary.get("by_class", {}))
        meta_parts = [f"Generated: {date.today()}", f"GUI v{APP_VERSION}"]
        if self._title:
            meta_parts.append(self._title)
        if total:
            meta_parts.append(f"Total: {total}  Classes: {n_classes}")
        meta_line = " &nbsp;&middot;&nbsp; ".join(meta_parts)

        # ---- overview ----
        s = self.summary
        imgs = s.get("images_labeled", {})
        by_ch = s.get("by_channel", {})
        by_src = s.get("by_source", {})
        src_parts = [f"{k}: {v}" for k in ("manual", "reviewed", "auto") if (v := by_src.get(k, 0))]
        overview_rows = (
            _row("Total annotations", str(s.get("total_annotations", 0)))
            + _row("Images labelled", f"visible: {imgs.get('visible', 0)}  ·  lwir: {imgs.get('lwir', 0)}")
            + _row("Annotations by channel", f"visible: {by_ch.get('visible', 0)}  ·  lwir: {by_ch.get('lwir', 0)}")
            + _row("Annotations by source", "  ·  ".join(src_parts) if src_parts else "—")
        )

        # ---- charts (all classes, no filter) ----
        by_class = s.get("by_class", {})
        sorted_classes: List = sorted(by_class.items(), key=lambda kv: kv[1].get("total", 0), reverse=True)
        class_color_map: Dict[str, QColor] = {
            cid: _CLASS_COLORS[idx % len(_CLASS_COLORS)]
            for idx, (cid, _) in enumerate(sorted_classes)
        }
        charts_html = ""
        if sorted_classes:
            cw, ch = _CHART_W, _CHART_H
            disp_w = 238
            hist_pix = _render_class_histogram(sorted_classes, class_color_map, cw, ch)
            all_charts = s.get("charts", {})
            bbox_pix = _render_bbox_overlay(
                all_charts.get("bbox_samples", []), cw, ch, QColor(70, 130, 180, 35)
            )
            pos_pix = _render_heatmap(
                all_charts.get("position_grid", []), cw, ch, "x (normalised)", "y (normalised)"
            )
            size_pix = _render_heatmap(
                all_charts.get("size_wh_grid", []), cw, ch, "width", "height"
            )
            charts_html = (
                f'<table width="100%" cellspacing="0"><tr>'
                f'<td align="center"><img src="{_pix_url(hist_pix)}" width="{disp_w}"/></td>'
                f'<td align="center"><img src="{_pix_url(bbox_pix)}" width="{disp_w}"/></td>'
                f'</tr><tr>'
                f'<td align="center"><img src="{_pix_url(pos_pix)}" width="{disp_w}"/></td>'
                f'<td align="center"><img src="{_pix_url(size_pix)}" width="{disp_w}"/></td>'
                f'</tr></table>'
            )

        # ---- per-class ----
        classes_html = ""
        for class_id, cls_data in sorted_classes:
            cls_name = cls_data.get("name") or class_id
            total_cls = cls_data.get("total", 0)
            display = f"{class_id}: {cls_name}" if class_id != cls_name else str(cls_name)
            cls_rows = ""
            cls_by_ch = cls_data.get("by_channel", {})
            ch_parts = [f"{ch}: {cls_by_ch[ch]}" for ch in ("visible", "lwir") if cls_by_ch.get(ch)]
            if ch_parts:
                cls_rows += _row("Channels", "  ·  ".join(ch_parts))
            cls_by_src = cls_data.get("by_source", {})
            src_p = [f"{k}: {v}" for k in ("manual", "reviewed", "auto") if (v := cls_by_src.get(k, 0))]
            if src_p:
                cls_rows += _row("Source", "  ·  ".join(src_p))
            attrs: Dict[str, Dict[str, int]] = cls_data.get("attributes", {})
            for attr_name, val_counts in sorted(attrs.items()):
                sorted_vals = sorted(val_counts.items(), key=lambda kv: kv[1], reverse=True)
                cls_rows += _row(attr_name, ", ".join(f"{v}: {c}" for v, c in sorted_vals))
            classes_html += (
                f'<p style="font-weight:bold; margin:6pt 0 2pt 0;">'
                f'{display} — {total_cls} annotations</p>'
                f'<table class="data" width="100%" cellspacing="0" style="margin-left:12pt;">'
                f'{cls_rows}</table>'
            )

        css = (
            f"body {{ font-family: Arial, sans-serif; font-size: 9pt; color: {style.TEXT_TITLE}; }}"
            f" h2 {{ font-size: 14pt; font-weight: bold; color: {style.TEXT_TITLE}; margin: 4pt 0 10pt 0; }}"
            " h3 { font-size: 9pt; color: #5c6370; font-weight: bold; margin: 8pt 0 2pt 0; }"
            " table.data td { padding: 2pt 4pt; vertical-align: top; }"
            f" td.lbl {{ font-weight: bold; color: {style.TEXT_TITLE}; }}"
        )

        overview_section = _panel(f'<table class="data" width="100%" cellspacing="0">{overview_rows}</table>')
        charts_section = _panel(charts_html) if charts_html else ""
        classes_section = _panel(classes_html) if classes_html else ""

        return (
            f'<!DOCTYPE html><html><head><meta charset="utf-8"><style>{css}</style></head><body>'
            f'<p style="color:#9099a8; font-size:8pt; padding-bottom:3pt; margin-bottom:6pt;'
            f' border-bottom:1px solid #c0c6d0;">{meta_line}</p>'
            f'<h2>Label Report</h2>'
            f'<h3>Overview</h3>{overview_section}'
            f'<h3>Distribution</h3>{charts_section}'
            f'<h3>Classes</h3>{classes_section}'
            f'</body></html>'
        )

    def _export_pdf(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export PDF", "label_report.pdf", "PDF Files (*.pdf)")
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


# ======================================================================
# Chart rendering — pure QPainter on QPixmap (no matplotlib dependency)
# ======================================================================

def _make_pixmap(w: int, h: int) -> QPixmap:
    """Create a HiDPI-ready pixmap filled with background colour."""
    pix = QPixmap(w * style.CHART_DPR, h * style.CHART_DPR)
    pix.setDevicePixelRatio(style.CHART_DPR)
    pix.fill(style.CHART_BG_QCOLOR)
    return pix


def _setup_painter(pix: QPixmap) -> QPainter:
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    f = p.font()
    f.setPixelSize(style.CHART_FONT_SIZE)
    p.setFont(f)
    return p


def _draw_title(p: QPainter, title: str, width: int) -> None:
    """Draw a centred title in the top margin."""
    f = p.font()
    f.setPixelSize(style.CHART_TITLE_FONT_SIZE)
    f.setBold(True)
    p.setFont(f)
    p.setPen(style.CHART_AXIS_QCOLOR)
    p.drawText(QRectF(0, 2, width, style.CHART_MARGINS[2] - 2),
               Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom, title)
    f.setBold(False)
    f.setPixelSize(style.CHART_FONT_SIZE)
    p.setFont(f)


def _draw_axes(p: QPainter, w: int, h: int,
               x_label: str, y_label: str,
               n_ticks: int = 6) -> None:
    """Draw axes, tick marks (0.0 … 1.0), and axis labels."""
    ml, mb, mt, mr = style.CHART_MARGINS
    pw = w - ml - mr
    ph = h - mt - mb

    pen = QPen(style.CHART_AXIS_QCOLOR, 1)
    p.setPen(pen)
    p.drawLine(ml, mt, ml, h - mb)
    p.drawLine(ml, h - mb, w - mr, h - mb)

    for i in range(n_ticks):
        t = i / (n_ticks - 1)
        # X ticks
        x = ml + t * pw
        p.drawLine(int(x), h - mb, int(x), h - mb + 3)
        p.drawText(QRectF(x - 16, h - mb + 4, 32, 14),
                   Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                   f"{t:.1f}")
        # Y ticks  (bottom=0, top=1 for heatmaps)
        y = h - mb - t * ph
        p.drawLine(ml - 3, int(y), ml, int(y))
        p.drawText(QRectF(0, y - 7, ml - 5, 14),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   f"{t:.1f}")

    # Axis labels
    p.drawText(QRectF(ml, h - 10, pw, 12),
               Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, x_label)
    # Y label (drawn vertically would need transform — keep horizontal short)
    p.save()
    p.translate(8, mt + ph / 2)
    p.rotate(-90)
    p.drawText(QRectF(-ph / 2, -8, ph, 16),
               Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, y_label)
    p.restore()


# ------------------------------------------------------------------
# 1. Class histogram (horizontal bars)
# ------------------------------------------------------------------

def _render_class_histogram(
    sorted_classes: List,
    color_map: Dict[str, QColor],
    width: int, height: int,
    highlight_class: Optional[str] = None,
) -> QPixmap:
    """Horizontal bar chart of annotation counts per class."""
    pix = _make_pixmap(width, height)
    if not sorted_classes:
        return pix

    p = _setup_painter(pix)
    _draw_title(p, "Labels per class", width)

    ml, mb, mt, mr = style.CHART_MARGINS  # identical margins to the other three charts
    pw = width - ml - mr
    ph = height - mt - mb

    p.fillRect(QRectF(ml, mt, pw, ph), style.CHART_PLOT_BG_QCOLOR)

    n = len(sorted_classes)
    max_count = max(cd.get("total", 0) for _, cd in sorted_classes) or 1
    bar_h = max(2, min(20, (ph - 4) / n - 2))
    gap = (ph - n * bar_h) / max(n, 1)

    for i, (cid, cdata) in enumerate(sorted_classes):
        cname = cdata.get("name") or cid
        total = cdata.get("total", 0)
        y = mt + i * (bar_h + gap) + gap / 2
        bw = (total / max_count) * pw if max_count else 0

        color = color_map.get(cid, _CLASS_COLORS[0])
        if highlight_class is not None and str(cid) != str(highlight_class):
            color = QColor(color.red(), color.green(), color.blue(), 80)

        p.fillRect(QRectF(ml, y, bw, bar_h), color)

        # Label: "id: name (count)" inside the bar when wide enough, else after it
        label = f"{cid}: {cname}" if cid != cname else str(cname)
        if len(label) > 14:
            label = label[:13] + "…"
        count_str = str(total)
        if bw >= pw * 0.35:
            # Wide bar: name left-inside (white), count right-inside (white)
            p.setPen(QColor("white"))
            p.drawText(QRectF(ml + 3, y, bw - 6, bar_h),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)
            p.drawText(QRectF(ml + 3, y, bw - 6, bar_h),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, count_str)
        else:
            # Narrow bar: name + count to the right of bar
            p.setPen(style.CHART_AXIS_QCOLOR)
            p.drawText(QRectF(ml + bw + 3, y, pw - bw - 3, bar_h),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       f"{label} {count_str}")

    # Border (matches the other three charts)
    p.setPen(QPen(style.CHART_PLACEHOLDER_QCOLOR, 1))
    p.drawRect(QRectF(ml, mt, pw, ph))

    p.end()
    return pix


# ------------------------------------------------------------------
# 2. Bbox overlay (semi-transparent rectangles on normalised canvas)
# ------------------------------------------------------------------

def _render_bbox_overlay(
    samples: Sequence[Sequence[float]],
    width: int, height: int,
    box_color: QColor,
) -> QPixmap:
    """Draw sampled bboxes as semi-transparent rectangles on a [0,1]² canvas."""
    pix = _make_pixmap(width, height)
    p = _setup_painter(pix)
    _draw_title(p, "Bounding boxes", width)

    ml, mb, mt, mr = style.CHART_MARGINS
    pw = width - ml - mr
    ph = height - mt - mb

    p.fillRect(QRectF(ml, mt, pw, ph), style.CHART_PLOT_BG_QCOLOR)

    # Axes with ticks
    _draw_axes(p, width, height, "x", "y")

    if samples:
        fill = QColor(box_color)
        stroke = QColor(fill.red(), fill.green(), fill.blue(), min(fill.alpha() + 40, 180))
        pen = QPen(stroke, 0.7)

        for entry in samples:
            if len(entry) < 4:
                continue
            cx, cy, bw, bh = entry[0], entry[1], entry[2], entry[3]
            x = ml + (cx - bw / 2) * pw
            y = mt + (cy - bh / 2) * ph
            rw = bw * pw
            rh = bh * ph
            p.setPen(pen)
            p.fillRect(QRectF(x, y, rw, rh), fill)
            p.drawRect(QRectF(x, y, rw, rh))

    # Border
    p.setPen(QPen(style.CHART_PLACEHOLDER_QCOLOR, 1))
    p.drawRect(QRectF(ml, mt, pw, ph))

    p.end()
    return pix


# ------------------------------------------------------------------
# 3 & 4. Generic heatmap (used for centre positions AND w×h sizes)
# ------------------------------------------------------------------

def _render_heatmap(
    grid: Sequence[Sequence[int]],
    width: int, height: int,
    x_label: str, y_label: str,
) -> QPixmap:
    """Render a 2-D heatmap with axis labels (0.0–1.0)."""
    pix = _make_pixmap(width, height)
    p = _setup_painter(pix)

    # Title from axis labels
    title = f"{y_label} vs {x_label}" if y_label != x_label else y_label
    _draw_title(p, title, width)

    ml, mb, mt, mr = style.CHART_MARGINS
    pw = width - ml - mr
    ph = height - mt - mb

    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    if rows == 0 or cols == 0:
        p.setPen(style.CHART_AXIS_QCOLOR)
        p.drawText(QRectF(ml, mt, pw, ph),
                   Qt.AlignmentFlag.AlignCenter, "No data")
        p.end()
        return pix

    max_val = max((max(row) for row in grid), default=0)
    max_val = max(max_val, 1)

    cell_w = pw / cols
    cell_h = ph / rows

    p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    for r in range(rows):
        for c in range(cols):
            t = grid[r][c] / max_val
            color = _heatmap_color(t)
            p.fillRect(QRectF(ml + c * cell_w, mt + r * cell_h,
                              cell_w + 0.5, cell_h + 0.5), color)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    # Draw axes and ticks on top of heatmap
    _draw_axes(p, width, height, x_label, y_label)

    # Border
    p.setPen(QPen(style.CHART_PLACEHOLDER_QCOLOR, 1))
    p.drawRect(QRectF(ml, mt, pw, ph))

    p.end()
    return pix


# ------------------------------------------------------------------
# Heatmap colour ramp: transparent white → yellow → orange → red → dark
# ------------------------------------------------------------------

def _heatmap_color(t: float) -> QColor:
    """Map *t* ∈ [0, 1] to a heat colour (white → yellow → orange → red → dark)."""
    t = max(0.0, min(1.0, t))
    if t == 0.0:
        return QColor(250, 250, 250)
    if t < 0.25:
        f = t / 0.25
        return QColor(255, 255, int(240 - 100 * f))
    if t < 0.50:
        f = (t - 0.25) / 0.25
        return QColor(255, int(240 - 60 * f), int(140 - 90 * f))
    if t < 0.75:
        f = (t - 0.50) / 0.25
        return QColor(255, int(180 - 120 * f), int(50 - 50 * f))
    f = (t - 0.75) / 0.25
    return QColor(int(255 - 80 * f), int(60 - 60 * f), 0)
