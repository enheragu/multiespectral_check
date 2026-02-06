"""Dialog for editing label attributes.

Provides a dialog to edit annotation attributes including:
- Class selection (with autocomplete)
- Bounding box corners (P1=top-left, P2=bottom-right) with live preview
- Universal attributes (source, confidence, occlusion, truncation)
- Class-specific attributes (dynamically generated based on selected class)
- Advanced attributes fallback (YAML text for unsupported attribute types)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from backend.services.labels.label_types import (
        Annotation,
        AttributeDefinition,
        ClassDefinition,
        LabelConfig,
    )

import yaml


class LabelEditDialog(QDialog):
    """Dialog for editing annotation class and attributes."""

    # Threshold for auto-detecting truncation at image edges
    EDGE_THRESHOLD = 0.02  # 2% of image dimension

    # Signal emitted when bbox changes (for live preview)
    # Emits: (x1, y1, x2, y2) as normalized corner coordinates
    bboxChanged = pyqtSignal(float, float, float, float)

    def __init__(
        self,
        parent: Optional[QWidget],
        config: Optional["LabelConfig"],
        annotation: Optional["Annotation"] = None,
        bbox: Optional[tuple] = None,
        is_new: bool = False,
        image_size: Optional[tuple] = None,
        on_bbox_changed: Optional[Callable[[float, float, float, float], None]] = None,
    ):
        """Initialize the label edit dialog.

        Args:
            parent: Parent widget
            config: Label configuration with class definitions
            annotation: Existing annotation to edit (None for new)
            bbox: Bounding box (x_center, y_center, width, height) for new annotations
            is_new: True if creating new annotation, False if editing
            image_size: Original image size (width, height) for pixel display
            on_bbox_changed: Callback for live bbox preview (receives x1, y1, x2, y2 normalized)
        """
        super().__init__(parent)
        self.config = config
        self.annotation = annotation
        self.bbox = bbox or (annotation.bbox if annotation else (0.5, 0.5, 0.1, 0.1))
        self.is_new = is_new
        self.image_size = image_size or (1920, 1080)  # Default assumption
        self._on_bbox_changed = on_bbox_changed

        self._class_widgets: Dict[str, QWidget] = {}
        self._universal_widgets: Dict[str, QWidget] = {}
        self._class_attr_widgets: Dict[str, QWidget] = {}
        self._advanced_text: Optional[QPlainTextEdit] = None
        self._class_attrs_layout: Optional[QFormLayout] = None
        self._class_attrs_group: Optional[QGroupBox] = None
        self._advanced_group: Optional[QGroupBox] = None

        # Bbox corner spinboxes (P1=top-left, P2=bottom-right)
        self._p1_x_spin: Optional[QSpinBox] = None
        self._p1_y_spin: Optional[QSpinBox] = None
        self._p2_x_spin: Optional[QSpinBox] = None
        self._p2_y_spin: Optional[QSpinBox] = None

        # Block signals during setup
        self._setup_in_progress = True
        self._setup_ui()
        self._populate_from_annotation()
        self._setup_in_progress = False

        # Trigger initial class change to populate attrs
        self._on_class_changed(self._class_combo.currentText())

        # Emit initial bbox for preview
        self._emit_bbox_preview()

        # Adjust dialog size to content
        self.adjustSize()

    def _setup_ui(self) -> None:
        """Build the dialog UI."""
        self.setWindowTitle("Edit Label" if not self.is_new else "New Label")
        # Let dialog size to content, with reasonable minimums
        self.setMinimumWidth(450)
        self.setMinimumHeight(400)
        # Allow dialog to expand as needed
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)

        # === Class Selection ===
        class_group = QGroupBox("Class")
        class_layout = QFormLayout(class_group)
        class_layout.setContentsMargins(8, 12, 8, 8)

        self._class_combo = QComboBox()
        self._class_combo.setEditable(True)
        self._class_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)

        # Populate with classes from config
        if self.config:
            choices = []
            for cls_id, cls_def in sorted(self.config.classes.items(), key=lambda x: int(x[0])):
                choices.append(f"{cls_id}: {cls_def.name}")
            self._class_combo.addItems(choices)

            # Setup autocomplete
            completer = self._class_combo.completer()
            if completer:
                completer.setFilterMode(Qt.MatchFlag.MatchContains)
                completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
                completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)

        class_layout.addRow("Class:", self._class_combo)
        content_layout.addWidget(class_group)

        # === Bounding Box (Corner-based) ===
        bbox_group = QGroupBox("Bounding Box")
        bbox_layout = QVBoxLayout(bbox_group)
        bbox_layout.setContentsMargins(8, 12, 8, 8)
        bbox_layout.setSpacing(4)

        # Convert center format to corners
        x_center, y_center, width, height = self.bbox
        img_w, img_h = self.image_size
        x1 = int((x_center - width / 2) * img_w)
        y1 = int((y_center - height / 2) * img_h)
        x2 = int((x_center + width / 2) * img_w)
        y2 = int((y_center + height / 2) * img_h)

        # P1 (Top-Left Corner)
        p1_layout = QHBoxLayout()
        p1_layout.setSpacing(4)
        p1_label = QLabel("P1 (top-left):")
        p1_label.setFixedWidth(85)
        p1_layout.addWidget(p1_label)

        p1_x_label = QLabel("x:")
        p1_x_label.setFixedWidth(12)
        p1_layout.addWidget(p1_x_label)
        self._p1_x_spin = QSpinBox()
        self._p1_x_spin.setRange(0, img_w - 1)
        self._p1_x_spin.setValue(max(0, x1))
        self._p1_x_spin.setFixedWidth(70)
        self._p1_x_spin.valueChanged.connect(self._on_bbox_spin_changed)
        p1_layout.addWidget(self._p1_x_spin)

        p1_y_label = QLabel("y:")
        p1_y_label.setFixedWidth(12)
        p1_layout.addWidget(p1_y_label)
        self._p1_y_spin = QSpinBox()
        self._p1_y_spin.setRange(0, img_h - 1)
        self._p1_y_spin.setValue(max(0, y1))
        self._p1_y_spin.setFixedWidth(70)
        self._p1_y_spin.valueChanged.connect(self._on_bbox_spin_changed)
        p1_layout.addWidget(self._p1_y_spin)
        p1_layout.addStretch()

        bbox_layout.addLayout(p1_layout)

        # P2 (Bottom-Right Corner)
        p2_layout = QHBoxLayout()
        p2_layout.setSpacing(4)
        p2_label = QLabel("P2 (bottom-right):")
        p2_label.setFixedWidth(85)
        p2_layout.addWidget(p2_label)

        p2_x_label = QLabel("x:")
        p2_x_label.setFixedWidth(12)
        p2_layout.addWidget(p2_x_label)
        self._p2_x_spin = QSpinBox()
        self._p2_x_spin.setRange(1, img_w)
        self._p2_x_spin.setValue(min(img_w, x2))
        self._p2_x_spin.setFixedWidth(70)
        self._p2_x_spin.valueChanged.connect(self._on_bbox_spin_changed)
        p2_layout.addWidget(self._p2_x_spin)

        p2_y_label = QLabel("y:")
        p2_y_label.setFixedWidth(12)
        p2_layout.addWidget(p2_y_label)
        self._p2_y_spin = QSpinBox()
        self._p2_y_spin.setRange(1, img_h)
        self._p2_y_spin.setValue(min(img_h, y2))
        self._p2_y_spin.setFixedWidth(70)
        self._p2_y_spin.valueChanged.connect(self._on_bbox_spin_changed)
        p2_layout.addWidget(self._p2_y_spin)
        p2_layout.addStretch()

        bbox_layout.addLayout(p2_layout)

        # Size info label
        self._size_label = QLabel()
        self._size_label.setStyleSheet("color: #666; font-size: 11px;")
        self._update_size_label()
        bbox_layout.addWidget(self._size_label)

        content_layout.addWidget(bbox_group)

        # === Universal Attributes ===
        universal_group = QGroupBox("Attributes")
        universal_layout = QFormLayout(universal_group)
        universal_layout.setContentsMargins(8, 12, 8, 8)
        universal_layout.setSpacing(4)

        # Source (read-only) + Confidence (read-only) on same row
        info_layout = QHBoxLayout()
        self._source_label = QLabel("manual")
        self._source_label.setStyleSheet("color: #666; font-style: italic;")
        info_layout.addWidget(QLabel("Source:"))
        info_layout.addWidget(self._source_label)
        info_layout.addSpacing(20)
        info_layout.addWidget(QLabel("Conf:"))
        self._confidence_label = QLabel("1.00")
        self._confidence_label.setStyleSheet("color: #666; font-style: italic;")
        info_layout.addWidget(self._confidence_label)
        info_layout.addStretch()
        universal_layout.addRow(info_layout)
        self._universal_widgets["source"] = self._source_label
        self._universal_widgets["confidence"] = self._confidence_label

        # Occlusion + Truncation on same row
        occlusion_trunc_layout = QHBoxLayout()
        occlusion_trunc_layout.addWidget(QLabel("Occlusion:"))
        self._occlusion_combo = QComboBox()
        self._occlusion_combo.addItems(["none", "partial", "heavy"])
        self._occlusion_combo.setFixedWidth(80)
        occlusion_trunc_layout.addWidget(self._occlusion_combo)
        occlusion_trunc_layout.addSpacing(10)
        self._truncation_check = QCheckBox("Truncated")
        occlusion_trunc_layout.addWidget(self._truncation_check)
        self._truncation_auto_label = QLabel("")
        self._truncation_auto_label.setStyleSheet("color: #888; font-size: 10px;")
        occlusion_trunc_layout.addWidget(self._truncation_auto_label)
        occlusion_trunc_layout.addStretch()
        universal_layout.addRow(occlusion_trunc_layout)
        self._universal_widgets["occlusion"] = self._occlusion_combo
        self._universal_widgets["truncation"] = self._truncation_check

        # Check if bbox touches edges and auto-set truncation
        self._check_auto_truncation()

        content_layout.addWidget(universal_group)

        # === Class-Specific Attributes (dynamic) ===
        self._class_attrs_group = QGroupBox("Class Attributes")
        self._class_attrs_layout = QFormLayout(self._class_attrs_group)
        self._class_attrs_layout.setContentsMargins(8, 12, 8, 8)
        self._class_attrs_layout.setSpacing(4)
        self._class_attrs_group.setVisible(False)  # Hidden until class with attrs selected
        content_layout.addWidget(self._class_attrs_group)

        # === Advanced Attributes (YAML fallback) ===
        self._advanced_group = QGroupBox("Advanced (YAML)")
        self._advanced_group.setCheckable(True)
        self._advanced_group.setChecked(False)
        advanced_layout = QVBoxLayout(self._advanced_group)
        advanced_layout.setContentsMargins(8, 12, 8, 8)

        self._advanced_text = QPlainTextEdit()
        self._advanced_text.setPlaceholderText("# custom_attr: value")
        self._advanced_text.setMaximumHeight(80)
        advanced_layout.addWidget(self._advanced_text)

        content_layout.addWidget(self._advanced_group)
        content_layout.addStretch()

        scroll.setWidget(content)
        layout.addWidget(scroll)

        # === Dialog Buttons ===
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Connect class change AFTER setup to avoid spurious updates
        self._class_combo.currentTextChanged.connect(self._on_class_changed)

    def _update_size_label(self) -> None:
        """Update the size info label with current dimensions."""
        if not self._p1_x_spin or not self._p2_x_spin:
            return
        w = abs(self._p2_x_spin.value() - self._p1_x_spin.value())
        h = abs(self._p2_y_spin.value() - self._p1_y_spin.value())
        self._size_label.setText(f"Size: {w} × {h} px")

    def _on_bbox_spin_changed(self) -> None:
        """Handle bbox spinbox value changes."""
        if self._setup_in_progress:
            return

        # Update size label
        self._update_size_label()

        # Update internal bbox (convert corners to center format)
        self._update_bbox_from_corners()

        # Update truncation auto-detection
        self._check_auto_truncation()

        # Emit preview
        self._emit_bbox_preview()

    def _update_bbox_from_corners(self) -> None:
        """Update internal bbox from corner spinbox values."""
        img_w, img_h = self.image_size

        # Get corner values (ensure P1 < P2)
        x1 = min(self._p1_x_spin.value(), self._p2_x_spin.value())
        y1 = min(self._p1_y_spin.value(), self._p2_y_spin.value())
        x2 = max(self._p1_x_spin.value(), self._p2_x_spin.value())
        y2 = max(self._p1_y_spin.value(), self._p2_y_spin.value())

        # Convert to normalized center format
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        x_center = (x1 + x2) / 2 / img_w
        y_center = (y1 + y2) / 2 / img_h

        self.bbox = (x_center, y_center, max(0.001, width), max(0.001, height))

    def _emit_bbox_preview(self) -> None:
        """Emit bbox change signal for live preview."""
        img_w, img_h = self.image_size

        # Get corner coordinates normalized
        x1 = self._p1_x_spin.value() / img_w
        y1 = self._p1_y_spin.value() / img_h
        x2 = self._p2_x_spin.value() / img_w
        y2 = self._p2_y_spin.value() / img_h

        # Emit signal
        self.bboxChanged.emit(x1, y1, x2, y2)

        # Call callback if provided
        if self._on_bbox_changed:
            self._on_bbox_changed(x1, y1, x2, y2)

    def _check_auto_truncation(self) -> None:
        """Check if bbox touches image edges and suggest truncation."""
        if not self.bbox:
            return

        x_center, y_center, width, height = self.bbox
        left = x_center - width / 2
        right = x_center + width / 2
        top = y_center - height / 2
        bottom = y_center + height / 2

        at_edge = (
            left <= self.EDGE_THRESHOLD
            or right >= (1.0 - self.EDGE_THRESHOLD)
            or top <= self.EDGE_THRESHOLD
            or bottom >= (1.0 - self.EDGE_THRESHOLD)
        )

        if at_edge and self.is_new:
            self._truncation_check.setChecked(True)
            self._truncation_auto_label.setText("(auto: bbox at image edge)")
        elif at_edge:
            self._truncation_auto_label.setText("(bbox at image edge)")
        else:
            self._truncation_auto_label.setText("")

    def _populate_from_annotation(self) -> None:
        """Populate dialog fields from existing annotation."""
        if not self.annotation:
            return

        # Set class
        cls_id = self.annotation.class_id
        if self.config and cls_id in self.config.classes:
            cls_def = self.config.classes[cls_id]
            text = f"{cls_id}: {cls_def.name}"
            idx = self._class_combo.findText(text)
            if idx >= 0:
                self._class_combo.setCurrentIndex(idx)

        # Set source (read-only)
        source = self.annotation.source.value if hasattr(self.annotation.source, 'value') else str(self.annotation.source)
        self._source_label.setText(source)

        # Set confidence (read-only)
        self._confidence_label.setText(f"{self.annotation.confidence:.2f}")

        # Set attributes from annotation
        attrs = self.annotation.attributes or {}

        # Occlusion
        occlusion = attrs.get("occlusion", "none")
        idx = self._occlusion_combo.findText(str(occlusion))
        if idx >= 0:
            self._occlusion_combo.setCurrentIndex(idx)

        # Truncation
        truncation = attrs.get("truncation", False)
        if truncation:
            self._truncation_check.setChecked(True)
            self._truncation_auto_label.setText("")  # Clear auto label if manually set

    def _populate_class_attrs(self, attrs: Dict[str, Any]) -> None:
        """Populate class-specific attribute widgets."""
        for attr_name, widget in self._class_attr_widgets.items():
            value = attrs.get(attr_name)
            if value is None:
                continue

            if isinstance(widget, QComboBox):
                idx = widget.findText(str(value))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))

    def _populate_advanced_attrs(self, attrs: Dict[str, Any]) -> None:
        """Put non-standard attributes in advanced YAML section."""
        if not self._advanced_text:
            return

        # Collect attrs not handled by standard widgets
        handled = {"occlusion", "truncation", "source", "confidence"}
        handled.update(self._class_attr_widgets.keys())

        extra = {k: v for k, v in attrs.items() if k not in handled}

        if extra:
            self._advanced_text.setPlainText(yaml.dump(extra, default_flow_style=False))
            # Expand the advanced section if there's content
            if self._advanced_group:
                self._advanced_group.setChecked(True)

    def _pre_populate_advanced_with_schema(self, cls_def: "ClassDefinition") -> None:
        """Pre-populate advanced section with schema attribute names as template."""
        if not self._advanced_text or not cls_def:
            return

        # Only pre-populate if no existing content and no class widgets created
        current_text = self._advanced_text.toPlainText().strip()
        if current_text or self._class_attr_widgets:
            return

        # Build template from class attributes
        template_lines = ["# Class-specific attributes (uncomment and fill):"]
        for attr_name, attr_def in cls_def.attributes.items():
            if attr_def.options:
                options_str = ", ".join(attr_def.options)
                template_lines.append(f"# {attr_name}: {attr_def.options[0]}  # options: {options_str}")
            elif attr_def.attr_type.value == "bool":
                template_lines.append(f"# {attr_name}: false")
            elif attr_def.attr_type.value == "float":
                range_str = f"range: {attr_def.range}" if attr_def.range else ""
                template_lines.append(f"# {attr_name}: 0.0  # {range_str}")
            else:
                template_lines.append(f"# {attr_name}: ")

        if len(template_lines) > 1:
            self._advanced_text.setPlainText("\n".join(template_lines))

    def _on_class_changed(self, text: str) -> None:
        """Handle class selection change - update class-specific attributes."""
        if self._setup_in_progress:
            return

        # Clear existing class attribute widgets
        self._clear_class_attrs()

        if not self.config or not text:
            self._class_attrs_group.setVisible(False)
            return

        # Parse class ID from "id: name" format
        cls_id = text.split(":")[0].strip() if ":" in text else text.strip()
        cls_def = self.config.classes.get(cls_id)

        if not cls_def:
            self._class_attrs_group.setVisible(False)
            return

        if not cls_def.attributes:
            self._class_attrs_group.setVisible(False)
            # Pre-populate advanced section with schema template
            self._pre_populate_advanced_with_schema(cls_def)
            return

        # Build widgets for class-specific attributes
        self._class_attrs_group.setVisible(True)
        self._class_attrs_group.setTitle(f"Attributes: {cls_def.name}")

        for attr_name, attr_def in cls_def.attributes.items():
            widget = self._create_attr_widget(attr_def)
            if widget:
                label_text = attr_name.replace('_', ' ').title() + ":"
                self._class_attrs_layout.addRow(label_text, widget)
                self._class_attr_widgets[attr_name] = widget

        # Populate from existing annotation if editing
        if self.annotation and self.annotation.attributes:
            self._populate_class_attrs(self.annotation.attributes)
            self._populate_advanced_attrs(self.annotation.attributes)

    def _clear_class_attrs(self) -> None:
        """Remove all class-specific attribute widgets."""
        for widget in self._class_attr_widgets.values():
            widget.deleteLater()
        self._class_attr_widgets.clear()

        # Clear the layout
        if self._class_attrs_layout:
            while self._class_attrs_layout.count():
                item = self._class_attrs_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

    def _create_attr_widget(self, attr_def: "AttributeDefinition") -> Optional[QWidget]:
        """Create appropriate widget for an attribute definition."""
        from backend.services.labels.label_types import AttributeType

        if attr_def.attr_type == AttributeType.ENUM and attr_def.options:
            combo = QComboBox()
            combo.addItems(attr_def.options)
            return combo

        elif attr_def.attr_type == AttributeType.BOOL:
            check = QCheckBox()
            return check

        elif attr_def.attr_type == AttributeType.FLOAT:
            spin = QDoubleSpinBox()
            if attr_def.range:
                spin.setRange(attr_def.range[0], attr_def.range[1])
            else:
                spin.setRange(0.0, 1.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.1)
            return spin

        return None

    def get_class_id(self) -> Optional[str]:
        """Get the selected class ID."""
        text = self._class_combo.currentText()
        if ":" in text:
            return text.split(":")[0].strip()
        return text.strip() if text else None

    def get_bbox(self) -> tuple:
        """Get the current bounding box (normalized center format)."""
        # Ensure bbox is updated from spinboxes
        self._update_bbox_from_corners()
        return self.bbox

    def get_bbox_corners(self) -> tuple:
        """Get bbox as normalized corner coordinates (x1, y1, x2, y2)."""
        img_w, img_h = self.image_size
        x1 = self._p1_x_spin.value() / img_w
        y1 = self._p1_y_spin.value() / img_h
        x2 = self._p2_x_spin.value() / img_w
        y2 = self._p2_y_spin.value() / img_h
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    def get_attributes(self) -> Dict[str, Any]:
        """Get all attribute values from the dialog."""
        attrs: Dict[str, Any] = {}

        # Universal editable attributes
        attrs["occlusion"] = self._occlusion_combo.currentText()
        attrs["truncation"] = self._truncation_check.isChecked()

        # Class-specific attributes
        for attr_name, widget in self._class_attr_widgets.items():
            if isinstance(widget, QComboBox):
                attrs[attr_name] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                attrs[attr_name] = widget.isChecked()
            elif isinstance(widget, QDoubleSpinBox):
                attrs[attr_name] = widget.value()

        # Advanced YAML attributes
        if self._advanced_text:
            yaml_text = self._advanced_text.toPlainText().strip()
            if yaml_text:
                try:
                    extra = yaml.safe_load(yaml_text)
                    if isinstance(extra, dict):
                        # Only include non-commented values
                        attrs.update(extra)
                except yaml.YAMLError:
                    pass  # Ignore invalid YAML

        return attrs

    @classmethod
    def edit_annotation(
        cls,
        parent: QWidget,
        config: Optional["LabelConfig"],
        annotation: "Annotation",
        image_size: Optional[tuple] = None,
        on_bbox_changed: Optional[Callable[[float, float, float, float], None]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Show dialog to edit an existing annotation.

        Returns:
            Dict with 'class_id', 'bbox', and 'attributes' if accepted, None if cancelled
        """
        dialog = cls(
            parent, config,
            annotation=annotation,
            is_new=False,
            image_size=image_size,
            on_bbox_changed=on_bbox_changed,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return {
                "class_id": dialog.get_class_id(),
                "bbox": dialog.get_bbox(),
                "attributes": dialog.get_attributes(),
            }
        return None

    @classmethod
    def new_annotation(
        cls,
        parent: QWidget,
        config: Optional["LabelConfig"],
        bbox: tuple,
        image_size: Optional[tuple] = None,
        on_bbox_changed: Optional[Callable[[float, float, float, float], None]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Show dialog to create a new annotation.

        Returns:
            Dict with 'class_id', 'bbox', and 'attributes' if accepted, None if cancelled
        """
        dialog = cls(
            parent, config,
            bbox=bbox,
            is_new=True,
            image_size=image_size,
            on_bbox_changed=on_bbox_changed,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return {
                "class_id": dialog.get_class_id(),
                "bbox": dialog.get_bbox(),
                "attributes": dialog.get_attributes(),
            }
        return None
