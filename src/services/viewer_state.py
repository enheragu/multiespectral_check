"""Mutable state container for per-dataset viewer data."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, List, Tuple

from PyQt6.QtGui import QPixmap

from utils.duplicates import SignatureCache


@dataclass
class ViewerState:
    pixmap_cache: Dict[str, Dict[str, Optional[QPixmap]]] = field(default_factory=dict)
    marked_for_delete: Dict[str, str] = field(default_factory=dict)
    auto_override: Set[str] = field(default_factory=set)
    mark_reason_counts: Dict[str, int] = field(default_factory=dict)
    signatures: SignatureCache = field(default_factory=dict)
    calibration_marked: Set[str] = field(default_factory=set)
    calibration_outliers: Set[str] = field(default_factory=set)
    calibration_results: Dict[str, Dict[str, Optional[bool]]] = field(default_factory=dict)
    calibration_corners: Dict[str, Dict[str, Optional[List[Tuple[float, float]]]]] = field(default_factory=dict)
    calibration_warnings: Dict[str, Dict[str, str]] = field(default_factory=dict)
    calibration_reproj_errors: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {"lwir": {}, "visible": {}}
    )
    extrinsic_pair_errors: Dict[str, float] = field(default_factory=dict)
    calibration_detection_bins: Dict[str, str] = field(default_factory=dict)
    calibration_detection_counts: Dict[str, int] = field(
        default_factory=lambda: {"both": 0, "partial": 0, "missing": 0}
    )
    calibration_suspect_bases: Set[str] = field(default_factory=set)
    calibration_matrices: Dict[str, Optional[Dict[str, Any]]] = field(
        default_factory=lambda: {"lwir": None, "visible": None}
    )
    calibration_extrinsic: Optional[Dict[str, Any]] = None

    def reset(self) -> None:
        self.pixmap_cache.clear()
        self.marked_for_delete.clear()
        self.mark_reason_counts.clear()
        self.auto_override.clear()
        self.signatures.clear()
        self.calibration_marked.clear()
        self.calibration_outliers.clear()
        self.calibration_results.clear()
        self.calibration_corners.clear()
        self.calibration_warnings.clear()
        self.calibration_reproj_errors = {"lwir": {}, "visible": {}}
        self.extrinsic_pair_errors.clear()
        self.calibration_detection_bins.clear()
        self.calibration_detection_counts = {"both": 0, "partial": 0, "missing": 0}
        self.calibration_suspect_bases.clear()
        self.calibration_matrices = {"lwir": None, "visible": None}
        self.calibration_extrinsic = None

    def rebuild_reason_counts(self) -> None:
        self.mark_reason_counts.clear()
        for reason in self.marked_for_delete.values():
            self._adjust_reason_count(reason, 1)

    def rebuild_calibration_summary(self) -> None:
        self.calibration_detection_bins.clear()
        self.calibration_detection_counts = {"both": 0, "partial": 0, "missing": 0}
        self.calibration_suspect_bases.clear()
        for base in self.calibration_marked:
            self._apply_calibration_summary_delta(base)

    def _adjust_reason_count(self, reason: Optional[str], delta: int) -> None:
        if not reason:
            return
        next_value = self.mark_reason_counts.get(reason, 0) + delta
        if next_value <= 0:
            self.mark_reason_counts.pop(reason, None)
        else:
            self.mark_reason_counts[reason] = next_value

    def refresh_calibration_entry(self, base: str) -> None:
        if base not in self.calibration_marked:
            self.remove_calibration_entry(base)
            return
        self._apply_calibration_summary_delta(base)

    def remove_calibration_entry(self, base: str) -> None:
        detected = self.calibration_detection_bins.pop(base, None)
        if detected:
            self.calibration_detection_counts[detected] = max(0, self.calibration_detection_counts.get(detected, 0) - 1)
        if base in self.calibration_suspect_bases:
            self.calibration_suspect_bases.remove(base)

    def _apply_calibration_summary_delta(self, base: str) -> None:
        classification = self._classify_calibration_detection(base)
        previous = self.calibration_detection_bins.get(base)
        if previous == classification:
            pass
        else:
            if previous:
                self.calibration_detection_counts[previous] = max(
                    0,
                    self.calibration_detection_counts.get(previous, 0) - 1,
                )
            self.calibration_detection_bins[base] = classification
            self.calibration_detection_counts[classification] = self.calibration_detection_counts.get(classification, 0) + 1
        has_warning = bool(self.calibration_warnings.get(base))
        if has_warning and base not in self.calibration_suspect_bases:
            self.calibration_suspect_bases.add(base)
        elif not has_warning and base in self.calibration_suspect_bases:
            self.calibration_suspect_bases.remove(base)

    def _classify_calibration_detection(self, base: str) -> str:
        results = self.calibration_results.get(base, {})
        positives = sum(1 for cam in ("lwir", "visible") if results.get(cam) is True)
        if positives >= 2:
            return "both"
        if positives == 1:
            return "partial"
        return "missing"

    def set_mark_reason(self, base: str, reason: Optional[str], manual_reason: str) -> bool:
        """Assign or clear a delete reason while managing auto overrides."""
        if reason is None:
            previous = self.marked_for_delete.pop(base, None)
            if previous is None:
                return False
            self._adjust_reason_count(previous, -1)
            if previous != manual_reason:
                self.auto_override.add(base)
            return True
        existing = self.marked_for_delete.get(base)
        if existing == reason:
            return False
        if existing:
            self._adjust_reason_count(existing, -1)
        self.auto_override.discard(base)
        self.marked_for_delete[base] = reason
        self._adjust_reason_count(reason, 1)
        return True

    def toggle_manual_mark(self, base: str, manual_reason: str) -> bool:
        """Toggle the manual deletion flag used by the Delete key."""
        if base in self.marked_for_delete:
            return self.set_mark_reason(base, None, manual_reason)
        return self.set_mark_reason(base, manual_reason, manual_reason)

    def clear_markings(self) -> None:
        """Remove all delete marks and auto overrides."""
        self.marked_for_delete.clear()
        self.mark_reason_counts.clear()
        self.auto_override.clear()
