"""Shared constants and labels for dataset filtering modes."""
from __future__ import annotations

FILTER_ALL = "all"
FILTER_CAL_ANY = "calibration_any"
FILTER_CAL_BOTH = "calibration_both"
FILTER_CAL_PARTIAL = "calibration_partial"
FILTER_CAL_MISSING = "calibration_missing"
FILTER_CAL_SUSPECT = "calibration_suspect"

FILTER_ACTION_NAMES = {
    FILTER_ALL: "action_filter_all",
    FILTER_CAL_ANY: "action_filter_calibration_any",
    FILTER_CAL_BOTH: "action_filter_calibration_both",
    FILTER_CAL_PARTIAL: "action_filter_calibration_partial",
    FILTER_CAL_MISSING: "action_filter_calibration_missing",
    FILTER_CAL_SUSPECT: "action_filter_calibration_suspect",
}

FILTER_STATUS_TITLES = {
    FILTER_CAL_ANY: "Calibration",
    FILTER_CAL_BOTH: "Both detections",
    FILTER_CAL_PARTIAL: "Single detection",
    FILTER_CAL_MISSING: "Missing detections",
    FILTER_CAL_SUSPECT: "Suspect",
}

FILTER_MESSAGE_LABELS = {
    FILTER_CAL_ANY: "calibration-tagged images",
    FILTER_CAL_BOTH: "dual-detection images",
    FILTER_CAL_PARTIAL: "single-detection images",
    FILTER_CAL_MISSING: "detection-missing images",
    FILTER_CAL_SUSPECT: "suspect images",
}
