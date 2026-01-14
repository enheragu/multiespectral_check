"""Shared constants and labels for dataset filtering modes."""
from __future__ import annotations

FILTER_ALL = "all"
FILTER_CAL_ANY = "calibration_any"
FILTER_CAL_BOTH = "calibration_both"
FILTER_CAL_PARTIAL = "calibration_partial"
FILTER_CAL_MISSING = "calibration_missing"
FILTER_DELETE = "delete_candidates"
FILTER_DELETE_MANUAL = "delete_manual"
FILTER_DELETE_DUP = "delete_duplicate"
FILTER_DELETE_BLURRY = "delete_blurry"
FILTER_DELETE_MOTION = "delete_motion"
FILTER_DELETE_SYNC = "delete_sync"
FILTER_DELETE_MISSING = "delete_missing_pair"

FILTER_ACTION_NAMES = {
    FILTER_ALL: "action_filter_all",
    FILTER_CAL_ANY: "action_filter_calibration_any",
    FILTER_CAL_BOTH: "action_filter_calibration_both",
    FILTER_CAL_PARTIAL: "action_filter_calibration_partial",
    FILTER_CAL_MISSING: "action_filter_calibration_missing",
    FILTER_DELETE: "action_filter_delete_candidates",
    FILTER_DELETE_MANUAL: "action_filter_delete_manual",
    FILTER_DELETE_DUP: "action_filter_delete_duplicate",
    FILTER_DELETE_BLURRY: "action_filter_delete_blurry",
    FILTER_DELETE_MOTION: "action_filter_delete_motion",
    FILTER_DELETE_SYNC: "action_filter_delete_sync",
    FILTER_DELETE_MISSING: "action_filter_delete_missing_pair",
}

FILTER_STATUS_TITLES = {
    FILTER_CAL_ANY: "Calibration",
    FILTER_CAL_BOTH: "Both detections",
    FILTER_CAL_PARTIAL: "Single detection",
    FILTER_CAL_MISSING: "Missing detections",
    FILTER_DELETE: "Delete candidates",
    FILTER_DELETE_MANUAL: "Delete (manual)",
    FILTER_DELETE_DUP: "Duplicates",
    FILTER_DELETE_BLURRY: "Blurry",
    FILTER_DELETE_MOTION: "Motion blur",
    FILTER_DELETE_SYNC: "Sync mismatch",
    FILTER_DELETE_MISSING: "Missing pair",
}

FILTER_MESSAGE_LABELS = {
    FILTER_CAL_ANY: "calibration-tagged images",
    FILTER_CAL_BOTH: "dual-detection images",
    FILTER_CAL_PARTIAL: "single-detection images",
    FILTER_CAL_MISSING: "detection-missing images",
    FILTER_DELETE: "images marked for delete",
    FILTER_DELETE_MANUAL: "manual delete candidates",
    FILTER_DELETE_DUP: "duplicate candidates",
    FILTER_DELETE_BLURRY: "blurry candidates",
    FILTER_DELETE_MOTION: "motion-blur candidates",
    FILTER_DELETE_SYNC: "sync-mismatch candidates",
    FILTER_DELETE_MISSING: "missing-pair candidates",
}
