"""Service layer aggregation for calibration, overlays, labels, signatures, and UI helpers.

Provides convenience re-exports so callers can import the core viewer services from a single namespace.
"""

from .calibration import (
    CALIBRATION_RESULTS_FILENAME,
    CalibrationController,
    CalibrationDebugger,
    CalibrationExtrinsicSample,
    CalibrationExtrinsicSolver,
    CalibrationRefiner,
    CalibrationSample,
    CalibrationSolver,
    CalibrationWorkflowMixin,
    DeferredCalibrationQueue,
)
from .labels import LabelWorkflow, LabelingController, build_controller
from .overlays import OverlayPrefetcher, OverlayWorkflow
from .signatures import SignatureController, SignatureScanManager
from .ui import CancelController, ProgressQueueManager, UiStateHelper

__all__ = [
    "CALIBRATION_RESULTS_FILENAME",
    "CancelController",
    "CalibrationController",
    "CalibrationDebugger",
    "CalibrationExtrinsicSample",
    "CalibrationExtrinsicSolver",
    "CalibrationRefiner",
    "CalibrationSample",
    "CalibrationSolver",
    "CalibrationWorkflowMixin",
    "DeferredCalibrationQueue",
    "LabelWorkflow",
    "LabelingController",
    "OverlayPrefetcher",
    "OverlayWorkflow",
    "ProgressQueueManager",
    "SignatureController",
    "SignatureScanManager",
    "UiStateHelper",
    "build_controller",
]
