"""Service layer aggregation for calibration, overlays, labels, and signatures.

Provides convenience re-exports so callers can import the core viewer services from a single namespace.
NOTE: UI services (CancelController, ProgressQueueManager, UiStateHelper) are now in frontend.services.ui
"""

from .calibration import (
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

__all__ = [
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
    "SignatureController",
    "SignatureScanManager",
    "build_controller",
]
