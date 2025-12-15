"""Signature services to discover duplicate frames and emit progress across a dataset sweep.

Includes controllers and managers that queue work, handle cancellation, and surface ready signals.
"""

from .signature_controller import SignatureController
from .signature_scan_manager import SignatureScanManager

__all__ = [
    "SignatureController",
    "SignatureScanManager",
]
