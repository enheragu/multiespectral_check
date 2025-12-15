"""Overlay services for building and prefetching composited layers around viewer navigation.

Keeps overlay caches warm and coordinates async fetching so image panes stay responsive.
"""

from .overlay_prefetcher import OverlayPrefetcher
from .overlay_workflow import OverlayWorkflow

__all__ = [
    "OverlayPrefetcher",
    "OverlayWorkflow",
]
