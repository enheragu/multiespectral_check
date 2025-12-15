"""Label services for loading, caching, and editing YOLO annotations tied to each dataset.

Exposes the workflow facade plus controller factory used to wire label interactions into the viewer.
"""

from .label_workflow import LabelWorkflow
from .labeling_controller import LabelingController, build_controller

__all__ = [
    "LabelWorkflow",
    "LabelingController",
    "build_controller",
]
