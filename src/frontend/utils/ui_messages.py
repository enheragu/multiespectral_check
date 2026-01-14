"""Shared UI strings and formatting helpers."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

DELETE_BUTTON_TEXT = "Delete selected"
RESTORE_ACTION_TEXT = "Restore from trash"
STATUS_SELECT_DATASET = "Select a dataset"
STATUS_NO_IMAGES = "No images"
SUPPORT_EMAIL = "e.heredia@umh.es"
HELP_OVERVIEW = (
    "Multispectral Check lets you compare LWIR/visible pairs, tag calibration images, "
    "and keep datasets tidy while detections and duplicate sweeps run in the background."
)
HELP_MENU_SECTIONS: Sequence[Tuple[str, Sequence[Tuple[str, str]]]] = [
    (
        "Dataset",
        [
            ("Load dataset…", "Choose the folder that contains lwir/ and visible/ directories."),
            ("Run duplicate sweep", "Compute LWIR+visible signatures to auto-tag identical pairs."),
            ("Delete selected pairs", "Move marked pairs into to_delete so they are skipped from review."),
            ("Restore from trash", "Move items out of to_delete and back into the dataset."),
            ("Clear empty datasets", "Remove dataset folders that no longer contain any images."),
        ],
    ),
    (
        "View",
        [
            ("Toggle rectified view", "Render images using the calibrated undistortion matrices."),
            ("Toggle grid/overlays", "Show framing guides, duplicate reasons, and calibration highlights."),
            ("Show labels", "Draw YOLO label boxes on visible/LWIR views when label files exist."),
            ("Calibration filters", "Focus on tagged images that match a calibration status."),
        ],
    ),
    (
        "Labelling",
        [
            ("Configure model…", "Select the YOLO model file to use for automatic labelling."),
            (
                "Configure labels YAML…",
                "Load a labels YAML (e.g., COCO); it is copied to labels/labels.yaml so the dataset remembers class ids/names.",
            ),
            ("Run labelling on current", "Run the model on the displayed pair for the active channel."),
            ("Run labelling on dataset…", "Batch-run the model across all images for the active channel."),
            ("Clear labels for current", "Delete saved label TXT files for the current pair."),
            (
                "Manual labelling mode",
                "Click two corners (rubber band) to draw a box; pick a class via id:name autocomplete; right-click a box to delete it; Esc cancels the selection.",
            ),
        ],
    ),
    (
        "Calibration",
        [
            ("Re-run detection", "Detect chessboards on every calibration candidate currently tagged."),
            ("Refine corners", "Apply subpixel refinement to existing detections for better accuracy."),
            ("Compute intrinsics", "Solve LWIR and visible camera matrices once enough samples exist."),
            ("Compute extrinsic", "Estimate the stereo transform using paired detections on both cameras."),
            ("Import calibration", "Load calibration_matrices.yaml files that were generated elsewhere."),
            ("Check calibration", "Open the review dialog to inspect matrices, reprojection errors, and exports."),
        ],
    ),
    (
        "Help",
        [("See help", "Open this dialog at any time for shortcuts and menu references."),],
    ),
]
HELP_SHORTCUTS: Sequence[Tuple[str, str]] = [
    ("← / → / Space", "Browse the dataset"),
    ("Ctrl+Shift+C", "Toggle calibration candidate for the current pair"),
    ("Ctrl+H", "Open the help dialog"),
    ("Delete", "Toggle the manual delete mark"),
    ("Esc", "Cancel a manual label selection in progress"),
    ("Ctrl+Shift+D / B / M / S", "Assign duplicate, blurry, motion, or sync reasons"),
    ("Ctrl+Shift+R", "Re-run calibration detection for the current tagged pair"),
    ("Dataset → Run duplicate sweep", "Queue the background duplicate scanner"),
    ("Dataset → Delete selected pairs", "Move all marked items to to_delete"),
    ("Dataset → Restore from trash", "Return items currently in to_delete"),
]


def format_move_confirmation(count: int) -> str:
    return f"{count} pairs will be moved into to_delete. Continue?"


def format_move_failure(failed: Iterable[str]) -> str:
    return "Could not move: " + ", ".join(failed)


def format_move_success(count: int, destination: str) -> str:
    return f"Moved {count} pairs into {destination}"


def restore_prompt_message() -> str:
    return "Load a dataset first."


def no_restore_items_message() -> str:
    return "No image pairs available in to_delete."


def restored_pairs_message(count: int) -> str:
    return f"Restored {count} pairs from to_delete."


def calibration_removed_message(base: str) -> str:
    return f"Calibration tag removed from {base}"

