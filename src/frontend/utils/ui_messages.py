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
    "and keep datasets tidy while detections and duplicate sweeps run in the background.\n\n"
    "⚠ Note: This help reference, along with the GUI, is a work in progress. While we try to keep it in sync "
    "with the application menus, some entries may be outdated or incomplete after recent updates."
)
HELP_MENU_SECTIONS: Sequence[Tuple[str, Sequence[Tuple[str, str]]]] = [
    (
        "File",
        [
            ("Save current status", "Save the current dataset state (Ctrl+S)."),
            ("Exit", "Close the application (Ctrl+Q)."),
        ],
    ),
    (
        "Dataset",
        [
            ("Load dataset…", "Choose the folder that contains lwir/ and visible/ directories."),
            ("Load recent", "Quick-access submenu with recently opened datasets."),
            (
                "Detect delete candidates",
                "Submenu with sweep tools: Run duplicate sweep (signature-based), "
                "\u26a0 Run blur/motion sweep (quality analysis \u2014 still experimental), "
                "and Run pattern sweep.",
            ),
            (
                "Untag delete candidates",
                "Remove delete tags by category (all, blurry, motion-blur, sync-mismatch, "
                "missing-pair, duplicates) without moving files.",
            ),
            (
                "Delete marked",
                "Move marked pairs into to_delete. Can target all selected or filter by "
                "category (blurry, motion-blur, sync-mismatch, missing-pair, duplicates).",
            ),
            (
                "Restore from trash",
                "Move items out of to_delete back into the dataset. Can restore all or by "
                "category (blurry, motion-blur, sync-mismatch, missing-pair, duplicates).",
            ),
            ("Reset dataset (dangerous)…", "Reset the dataset state — removes all tags and metadata changes."),
        ],
    ),
    (
        "View",
        [
            ("Undistort images", "Render images using the calibrated undistortion matrices (toggle)."),
            (
                "Show Grid",
                "Overlay a framing grid on images: Off, Thirds, or Detailed (9×9).",
            ),
            (
                "Stereo Alignment",
                "Control how stereo-rectified images are displayed: Disabled, Full View, "
                "FOV Focus, or Max Overlap.",
            ),
            (
                "Corner Display",
                "Choose which chessboard corners to draw: Original Only, Subpixel Only, or Both (debug).",
            ),
            ("Show labels", "Draw label boxes on visible/LWIR views when label files exist."),
            ("Show image info overlay", "Display image metadata overlays on top of the images."),
            (
                "Filter",
                "Filter the image list by status: all images, calibration candidates "
                "(any / both / partial / missing detections), or delete candidates "
                "(all / manual / duplicate / blurry / motion / sync / missing pair).",
            ),
        ],
    ),
    (
        "Calibration",
        [
            ("Auto search calibration candidates…", "Automatically search for chessboard patterns across the dataset."),
            ("Detect chessboards", "Detect chessboards on every calibration candidate currently tagged."),
            ("Refine chessboard corners", "Apply subpixel refinement to existing detections for better accuracy."),
            ("Use Subpixel Corners", "Toggle whether calibration computation uses subpixel-refined corners."),
            ("Compute calibration matrices", "Solve LWIR and visible camera intrinsic matrices."),
            ("Compute extrinsic transform", "Estimate the stereo transform using paired detections on both cameras."),
            ("Check calibration report", "Open the review dialog to inspect matrices, reprojection errors, and exports."),
            ("Export calibration debug overlays", "Export debug images with drawn corners and reprojection overlays."),
            ("Import calibration data", "Load calibration_matrices.yaml files that were generated elsewhere."),
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
            ("Reload labels config from source", "Re-read the labels YAML configuration from disk."),
            ("Run labelling on current", "Run the model on the displayed pair for the active channel."),
            ("Run labelling on dataset…", "Batch-run the model across all images for the active channel."),
            ("Clear labels for current", "Delete saved label TXT files for the current pair."),
            (
                "Manual labelling mode",
                "Click two corners (rubber band) to draw a box; pick a class via id:name autocomplete; "
                "right-click a box to delete it; Esc cancels the selection (Ctrl+L).",
            ),
        ],
    ),
    (
        "Help",
        [("See help", "Open this dialog at any time for shortcuts and menu references (Ctrl+H)."),],
    ),
    (
        "Workspace",
        [
            ("Open workspace…", "Select a folder containing multiple datasets/collections to manage."),
            ("Refresh workspace", "Re-scan the workspace directory and update dataset statistics."),
            (
                "Detect delete candidates",
                "Run sweeps across all datasets: Duplicates (signature-based), "
                "⚠ Blur/motion (quality analysis — still experimental), "
                "Patterns (calibration board detection), Missing pairs, or Run all sweeps.",
            ),
            (
                "Untag delete candidates",
                "Remove delete tags from all datasets: Untag all, or by category "
                "(blurry, motion-blur, sync-mismatch, missing-pair, duplicates).",
            ),
            (
                "Delete marked",
                "Move marked pairs to trash across all datasets: Delete all marked, or by category.",
            ),
            (
                "Restore from trash",
                "Restore deleted pairs from trash across all datasets: Restore all, or by category.",
            ),
            ("Clear empty dataset folders", "Remove empty lwir/visible/to_delete folders from datasets."),
            (
                "Default Calibration",
                "Manage workspace-wide default calibration: Set from current dataset, Clear default, "
                "or Show calibration info.",
            ),
            ("Reset selected dataset…", "Reset the selected dataset state (dangerous)."),
            ("Reset workspace…", "Reset all datasets in the workspace (dangerous)."),
        ],
    ),
]

# Workspace Panel - the panel in the Workspace tab
HELP_WORKSPACE_PANEL: Sequence[Tuple[str, str]] = [
    ("Dataset table", "Shows all datasets/collections with statistics: pairs count, deleted, delete reasons, tagged to delete, calibration marked, and sweep status."),
    ("Open selected", "Load the selected dataset or collection into the image viewer."),
    ("Dataset Notes", "Free-form notes field saved per-dataset. Use it to track observations or progress."),
]

# Dataset/Collection View - the main image viewer panel
HELP_DATASET_VIEW: Sequence[Tuple[str, str]] = [
    ("Image pair display", "Shows LWIR (thermal) and Visible images side by side. Supports zoom, pan, and synchronized navigation."),
    ("Navigation", "Use arrow keys (← / →) or Space to browse images. The slider and counter show current position."),
    ("Status overlays", "Calibration and delete markers appear as icons on the images. Toggle with View menu options."),
    ("Labels overlay", "When labels exist, bounding boxes are drawn on the images. Enable via View > Show labels."),
    ("Zoom & Pan", "Scroll to zoom, drag to pan. Views synchronize when both images are visible."),
    ("Input mode selector", "Choose which channel (LWIR, Visible, or Both) is used for labelling operations."),
]

# Context menus (right-click) - separate from menubar
HELP_CONTEXT_MENUS: Sequence[Tuple[str, Sequence[Tuple[str, str]]]] = [
    (
        "Image Viewer (Right-click on image)",
        [
            ("Previous / Next image", "Navigate between images (← / →)."),
            (
                "Delete reason marks",
                "Toggle delete marks: Manual (Del), Duplicate (Ctrl+Shift+D), "
                "Blurry (Ctrl+Shift+B), Motion-blur (Ctrl+Shift+M), Sync-mismatch (Ctrl+Shift+S), "
                "or Missing-pair. The current mark appears checked.",
            ),
            ("Mark as calibration candidate", "Toggle calibration candidate flag (Ctrl+Shift+C)."),
            (
                "Re-run calibration detection",
                "Force re-detection of chessboard corners on this image (Ctrl+Shift+R). "
                "Only available for images already marked as calibration candidates.",
            ),
            ("Enter manual labelling mode", "Switch to manual label drawing mode (Ctrl+L)."),
            ("Copy image path", "Copy the LWIR or Visible image path to clipboard."),
        ],
    ),
    (
        "Manual Labelling Mode (Right-click on label)",
        [
            ("Edit label", "Change the class or bounding box of an existing label."),
            ("Delete label", "Remove the label under the cursor."),
            ("Exit manual labelling mode", "Return to normal viewing mode (Ctrl+L)."),
        ],
    ),
]

HELP_SHORTCUTS: Sequence[Tuple[str, str]] = [
    ("← / → / Space", "Browse the dataset"),
    ("Ctrl+S", "Save current dataset status"),
    ("Ctrl+Q", "Exit the application"),
    ("Ctrl+H", "Open the help dialog"),
    ("Ctrl+L", "Toggle manual labelling mode"),
    ("Ctrl+Shift+C", "Toggle calibration candidate for the current pair"),
    ("Delete", "Toggle the manual delete mark"),
    ("Esc", "Cancel a manual label selection in progress"),
    ("Ctrl+Shift+D / B / M / S", "Assign duplicate, blurry, motion, or sync reasons"),
    ("Ctrl+Shift+R", "Re-run calibration detection for the current tagged pair"),
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

