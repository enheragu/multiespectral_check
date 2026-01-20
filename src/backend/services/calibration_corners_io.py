"""I/O functions for storing/loading calibration corners in individual files."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common.log_utils import log_debug
from common.yaml_utils import load_yaml, save_yaml


CALIBRATION_DIR = "calibration"


def get_calibration_dir(dataset_path: Path) -> Path:
    """Get the calibration directory for a dataset."""
    return dataset_path / CALIBRATION_DIR


def get_corners_file(dataset_path: Path, image_base: str) -> Path:
    """Get the file path for storing corners of a specific image.

    For collections, image_base format is "child_dataset/IMG_001".
    This will create: dataset_path/child_dataset/calibration/IMG_001.yaml

    For simple datasets, image_base is just "IMG_001".
    This will create: dataset_path/calibration/IMG_001.yaml
    """
    # Handle collections: save to child dataset's calibration folder
    if "/" in image_base:
        # e.g., "25-12-22_11-18/000949" → dataset_path/25-12-22_11-18/calibration/000949.yaml
        child_dataset, filename = image_base.rsplit("/", 1)
        child_dataset_path = dataset_path / child_dataset
        calib_dir = child_dataset_path / CALIBRATION_DIR
        return calib_dir / f"{filename}.yaml"
    else:
        # Simple case: dataset/calibration/IMG_001.yaml
        calib_dir = get_calibration_dir(dataset_path)
        return calib_dir / f"{image_base}.yaml"


# Channels that can have corners (original and subpixel)
CORNER_CHANNELS = ("lwir", "visible", "lwir_subpixel", "visible_subpixel")


def save_corners(
    dataset_path: Path,
    image_base: str,
    corners: Dict[str, Optional[List[List[float]]]],
    image_sizes: Optional[Dict[str, Tuple[int, int]]] = None,
) -> None:
    """Save corners for a specific image to its individual file.

    Args:
        dataset_path: Path to the dataset (or collection root)
        image_base: Image base name (e.g., "IMG_001" or "child_dataset/IMG_001" for collections)
        corners: Dict with corner keys:
            - "lwir": Original LWIR corners (list of [x, y])
            - "visible": Original visible corners
            - "lwir_subpixel": Subpixel-refined LWIR corners (optional)
            - "visible_subpixel": Subpixel-refined visible corners (optional)
        image_sizes: Optional dict with image sizes per channel:
            - "lwir": (width, height) of LWIR image
            - "visible": (width, height) of visible image
    """
    corners_file = get_corners_file(dataset_path, image_base)

    # Create parent directory (including calibration dir)
    corners_file.parent.mkdir(parents=True, exist_ok=True)

    # Serialize corners (convert tuples to lists for YAML)
    serialized: Dict[str, Any] = {}
    for channel in CORNER_CHANNELS:
        corner_list = corners.get(channel)
        if corner_list is not None:
            serialized[channel] = [[float(x), float(y)] for x, y in corner_list]

    # Add image sizes if provided
    if image_sizes:
        sizes_dict: Dict[str, List[int]] = {}
        for ch in ("lwir", "visible"):
            if ch in image_sizes and image_sizes[ch]:
                sizes_dict[ch] = list(image_sizes[ch])
        if sizes_dict:
            serialized["image_size"] = sizes_dict

    log_debug(f"💾 Saving corners for {image_base} in {dataset_path.name} → {corners_file.relative_to(dataset_path)}", "CORNERS")
    lwir_count = len(serialized.get('lwir', []) or [])
    vis_count = len(serialized.get('visible', []) or [])
    lwir_sub = len(serialized.get('lwir_subpixel', []) or [])
    vis_sub = len(serialized.get('visible_subpixel', []) or [])
    log_debug(f"   LWIR: {lwir_count} corners{f' (+{lwir_sub} subpixel)' if lwir_sub else ''}, VIS: {vis_count} corners{f' (+{vis_sub} subpixel)' if vis_sub else ''}", "CORNERS")

    save_yaml(corners_file, serialized)


def load_corners(dataset_path: Path, image_base: str) -> Optional[Dict[str, Any]]:
    """Load corners for a specific image from its individual file.

    Args:
        dataset_path: Path to the dataset
        image_base: Image base name (e.g., "IMG_001")

    Returns:
        Dict with corner keys ("lwir", "visible", etc.) and optionally "image_size".
        Returns None if file doesn't exist.
    """
    corners_file = get_corners_file(dataset_path, image_base)

    if not corners_file.exists():
        log_debug(f"📂 No corners file for {image_base} in {dataset_path.name}", "CORNERS")
        return None

    try:
        data = load_yaml(corners_file)

        if not data:
            return None

        # Deserialize corners (convert lists back to tuples)
        corners: Dict[str, Any] = {}
        for channel in CORNER_CHANNELS:
            corner_data = data.get(channel)
            if corner_data is None:
                continue  # Skip None/missing channels
            elif isinstance(corner_data, list):
                corners[channel] = [[float(pt[0]), float(pt[1])] for pt in corner_data if len(pt) == 2]

        # Load image sizes if present
        image_size_data = data.get("image_size")
        if isinstance(image_size_data, dict):
            image_sizes: Dict[str, Tuple[int, int]] = {}
            for ch in ("lwir", "visible"):
                ch_size = image_size_data.get(ch)
                if isinstance(ch_size, (list, tuple)) and len(ch_size) >= 2:
                    image_sizes[ch] = (int(ch_size[0]), int(ch_size[1]))
            if image_sizes:
                corners["image_size"] = image_sizes

        log_debug(f"📖 Loaded corners for {image_base} from {dataset_path.name}", "CORNERS")
        lwir_count = len(corners.get('lwir', []) or [])
        vis_count = len(corners.get('visible', []) or [])
        lwir_sub = len(corners.get('lwir_subpixel', []) or [])
        vis_sub = len(corners.get('visible_subpixel', []) or [])
        log_debug(f"   LWIR: {lwir_count} corners{f' (+{lwir_sub} subpixel)' if lwir_sub else ''}, VIS: {vis_count} corners{f' (+{vis_sub} subpixel)' if vis_sub else ''}", "CORNERS")
        return corners

    except Exception as e:
        log_debug(f"❌ Failed to load corners for {image_base}: {e}", "CORNERS")
        return None


def load_corners_for_dataset(dataset_path: Path) -> Dict[str, Dict[str, Optional[List[List[float]]]]]:
    """Load all corners for all images in a dataset or collection.

    Args:
        dataset_path: Path to the dataset or collection root

    Returns:
        Dict mapping base names to corner dicts {"lwir": [...], "visible": [...]}
    """
    result: Dict[str, Dict[str, Optional[List[List[float]]]]] = {}
    bases = list_images_with_corners(dataset_path)

    for base in bases:
        corners = load_corners(dataset_path, base)
        if corners:
            result[base] = corners

    return result


def delete_corners(dataset_path: Path, image_base: str) -> None:
    """Delete the corners file for a specific image."""
    corners_file = get_corners_file(dataset_path, image_base)
    if corners_file.exists():
        log_debug(f"🗑️  Deleting corners file for {image_base} in {dataset_path.name}", "CORNERS")
        corners_file.unlink()


def list_images_with_corners(dataset_path: Path) -> List[str]:
    """List all image bases that have corners stored.

    For collections, recursively checks child datasets and returns bases in "child/IMG" format.
    For simple datasets, returns flat list of image bases.

    Returns:
        List of image base names (e.g., ["IMG_001", ...] or ["child/IMG_001", ...])
    """
    bases = []

    # Check if this is a collection (has child datasets)
    child_dirs = []
    if dataset_path.is_dir():
        for child in dataset_path.iterdir():
            if child.is_dir() and (child / "lwir").is_dir() and (child / "visible").is_dir():
                child_dirs.append(child)

    if child_dirs:
        # Collection: check each child dataset
        for child_dir in child_dirs:
            calib_dir = child_dir / CALIBRATION_DIR
            if calib_dir.exists():
                for file in calib_dir.glob("*.yaml"):
                    if file.stem:
                        bases.append(f"{child_dir.name}/{file.stem}")
    else:
        # Simple dataset: check local calibration dir
        calib_dir = get_calibration_dir(dataset_path)
        if calib_dir.exists():
            for file in calib_dir.glob("*.yaml"):
                if file.stem:
                    bases.append(file.stem)

    return sorted(bases)


def clear_all_corners(dataset_path: Path) -> int:
    """Delete all corner files for a dataset or collection.

    For collections, clears corners from all child datasets.

    Returns:
        Number of files deleted
    """
    count = 0

    # Check if this is a collection (has child datasets)
    child_dirs = []
    if dataset_path.is_dir():
        for child in dataset_path.iterdir():
            if child.is_dir() and (child / "lwir").is_dir() and (child / "visible").is_dir():
                child_dirs.append(child)

    if child_dirs:
        # Collection: clear each child dataset
        for child_dir in child_dirs:
            calib_dir = child_dir / CALIBRATION_DIR
            if calib_dir.exists():
                for file in calib_dir.glob("*.yaml"):
                    try:
                        file.unlink()
                        count += 1
                    except Exception:
                        pass
                # Try to remove directory if empty
                try:
                    if not any(calib_dir.iterdir()):
                        calib_dir.rmdir()
                except Exception:
                    pass
    else:
        # Simple dataset: clear local calibration dir
        calib_dir = get_calibration_dir(dataset_path)
        if calib_dir.exists():
            for file in calib_dir.glob("*.yaml"):
                try:
                    file.unlink()
                    count += 1
                except Exception:
                    pass
            # Try to remove directory if empty
            try:
                if not any(calib_dir.iterdir()):
                    calib_dir.rmdir()
            except Exception:
                pass

    return count
