"""I/O functions for storing/loading calibration corners in individual files."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import yaml

from common.log_utils import log_debug


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


def save_corners(
    dataset_path: Path,
    image_base: str,
    corners: Dict[str, Optional[List[List[float]]]],
    refined: Optional[Dict[str, bool]] = None,
) -> None:
    """Save corners for a specific image to its individual file.

    Args:
        dataset_path: Path to the dataset (or collection root)
        image_base: Image base name (e.g., "IMG_001" or "child_dataset/IMG_001" for collections)
        corners: Dict with "lwir" and "visible" keys, each containing list of (x,y) tuples or None
        refined: Optional dict with "lwir" and "visible" keys indicating if corners were subpixel refined
    """
    corners_file = get_corners_file(dataset_path, image_base)

    # Create parent directory (including calibration dir)
    corners_file.parent.mkdir(parents=True, exist_ok=True)

    # Serialize corners (convert tuples to lists for YAML)
    serialized: Dict[str, Optional[List[List[float]]]] = {}
    for channel in ["lwir", "visible"]:
        corner_list = corners.get(channel)
        if corner_list is not None:
            serialized[channel] = [[float(x), float(y)] for x, y in corner_list]
        else:
            serialized[channel] = None

    # Add refined flags if provided
    if refined:
        serialized["refined"] = refined  # type: ignore[assignment]
        log_debug(f"   Refined flags: {refined}", "CORNERS")

    log_debug(f"💾 Saving corners for {image_base} in {dataset_path.name} → {corners_file.relative_to(dataset_path)}", "CORNERS")
    log_debug(f"   LWIR: {len(serialized.get('lwir', []) or [])} corners, VIS: {len(serialized.get('visible', []) or [])} corners", "CORNERS")

    with open(corners_file, "w") as f:
        yaml.safe_dump(serialized, f, default_flow_style=False, sort_keys=False)


def load_corners(dataset_path: Path, image_base: str) -> Optional[Dict[str, Optional[List[List[float]]]]]:
    """Load corners for a specific image from its individual file.

    Args:
        dataset_path: Path to the dataset
        image_base: Image base name (e.g., "IMG_001")

    Returns:
        Dict with "lwir" and "visible" keys, or None if file doesn't exist.
        May also include "refined" key with dict of channel → bool.
    """
    corners_file = get_corners_file(dataset_path, image_base)

    if not corners_file.exists():
        log_debug(f"📂 No corners file for {image_base} in {dataset_path.name}", "CORNERS")
        return None

    try:
        with open(corners_file, "r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            return None

        # Deserialize corners (convert lists back to tuples)
        corners: Dict[str, Optional[List[List[float]]]] = {}
        for channel in ["lwir", "visible"]:
            corner_data = data.get(channel)
            if corner_data is None:
                corners[channel] = None
            elif isinstance(corner_data, list):
                corners[channel] = [[float(pt[0]), float(pt[1])] for pt in corner_data if len(pt) == 2]
            else:
                corners[channel] = None

        # Load refined flags if present
        refined = data.get("refined")
        if isinstance(refined, dict):
            corners["refined"] = refined  # type: ignore[assignment]

        log_debug(f"📖 Loaded corners for {image_base} from {dataset_path.name}", "CORNERS")
        log_debug(f"   LWIR: {len(corners.get('lwir', []) or [])} corners, VIS: {len(corners.get('visible', []) or [])} corners", "CORNERS")
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
