"""Cache consistency checker and recovery utilities.

Detects and fixes inconsistencies between:
- Corner files (calibration/*.yaml) vs calibration marks in .reviewer_cache.yaml
- Collection-level vs child-level calibration marks
- Summary cache vs full cache
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Set

from backend.services.calibration_corners_io import list_images_with_corners
from backend.services.cache_service import load_dataset_cache_file, save_dataset_cache_file, DATASET_CACHE_FILENAME
from common.log_utils import log_debug


def _is_dataset_dir(path: Path) -> bool:
    """Check if path is a dataset (has lwir/ and visible/ dirs)."""
    return path.is_dir() and (path / "lwir").is_dir() and (path / "visible").is_dir()


def _is_collection_dir(path: Path) -> bool:
    """Check if path is a collection (has child datasets)."""
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if _is_dataset_dir(child):
            return True
    return False


def get_calibration_marks_from_cache(cache_entry: Dict) -> Set[str]:
    """Extract set of image bases marked for calibration from cache entry."""
    calibration = cache_entry.get("calibration", {})
    if not isinstance(calibration, dict):
        return set()

    marked = set()
    for base, data in calibration.items():
        # Presence in dict = marked (no explicit 'marked' check)
        if isinstance(data, dict):
            marked.add(str(base))

    return marked


def check_dataset_consistency(dataset_path: Path) -> Dict[str, Any]:
    """Check consistency between corner files and cache for a single dataset.

    Returns dict with:
        - corners_without_marks: Set of bases with corners but not marked
        - marks_without_corners: Set of bases marked but no corners
        - fixed_marks: Set of bases added to marks (recovered from corners)
        - cleaned_marks: Set of bases removed from marks (no corners found)
    """
    log_debug(f"Checking consistency for dataset: {dataset_path.name}", "CONSISTENCY")

    # Load cache
    cache_path = dataset_path / DATASET_CACHE_FILENAME
    if not cache_path.exists():
        log_debug("  No cache file found", "CONSISTENCY")
        return {
            "corners_without_marks": set(),
            "marks_without_corners": set(),
            "fixed_marks": set(),
            "cleaned_marks": set(),
        }

    cache_entry = load_dataset_cache_file(cache_path)

    # Get marks from cache
    marked_bases = get_calibration_marks_from_cache(cache_entry)
    log_debug(f"  Cache has {len(marked_bases)} calibration marks", "CONSISTENCY")

    # Get bases with corner files
    corner_bases = set(list_images_with_corners(dataset_path))
    log_debug(f"  Found {len(corner_bases)} corner files", "CONSISTENCY")

    # Find inconsistencies
    corners_without_marks = corner_bases - marked_bases
    marks_without_corners = marked_bases - corner_bases

    if corners_without_marks:
        log_debug(f"  ⚠️  {len(corners_without_marks)} corner files without marks: {list(corners_without_marks)[:5]}", "CONSISTENCY")
    if marks_without_corners:
        log_debug(f"  ⚠️  {len(marks_without_corners)} marks without corner files: {list(marks_without_corners)[:5]}", "CONSISTENCY")

    return {
        "corners_without_marks": corners_without_marks,
        "marks_without_corners": marks_without_corners,
        "fixed_marks": set(),
        "cleaned_marks": set(),
    }


def fix_dataset_consistency(dataset_path: Path, add_missing_marks: bool = True, remove_orphan_marks: bool = False) -> Dict[str, Any]:
    """Fix inconsistencies in a dataset by reconciling cache with corner files.

    Args:
        dataset_path: Path to dataset
        add_missing_marks: If True, add calibration marks for bases with corner files
        remove_orphan_marks: If True, remove marks that don't have corner files

    Returns:
        Dict with fixed_marks and cleaned_marks sets
    """
    log_debug(f"Fixing consistency for dataset: {dataset_path.name}", "CONSISTENCY")

    result = check_dataset_consistency(dataset_path)
    corners_without_marks = result["corners_without_marks"]
    marks_without_corners = result["marks_without_corners"]

    if not corners_without_marks and not marks_without_corners:
        log_debug("  ✓ No inconsistencies found", "CONSISTENCY")
        return result

    # Load cache
    cache_path = dataset_path / DATASET_CACHE_FILENAME
    cache_entry = load_dataset_cache_file(cache_path)
    calibration = cache_entry.get("calibration", {})
    if not isinstance(calibration, dict):
        calibration = {}
        cache_entry["calibration"] = calibration

    cache_modified = False
    fixed_marks = set()
    cleaned_marks = set()

    # Add missing marks (recover from corner files)
    if add_missing_marks and corners_without_marks:
        log_debug(f"  Adding {len(corners_without_marks)} missing marks", "CONSISTENCY")
        for base in corners_without_marks:
            # Presence in dict = marked (no explicit 'marked' field needed)
            calibration[base] = {
                "auto": False,
                "outlier": {"lwir": False, "visible": False, "stereo": False},
                "results": {"lwir": False, "visible": False}
            }
            fixed_marks.add(base)
            cache_modified = True

    # Remove orphan marks (cleanup marks without corners)
    if remove_orphan_marks and marks_without_corners:
        log_debug(f"  Removing {len(marks_without_corners)} orphan marks", "CONSISTENCY")
        for base in marks_without_corners:
            if base in calibration:
                del calibration[base]
                cleaned_marks.add(base)
                cache_modified = True

    # Save cache if modified
    if cache_modified:
        save_dataset_cache_file(cache_path, cache_entry)
        log_debug(f"  ✓ Saved updated cache: +{len(fixed_marks)} marks, -{len(cleaned_marks)} marks", "CONSISTENCY")

    result["fixed_marks"] = fixed_marks
    result["cleaned_marks"] = cleaned_marks
    return result


def check_collection_consistency(collection_path: Path) -> Dict[str, Any]:
    """Check consistency for a collection and its children.

    For collections, calibration marks can be at collection level (with child prefix)
    or at child dataset level. Corner files are always at child dataset level.
    """
    log_debug(f"Checking collection consistency: {collection_path.name}", "CONSISTENCY")

    # Get child datasets
    child_datasets = [p for p in collection_path.iterdir() if _is_dataset_dir(p)]
    if not child_datasets:
        log_debug("  No child datasets found", "CONSISTENCY")
        return {"children": {}}

    log_debug(f"  Found {len(child_datasets)} child datasets", "CONSISTENCY")

    # Load collection cache
    collection_cache_path = collection_path / DATASET_CACHE_FILENAME
    collection_entry = load_dataset_cache_file(collection_cache_path) if collection_cache_path.exists() else {}
    collection_marks = get_calibration_marks_from_cache(collection_entry)

    log_debug(f"  Collection cache has {len(collection_marks)} marks", "CONSISTENCY")

    # Check each child
    children_results = {}
    for child_path in child_datasets:
        child_name = child_path.name
        child_result = check_dataset_consistency(child_path)

        # Find marks at collection level that belong to this child
        child_prefix = f"{child_name}/"
        collection_marks_for_child = {base for base in collection_marks if base.startswith(child_prefix)}

        if collection_marks_for_child:
            # Extract base without prefix
            unprefixed = {base[len(child_prefix):] for base in collection_marks_for_child}
            child_result["collection_marks"] = unprefixed
            log_debug(f"    {child_name}: {len(unprefixed)} marks at collection level", "CONSISTENCY")

        children_results[child_name] = child_result

    return {
        "children": children_results,
        "collection_marks": collection_marks,
    }


def fix_collection_consistency(collection_path: Path, push_marks_to_children: bool = True) -> Dict[str, Any]:
    """Fix collection consistency by pushing marks to child datasets.

    For collections, we ensure calibration marks are stored at child dataset level,
    and corner files match the marks.
    """
    log_debug(f"Fixing collection consistency: {collection_path.name}", "CONSISTENCY")

    check_result = check_collection_consistency(collection_path)

    if not push_marks_to_children:
        return check_result

    # Get child datasets
    child_datasets = [p for p in collection_path.iterdir() if _is_dataset_dir(p)]

    # Load collection cache
    collection_cache_path = collection_path / DATASET_CACHE_FILENAME
    if not collection_cache_path.exists():
        log_debug("  No collection cache found", "CONSISTENCY")
        return check_result

    collection_entry = load_dataset_cache_file(collection_cache_path)
    collection_calibration = collection_entry.get("calibration", {})

    if not isinstance(collection_calibration, dict):
        log_debug("  No calibration marks in collection cache", "CONSISTENCY")
        return check_result

    collection_modified = False

    # Push marks to children
    for child_path in child_datasets:
        child_name = child_path.name
        child_prefix = f"{child_name}/"

        # Load child cache
        child_cache_path = child_path / DATASET_CACHE_FILENAME
        if not child_cache_path.exists():
            log_debug(f"    {child_name}: No cache, skipping", "CONSISTENCY")
            continue

        child_entry = load_dataset_cache_file(child_cache_path)
        child_calibration = child_entry.get("calibration", {})
        if not isinstance(child_calibration, dict):
            child_calibration = {}
            child_entry["calibration"] = child_calibration

        child_modified = False
        pushed_count = 0

        # Find marks for this child at collection level
        for collection_base, collection_data in list(collection_calibration.items()):
            if not collection_base.startswith(child_prefix):
                continue

            # Extract base without prefix
            child_base = collection_base[len(child_prefix):]

            # Copy mark to child if not already there
            if child_base not in child_calibration:
                child_calibration[child_base] = collection_data.copy() if isinstance(collection_data, dict) else {}
                child_modified = True
                pushed_count += 1

            # Remove from collection cache (keep calibration at child level only)
            del collection_calibration[collection_base]
            collection_modified = True

        # Save child cache if modified
        if child_modified:
            save_dataset_cache_file(child_cache_path, child_entry)
            log_debug(f"    ✓ {child_name}: Pushed {pushed_count} marks from collection", "CONSISTENCY")

    # Save collection cache if modified
    if collection_modified:
        save_dataset_cache_file(collection_cache_path, collection_entry)
        log_debug("  ✓ Updated collection cache (removed child-specific marks)", "CONSISTENCY")

    # Now fix each child's consistency
    for child_path in child_datasets:
        fix_dataset_consistency(child_path, add_missing_marks=True, remove_orphan_marks=False)

    return check_result


def check_and_fix_all(root_path: Path, fix: bool = False) -> Dict[str, Any]:
    """Check and optionally fix all datasets and collections under root.

    Args:
        root_path: Workspace root
        fix: If True, apply fixes automatically
    """
    log_debug(f"Checking workspace: {root_path}", "CONSISTENCY")

    results = {
        "datasets": {},
        "collections": {},
    }

    for entry in root_path.iterdir():
        if not entry.is_dir():
            continue

        if _is_collection_dir(entry):
            log_debug(f"Found collection: {entry.name}", "CONSISTENCY")
            if fix:
                result = fix_collection_consistency(entry, push_marks_to_children=True)
            else:
                result = check_collection_consistency(entry)
            results["collections"][entry.name] = result
        elif _is_dataset_dir(entry):
            log_debug(f"Found dataset: {entry.name}", "CONSISTENCY")
            if fix:
                result = fix_dataset_consistency(entry, add_missing_marks=True, remove_orphan_marks=False)
            else:
                result = check_dataset_consistency(entry)
            results["datasets"][entry.name] = result

    return results
