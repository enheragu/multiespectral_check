#!/usr/bin/env python3
"""Migrate .image_labels.yaml files to unified marks format.

This script converts the old format:
    marks: {base: reason_str}
    auto_marks: {reason: [bases]}

To the new unified format:
    marks:
      base:
        reason: str
        auto: bool

Run from project root:
    python scripts/migrate_yaml_format.py /path/to/workspace

Options:
    --dry-run    Show what would be changed without modifying files
    --verbose    Show detailed progress
"""
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml


CACHE_FILENAME = ".image_labels.yaml"


def load_yaml_file(path: Path) -> Dict[str, Any]:
    """Load YAML file safely."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"  WARNING: Failed to load {path}: {e}")
        return {}


def save_yaml_file(path: Path, data: Dict[str, Any]) -> bool:
    """Save YAML file with proper formatting."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return True
    except Exception as e:
        print(f"  ERROR: Failed to save {path}: {e}")
        return False


def needs_migration(data: Dict[str, Any]) -> bool:
    """Check if a cache file needs migration."""
    marks = data.get("marks", {})

    # If marks is empty, check if auto_marks exists (needs cleanup)
    if "auto_marks" in data:
        return True

    # Check if any mark entry is a string (old format)
    for base, entry in marks.items():
        if isinstance(entry, str):
            return True
        # New format should be dict with 'reason' key
        if isinstance(entry, dict) and "reason" in entry:
            continue
        # Unknown format
        return True

    return False


def migrate_cache_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate cache data to unified format."""
    old_marks = data.get("marks", {})
    old_auto_marks = data.get("auto_marks", {})

    # Build set of auto bases from old auto_marks
    auto_bases: Set[str] = set()
    auto_base_reason: Dict[str, str] = {}  # base -> reason (for auto marks)

    if isinstance(old_auto_marks, dict):
        for reason, bases in old_auto_marks.items():
            if isinstance(bases, (list, set)):
                for base in bases:
                    auto_bases.add(base)
                    auto_base_reason[base] = reason

    # Convert marks to new format
    new_marks: Dict[str, Dict[str, Any]] = {}

    for base, mark_entry in old_marks.items():
        if isinstance(mark_entry, dict) and "reason" in mark_entry:
            # Already in new format
            new_marks[base] = mark_entry
        elif isinstance(mark_entry, str):
            # Old format: string reason
            reason = mark_entry
            is_auto = base in auto_bases
            new_marks[base] = {"reason": reason, "auto": is_auto}
        else:
            # Unknown format, skip
            continue

    # Add any auto marks that weren't in marks
    for base in auto_bases:
        if base not in new_marks:
            reason = auto_base_reason.get(base, "unknown")
            new_marks[base] = {"reason": reason, "auto": True}

    # Update data
    data["marks"] = new_marks

    # Remove legacy fields
    if "auto_marks" in data:
        del data["auto_marks"]
    if "auto_counts" in data:
        del data["auto_counts"]

    return data


def find_cache_files(root: Path) -> List[Path]:
    """Find all .image_labels.yaml files recursively."""
    cache_files = []

    for path in root.rglob(CACHE_FILENAME):
        cache_files.append(path)

    return sorted(cache_files)


def migrate_file(path: Path, dry_run: bool, verbose: bool) -> bool:
    """Migrate a single cache file. Returns True if migrated."""
    data = load_yaml_file(path)

    if not data:
        if verbose:
            print(f"  SKIP: {path} (empty or invalid)")
        return False

    if not needs_migration(data):
        if verbose:
            print(f"  OK: {path} (already migrated)")
        return False

    # Migrate
    old_mark_count = len(data.get("marks", {}))
    old_auto_count = sum(len(v) for v in data.get("auto_marks", {}).values() if isinstance(v, (list, set)))

    migrated_data = migrate_cache_data(data)
    new_mark_count = len(migrated_data.get("marks", {}))

    if dry_run:
        print(f"  WOULD MIGRATE: {path}")
        print(f"    Old: {old_mark_count} marks + {old_auto_count} auto_marks")
        print(f"    New: {new_mark_count} marks (unified)")
        return True

    if save_yaml_file(path, migrated_data):
        print(f"  MIGRATED: {path}")
        if verbose:
            print(f"    {old_mark_count} marks + {old_auto_count} auto_marks -> {new_mark_count} unified marks")
        return True

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Migrate .image_labels.yaml files to unified marks format"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to workspace or dataset directory to migrate"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress"
    )

    args = parser.parse_args()

    if not args.path.exists():
        print(f"ERROR: Path does not exist: {args.path}")
        sys.exit(1)

    print(f"Scanning for {CACHE_FILENAME} files in: {args.path}")
    if args.dry_run:
        print("DRY RUN - no files will be modified")
    print()

    cache_files = find_cache_files(args.path)
    print(f"Found {len(cache_files)} cache files")
    print()

    migrated_count = 0
    for path in cache_files:
        if migrate_file(path, args.dry_run, args.verbose):
            migrated_count += 1

    print()
    print(f"{'Would migrate' if args.dry_run else 'Migrated'}: {migrated_count} files")
    print(f"Already up-to-date: {len(cache_files) - migrated_count} files")


if __name__ == "__main__":
    main()
