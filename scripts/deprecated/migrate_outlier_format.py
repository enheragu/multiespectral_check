#!/usr/bin/env python3
"""Migrate calibration outlier format in YAML files.

Old format:
    calibration:
      000123:
        marked: true
        outlier_lwir: false
        outlier_visible: false
        outlier_stereo: false
        results: {lwir: true, visible: true}

New format:
    calibration:
      000123:
        marked: true
        outlier:
          lwir: false
          visible: false
          stereo: false
        results: {lwir: true, visible: true}

Usage:
    python scripts/migrate_outlier_format.py /path/to/workspace

The script finds all .image_labels.yaml files and migrates them in place.
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml


def needs_migration(data: dict) -> bool:
    """Check if any calibration entry uses old outlier format."""
    calibration = data.get("calibration", {})
    if not isinstance(calibration, dict):
        return False

    for base, entry in calibration.items():
        if not isinstance(entry, dict):
            continue
        # Old format has outlier_lwir/outlier_visible/outlier_stereo at top level
        if any(key in entry for key in ("outlier_lwir", "outlier_visible", "outlier_stereo")):
            return True
    return False


def migrate_file(filepath: Path, dry_run: bool = False) -> bool:
    """Migrate a single YAML file. Returns True if migrated."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"  ERROR reading {filepath}: {e}")
        return False

    if not isinstance(data, dict):
        return False

    if not needs_migration(data):
        return False

    calibration = data.get("calibration", {})
    migrated_count = 0

    for base, entry in list(calibration.items()):
        if not isinstance(entry, dict):
            continue

        # Check for old format keys
        has_old_format = any(key in entry for key in ("outlier_lwir", "outlier_visible", "outlier_stereo"))
        if not has_old_format:
            continue

        # Extract old values
        outlier_lwir = entry.pop("outlier_lwir", False)
        outlier_visible = entry.pop("outlier_visible", False)
        outlier_stereo = entry.pop("outlier_stereo", False)

        # Create new format
        entry["outlier"] = {
            "lwir": bool(outlier_lwir),
            "visible": bool(outlier_visible),
            "stereo": bool(outlier_stereo),
        }
        migrated_count += 1

    if migrated_count == 0:
        return False

    if not dry_run:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            print(f"  ✓ Migrated {filepath.name}: {migrated_count} entries")
        except Exception as e:
            print(f"  ERROR writing {filepath}: {e}")
            return False
    else:
        print(f"  [DRY-RUN] Would migrate {filepath.name}: {migrated_count} entries")

    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/migrate_outlier_format.py <workspace_path> [--dry-run]")
        sys.exit(1)

    workspace_path = Path(sys.argv[1])
    dry_run = "--dry-run" in sys.argv

    if not workspace_path.exists():
        print(f"Error: Path does not exist: {workspace_path}")
        sys.exit(1)

    print(f"Scanning for .image_labels.yaml files in: {workspace_path}")
    if dry_run:
        print("[DRY-RUN MODE - no files will be modified]")
    print()

    yaml_files = list(workspace_path.rglob(".image_labels.yaml"))
    print(f"Found {len(yaml_files)} YAML files")
    print()

    migrated = 0
    for filepath in sorted(yaml_files):
        if migrate_file(filepath, dry_run=dry_run):
            migrated += 1

    print()
    print(f"Migration complete: {migrated}/{len(yaml_files)} files migrated")


if __name__ == "__main__":
    main()
