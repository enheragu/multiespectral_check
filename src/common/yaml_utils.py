"""Utilities for YAML serialization with compact, human-readable output.

This module provides functions to read and write YAML files with smart formatting:
- Dictionaries containing only basic types are written in flow style: {key: value}
- Lists of basic types are written in flow style: [a, b, c]
- Complex nested structures use block style for readability
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import yaml


def get_timestamp_fields() -> Dict[str, Any]:
    """Generate last_updated timestamp fields for YAML persistence.

    Returns:
        Dict with 'last_updated' (float epoch) and 'last_updated_str' (human readable).

    Example output:
        {
            "last_updated": 1768477715.2098563,
            "last_updated_str": "2026-01-15 12:48:35"
        }
    """
    now = time.time()
    return {
        "last_updated": now,
        "last_updated_str": datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
    }


def get_timestamp_tuple() -> Tuple[float, str]:
    """Generate last_updated timestamp as tuple.

    Returns:
        Tuple of (epoch_float, human_readable_str).
    """
    now = time.time()
    return now, datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")


def _is_basic_types(values: Any) -> bool:
    """Check if all values are of basic serializable types."""
    basic_types = (int, str, bool, float, type(None))
    if isinstance(values, dict):
        return all(isinstance(v, basic_types) for v in values.values())
    if isinstance(values, (list, tuple)):
        return all(isinstance(v, basic_types) for v in values)
    return isinstance(values, basic_types)


def _is_compact_dict(data: dict) -> bool:
    """Check if a dict should be written in flow style (one line)."""
    if not data:
        return True
    # Flow style for dicts with only basic values and not too many items
    if len(data) > 6:
        return False
    return _is_basic_types(data)


def _is_compact_list(data: list) -> bool:
    """Check if a list should be written in flow style (one line)."""
    if not data:
        return True
    # Flow style for short lists of basic types
    if len(data) > 20:
        return False
    return _is_basic_types(data)


class CompactDumper(yaml.SafeDumper):
    """Custom YAML dumper that uses flow style for simple structures."""

    pass


def _represent_dict(dumper: CompactDumper, data: dict) -> yaml.Node:
    """Represent dict with flow style for simple dicts."""
    if _is_compact_dict(data):
        return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=True)
    return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=False)


def _represent_list(dumper: CompactDumper, data: list) -> yaml.Node:
    """Represent list with flow style for simple lists."""
    if _is_compact_list(data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)


# Register custom representers
CompactDumper.add_representer(dict, _represent_dict)
CompactDumper.add_representer(list, _represent_list)


def load_yaml(file_path: Union[str, Path]) -> dict:
    """Load a YAML file and return its contents as a dict.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Parsed YAML content as dict. Returns empty dict on error.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            result = yaml.safe_load(f)
            return result if isinstance(result, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


def save_yaml(file_path: Union[str, Path], data: Any, *, sort_keys: bool = False) -> bool:
    """Save data to a YAML file with compact formatting.

    Args:
        file_path: Path to write the YAML file.
        data: Data structure to serialize.
        sort_keys: Whether to sort dictionary keys.

    Returns:
        True if successful, False on error.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(
                data,
                f,
                Dumper=CompactDumper,
                default_flow_style=False,
                sort_keys=sort_keys,
                allow_unicode=True,
                width=5000,  # Prevent line wrapping in flow style
            )
        return True
    except (OSError, yaml.YAMLError):
        return False


def dumps_yaml(data: Any, *, sort_keys: bool = False) -> str:
    """Serialize data to a YAML string with compact formatting.

    Args:
        data: Data structure to serialize.
        sort_keys: Whether to sort dictionary keys.

    Returns:
        YAML string representation.
    """
    return yaml.dump(
        data,
        Dumper=CompactDumper,
        default_flow_style=False,
        sort_keys=sort_keys,
        allow_unicode=True,
        width=5000,
    )
