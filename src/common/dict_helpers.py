"""Safe dictionary access helpers to reduce intermediate variables and checks."""
from typing import Any, List, Optional, TypeVar, Union, cast


T = TypeVar('T')


def get_dict_path(
    data: Any,
    path: Union[str, List[str]],
    default: Optional[T] = None,
    expected_type: Optional[type] = None,
) -> Optional[T]:
    """Safely navigate nested dictionary structure.

    Args:
        data: Root dictionary or object
        path: Key path as string "key1.key2.key3" or list ["key1", "key2", "key3"]
        default: Default value if path not found or type mismatch
        expected_type: Expected type for final value (validates type)

    Returns:
        Value at path or default if not found/type mismatch

    Examples:
        >>> data = {"dataset_info": {"sweep_flags": {"duplicates": True}}}
        >>> get_dict_path(data, "dataset_info.sweep_flags.duplicates")
        True
        >>> get_dict_path(data, ["dataset_info", "sweep_flags", "missing"], default=False)
        False
        >>> get_dict_path(data, "dataset_info.note", default="", expected_type=str)
        ""
    """
    # Convert string path to list
    if isinstance(path, str):
        keys = path.split(".")
    else:
        keys = path

    if not keys:
        return default

    # Navigate through the path
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default

        if key not in current:
            return default

        current = current[key]

    # Type validation if requested
    if expected_type is not None:
        if not isinstance(current, expected_type):
            return default

    return cast(Optional[T], current)


def set_dict_path(
    data: dict,
    path: Union[str, List[str]],
    value: Any,
    create_missing: bool = True,
) -> bool:
    """Safely set value in nested dictionary structure.

    Args:
        data: Root dictionary (will be modified)
        path: Key path as string "key1.key2.key3" or list ["key1", "key2", "key3"]
        value: Value to set
        create_missing: If True, create missing intermediate dicts

    Returns:
        True if successful, False if path invalid or create_missing=False and path doesn't exist

    Examples:
        >>> data = {}
        >>> set_dict_path(data, "dataset_info.sweep_flags.duplicates", True)
        True
        >>> data
        {'dataset_info': {'sweep_flags': {'duplicates': True}}}
    """
    # Convert string path to list
    if isinstance(path, str):
        keys = path.split(".")
    else:
        keys = path

    if not keys:
        return False

    # Navigate to parent of target key
    current = data
    for key in keys[:-1]:
        if key not in current:
            if not create_missing:
                return False
            current[key] = {}

        if not isinstance(current[key], dict):
            return False

        current = current[key]

    # Set the final value
    current[keys[-1]] = value
    return True


def has_dict_path(data: Any, path: Union[str, List[str]]) -> bool:
    """Check if path exists in nested dictionary.

    Args:
        data: Root dictionary or object
        path: Key path as string "key1.key2.key3" or list ["key1", "key2", "key3"]

    Returns:
        True if path exists, False otherwise

    Examples:
        >>> data = {"dataset_info": {"sweep_flags": {"duplicates": True}}}
        >>> has_dict_path(data, "dataset_info.sweep_flags.duplicates")
        True
        >>> has_dict_path(data, "dataset_info.sweep_flags.missing")
        False
    """
    # Convert string path to list
    if isinstance(path, str):
        keys = path.split(".")
    else:
        keys = path

    if not keys:
        return False

    current = data
    for key in keys:
        if not isinstance(current, dict):
            return False

        if key not in current:
            return False

        current = current[key]

    return True


def merge_dict_path(
    target: dict,
    source: dict,
    path: Union[str, List[str]],
    overwrite: bool = False,
) -> bool:
    """Merge dictionary from source path into target path.

    Args:
        target: Target dictionary (will be modified)
        source: Source dictionary to read from
        path: Key path in both dicts
        overwrite: If True, overwrite existing keys; if False, only add missing keys

    Returns:
        True if successful, False if path invalid

    Examples:
        >>> target = {"dataset_info": {"note": "old"}}
        >>> source = {"dataset_info": {"note": "new", "updated": 123}}
        >>> merge_dict_path(target, source, "dataset_info", overwrite=False)
        True
        >>> target
        {'dataset_info': {'note': 'old', 'updated': 123}}
    """
    # Get source value
    source_value = get_dict_path(source, path)
    if source_value is None:
        return False

    if not isinstance(source_value, dict):
        # Atomic value, just set it
        return set_dict_path(target, path, source_value)

    # Get or create target dict
    target_value = get_dict_path(target, path)
    if target_value is None:
        return set_dict_path(target, path, source_value.copy())

    if not isinstance(target_value, dict):
        if overwrite:
            return set_dict_path(target, path, source_value.copy())
        return False

    # Merge dictionaries
    for key, value in source_value.items():
        if key not in target_value or overwrite:
            target_value[key] = value

    return True


__all__ = [
    "get_dict_path",
    "set_dict_path",
    "has_dict_path",
    "merge_dict_path",
]
