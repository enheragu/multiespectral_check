"""Utility for writing formatted tables to log files using tabulate."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tabulate import tabulate

from common.log_utils import log_info, log_warning


def write_table_to_log(
    table_data: List[List[Any]],
    headers: List[str],
    log_name: str,
    *,
    tablefmt: str = "grid",
    logs_dir: Optional[Path] = None,
    overwrite: bool = True,
) -> Optional[Path]:
    """Write a formatted table to a log file.

    Args:
        table_data: List of rows, where each row is a list of cell values
        headers: List of column headers
        log_name: Base name for the log file (e.g., "workspace_sweep", "calibration_outliers")
        tablefmt: Table format for tabulate (default: "grid"). Options: grid, simple, fancy_grid, etc.
        logs_dir: Directory to write logs (default: <project_root>/logs/)
        overwrite: If True, overwrite existing file. If False, append timestamp to filename.

    Returns:
        Path to the written log file, or None if writing failed
    """
    # Determine logs directory
    if logs_dir is None:
        # Default: <project_root>/logs/
        project_root = Path(__file__).resolve().parent.parent.parent
        logs_dir = project_root / "logs"

    # Ensure logs directory exists
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Determine filename
    if overwrite:
        filename = f"{log_name}.txt"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{log_name}_{timestamp}.txt"

    log_path = logs_dir / filename

    try:
        # Format table using tabulate
        table_str = tabulate(table_data, headers=headers, tablefmt=tablefmt)

        # Add header with metadata
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header_lines = [
            "=" * 80,
            f"Table: {log_name}",
            f"Generated: {timestamp_str}",
            f"Rows: {len(table_data)}",
            "=" * 80,
            "",
        ]

        # Write to file
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(header_lines))
            f.write(table_str)
            f.write("\n")

        log_info(f"Table saved to {log_path}", "TABLE_WRITER")
        return log_path

    except Exception as e:
        log_warning(f"Failed to write table to {log_path}: {e}", "TABLE_WRITER")
        return None


def format_workspace_table(infos: List[Any]) -> tuple[List[List[Any]], List[str]]:
    """Format workspace dataset infos into table data for tabulate.

    Args:
        infos: List of WorkspaceDatasetInfo objects

    Returns:
        Tuple of (table_data, headers) ready for tabulate
    """
    headers = [
        "Name",
        "Type",
        "Pairs",
        "Manual",
        "Auto",
        "Removed",
        "Calib Marked",
        "Calib Complete",
        "Sweeps Done",
    ]

    table_data = []
    for info in infos:
        # Format sweep status
        sweep_parts = []
        if info.stats.sweep_duplicates_done:
            sweep_parts.append("D")
        if info.stats.sweep_quality_done:
            sweep_parts.append("Q")
        if info.stats.sweep_patterns_done:
            sweep_parts.append("P")
        sweep_status = ", ".join(sweep_parts) if sweep_parts else "—"

        row = [
            info.name,
            "Collection" if info.is_collection else "Dataset",
            info.stats.total_pairs,
            info.stats.tagged_manual,
            info.stats.tagged_auto,
            info.stats.removed_total,
            info.stats.calibration_marked,
            f"{info.stats.calibration_both}/{info.stats.calibration_marked}" if info.stats.calibration_marked > 0 else "—",
            sweep_status,
        ]
        table_data.append(row)

    return table_data, headers


def format_outliers_table(outliers_data: Dict[str, List[Dict[str, Any]]]) -> tuple[List[List[Any]], List[str]]:
    """Format calibration outliers data into table data for tabulate.

    Args:
        outliers_data: Dict mapping channel to list of outlier entries
                      Each entry: {"base": str, "error": float, "threshold": float}

    Returns:
        Tuple of (table_data, headers) ready for tabulate
    """
    headers = ["Base", "Channel", "Reprojection Error", "Threshold", "Severity"]

    table_data = []
    for channel, outliers in outliers_data.items():
        for outlier in outliers:
            base = outlier.get("base", "?")
            error = outlier.get("error", 0.0)
            threshold = outlier.get("threshold", 0.0)

            # Calculate severity (how much over threshold)
            if threshold > 0:
                severity_ratio = error / threshold
                if severity_ratio > 2.0:
                    severity = "CRITICAL"
                elif severity_ratio > 1.5:
                    severity = "HIGH"
                else:
                    severity = "MODERATE"
            else:
                severity = "UNKNOWN"

            row = [
                base,
                channel,
                f"{error:.4f}",
                f"{threshold:.4f}",
                severity,
            ]
            table_data.append(row)

    # Sort by error (descending)
    table_data.sort(key=lambda r: float(r[2]), reverse=True)

    return table_data, headers
