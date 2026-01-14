"""Application-wide configuration constants.

Centralizes all configuration values to make them easy to find, modify, and test.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class AppConfig:
    """Main application configuration."""

    # Dataset defaults
    default_dataset_dir: Path = Path.home() / "umh" / "ros2_ws" / "images_eeha"

    # Pattern sweep
    patterns_dir: Path = Path(__file__).resolve().parent.parent / "config" / "patterns"
    pattern_match_threshold: float = 0.85

    # Calibration settings
    chessboard_size: Tuple[int, int] = (7, 7)
    calibration_prefetch_limit: int = 6
    calibration_toggle_shortcut: str = "Ctrl+Shift+C"

    # Overlay settings
    overlay_cache_limit: int = 24

    # Duplicate detection
    signature_threshold: float = 0.003  # Lower = more strict (fewer duplicates), Higher = less strict
    signature_size: int = 64

    # Cache settings
    cache_version: int = 1
    cache_max_datasets: int = 5
    cache_large_size_mb: int = 10

    # Calibration detection
    calibration_detection_max_edge: int = 1600

    # Scan timers (milliseconds)
    signature_scan_timer_interval_ms: int = 20
    cache_flush_timer_interval_ms: int = 2000
    calibration_queue_interval_ms: int = 200

    # Progress task identifiers
    progress_task_detection: str = "calibration-detect"
    progress_task_signatures: str = "signature-scan"
    progress_task_refinement: str = "calibration-refine"
    progress_task_solver: str = "calibration-solver"
    progress_task_extrinsic: str = "extrinsic-solver"
    progress_task_save: str = "cache-save"
    progress_task_workspace_scan: str = "workspace-scan"
    progress_task_workspace_sweep: str = "workspace-sweep"
    progress_task_workspace_reset: str = "workspace-reset"
    progress_task_quality: str = "quality-scan"

    # Cancel action labels
    @property
    def cancel_action_labels(self) -> dict[str, str]:
        """Labels shown when canceling each task type."""
        return {
            self.progress_task_detection: "Cancelling chessboard detection",
            self.progress_task_signatures: "Cancelling duplicate sweep",
            self.progress_task_refinement: "Cancelling corner refinement",
            self.progress_task_solver: "Cancelling calibration solve",
            self.progress_task_extrinsic: "Cancelling stereo solve",
            self.progress_task_workspace_scan: "Scanning workspace",
            self.progress_task_workspace_sweep: "Cancelling workspace sweep",
            self.progress_task_workspace_reset: "Cancelling workspace reset",
            self.progress_task_quality: "Cancelling quality sweep",
        }

    # Calibration files
    calibration_intrinsic_filename: str = "calibration_intrinsic.yaml"
    calibration_extrinsic_filename: str = "calibration_extrinsic.yaml"


# Global singleton instance
_CONFIG: AppConfig | None = None


def get_config() -> AppConfig:
    """Get the global application configuration.

    Returns:
        AppConfig instance (singleton)
    """
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = AppConfig()
    return _CONFIG


def reset_config() -> None:
    """Reset configuration to default (mainly for testing)."""
    global _CONFIG
    _CONFIG = None
