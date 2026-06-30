"""Application-wide configuration constants.

Centralizes all configuration values to make them easy to find, modify, and test.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


APP_NAME = "Multispectral Dataset Viewer"
APP_VERSION = "0.11.0"
APP_DESCRIPTION = "GUI for multispectral dataset review, calibration, and labelling."
SUPPORT_EMAIL = "e.heredia@umh.es"
REPO_URL = "https://github.com/enheragu/multiespectral_check"
ISSUES_URL = f"{REPO_URL}/issues"


@dataclass(frozen=True)
class AppConfig:
    """Main application configuration."""

    # Application metadata
    app_name: str = APP_NAME
    app_version: str = APP_VERSION
    app_description: str = APP_DESCRIPTION
    support_email: str = SUPPORT_EMAIL
    repo_url: str = REPO_URL
    issues_url: str = ISSUES_URL

    # Dataset defaults
    default_dataset_dir: Path = Path("")

    # Pattern sweep
    patterns_dir: Path = Path(__file__).resolve().parent.parent / "config" / "patterns"
    pattern_match_threshold: float = 0.85

    # Calibration settings
    chessboard_size: Tuple[int, int] = (7, 7)
    chessboard_square_size_mm: float = 60.0  # Physical side length of each chessboard square
    default_parallax_depth_m: float = 10.0   # Assumed scene depth for auto-parallax (metres)
    calibration_prefetch_limit: int = 6
    calibration_toggle_shortcut: str = "Ctrl+Shift+C"

    # Overlay settings
    overlay_cache_limit: int = 24

    # Duplicate detection
    signature_threshold: float = 0.001  # Lower = more strict (fewer duplicates), Higher = less strict
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
    progress_task_solver: str = "calibration-solver"
    progress_task_extrinsic: str = "extrinsic-solver"
    progress_task_save: str = "cache-save"
    progress_task_workspace_scan: str = "workspace-scan"
    progress_task_workspace_sweep: str = "workspace-sweep"
    progress_task_workspace_reset: str = "workspace-reset"
    progress_task_quality: str = "quality-scan"
    progress_task_patterns: str = "pattern-scan"
    progress_task_label_detect: str = "label-detect"
    progress_task_label_dataset: str = "label-dataset"

    # Cancel action labels
    @property
    def cancel_action_labels(self) -> dict[str, str]:
        """Labels shown when canceling each task type."""
        return {
            self.progress_task_detection: "Cancelling chessboard detection",
            self.progress_task_signatures: "Cancelling duplicate sweep",
            self.progress_task_solver: "Cancelling calibration solve",
            self.progress_task_extrinsic: "Cancelling stereo solve",
            self.progress_task_workspace_scan: "Scanning workspace",
            self.progress_task_workspace_sweep: "Cancelling workspace sweep",
            self.progress_task_workspace_reset: "Cancelling workspace reset",
            self.progress_task_quality: "Cancelling quality sweep",
            self.progress_task_patterns: "Cancelling pattern sweep",
            self.progress_task_label_detect: "Cancelling detection",
            self.progress_task_label_dataset: "Cancelling dataset labelling",
        }

    # Calibration files
    calibration_intrinsic_filename: str = "calibration_intrinsic.yaml"
    calibration_extrinsic_filename: str = "calibration_extrinsic.yaml"
    calibration_errors_filename: str = ".calibration_errors_cached.yaml"  # Hidden cache file

    # Intrinsic outlier rejection: iterative drop of views whose per-view reprojection
    # error is a robust (median + k*1.4826*MAD) outlier, refitting after each round.
    # k=2.5 ≈ 3.7×MAD; the panel highlights at 2.5×raw-MAD (~1.7σ), so this keeps
    # slightly more views than the panel threshold while still being tighter than the
    # previous k=3.5 (≈5.2×MAD).
    intrinsic_reject_max_iters: int = 5
    intrinsic_reject_k_mad: float = 2.5
    intrinsic_reject_floor_px: float = 0.5
    intrinsic_reject_min_views: int = 6
    # Hard ceiling: a view whose reprojection error exceeds this is ALWAYS rejected, even if the
    # robust (median + k*MAD) band would keep it. Catches absurd views when the whole distribution
    # is poor (the relative MAD test misses them). Set high enough never to fire on a good fit.
    intrinsic_reject_ceiling_px: float = 1.7
    # Intrinsic precision (RMS) saturates around 30-50 views, while cv2.calibrateCamera
    # cost is superlinear in view count (~30 views=2.3s, 100=26s, ~500≈1h). Cap the number
    # of views fed to the solver to keep the same precision in a couple of minutes. The
    # compute-calibration popup uses this as the default cap; view_selection.py picks the
    # best N per channel before the solve.
    intrinsic_max_views_default: int = 150

    # Extrinsic outlier rejection: iterative drop of pairs whose stereo reprojection
    # error is a robust (median + k*1.4826*MAD) outlier, refitting after each round.
    extrinsic_reject_max_iters: int = 5
    extrinsic_reject_k_mad: float = 2.5
    extrinsic_reject_floor_px: float = 0.5
    extrinsic_reject_min_pairs: int = 10
    # Hard ceiling: a pair whose stereo reprojection error exceeds this is ALWAYS rejected (see
    # intrinsic_reject_ceiling_px). For LWIR+visible a pair above this is typically motion-blurred
    # (long LWIR exposure with a moving board); for good stereo it never fires.
    extrinsic_reject_ceiling_px: float = 7.0
    # Stereo view cap: cv2.stereoCalibrate is superlinear (~quadratic) in pair count, while the
    # rigid transform saturates with far fewer good pairs. The compute-extrinsic popup uses this as
    # the default cap; view_selection.py picks the best N pairs (direct corners + board-pose
    # coverage) before the solve. Both-channel pairs are scarce, so the default is generous.
    extrinsic_max_pairs_default: int = 200

    # Blur/motion sweep auto-marking (Dataset -> Detect Delete Candidates -> blur/motion sweep).
    # Thresholds are robust (median +/- k*1.4826*MAD) computed per channel over the whole dataset,
    # so they adapt to each set's sharpness. These are heuristics meant to SUGGEST candidates for
    # review, not a ground-truth blur gate -- auto-marks stay reviewable/undoable in the GUI.
    # Skip auto-marking unless at least this many valid images were scanned (robust median/MAD
    # need enough data; on tiny sets the stats are meaningless).
    quality_sweep_min_samples: int = 12
    # Blurry: laplacian variance below median - k*sigma, AND below a fraction of the median. The
    # relative cap means we only flag a clear drop in sharpness, not the bottom of a uniformly-OK
    # distribution. Lower rel_cap = stricter (fewer, more confident marks).
    quality_blur_k_mad: float = 1.5
    quality_blur_rel_cap: float = 0.6
    # Motion: gradient-energy anisotropy above median + k*sigma, with an absolute floor. A scene
    # with strong directional structure (horizon, road) is anisotropic without any motion, so a
    # motion candidate must ALSO have at-or-below-median sharpness (true motion blur smears detail).
    quality_motion_k_mad: float = 1.5
    quality_motion_floor: float = 2.0

    # Cache files
    summary_cache_filename: str = ".summary_cache.yaml"
    labels_summary_cache_filename: str = ".labels_summary_cache.yaml"
    stereo_alignment_filename: str = ".stereo_alignment.yaml"


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
