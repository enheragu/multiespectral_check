"""Batch operations runner for workspace/collection sweeps.

Provides unified interface for running sweeps across multiple datasets,
with progress reporting, cancellation, and parallel execution.

This eliminates code duplication between workspace_dialog, collection sweeps,
and other batch operations.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Callable, List, Optional, TYPE_CHECKING

from common.log_utils import log_debug

if TYPE_CHECKING:
    from backend.services.dataset_session import DatasetSession
    from backend.services.patterns.pattern_matcher import PatternMatcher
    from backend.services.progress_reporter import ProgressReporter


@dataclass
class SweepConfig:
    """Configuration for batch sweep operations."""
    run_missing: bool = False
    run_duplicates: bool = False
    run_quality: bool = False
    run_patterns: bool = False
    restore_all: bool = False
    delete_marked: bool = False
    matcher: Optional["PatternMatcher"] = None


@dataclass
class SweepResult:
    """Result from processing a single dataset."""
    dataset_name: str
    restored: int = 0
    deleted: int = 0
    duplicates: int = 0
    blurry: int = 0
    motion: int = 0
    patterns: int = 0
    success: bool = True
    error: Optional[str] = None

    def summary_parts(self) -> List[str]:
        """Return list of non-zero result descriptions."""
        parts = []
        if self.restored:
            parts.append(f"restored {self.restored}")
        if self.deleted:
            parts.append(f"deleted {self.deleted}")
        if self.duplicates:
            parts.append(f"dups {self.duplicates}")
        if self.blurry or self.motion:
            parts.append(f"blur {self.blurry}/motion {self.motion}")
        if self.patterns:
            parts.append(f"patterns {self.patterns}")
        return parts


@dataclass
class BatchSweepResult:
    """Aggregated result from processing multiple datasets."""
    total_datasets: int = 0
    processed: int = 0
    failed: int = 0
    restored: int = 0
    deleted: int = 0
    duplicates: int = 0
    blurry: int = 0
    motion: int = 0
    patterns: int = 0
    results: List[SweepResult] = field(default_factory=list)

    def add(self, result: SweepResult) -> None:
        """Add a single dataset result to the batch."""
        self.results.append(result)
        self.processed += 1
        if not result.success:
            self.failed += 1
        self.restored += result.restored
        self.deleted += result.deleted
        self.duplicates += result.duplicates
        self.blurry += result.blurry
        self.motion += result.motion
        self.patterns += result.patterns


def run_sweeps_on_session(
    session: "DatasetSession",
    config: SweepConfig,
    cancel_check: Optional[Callable[[], bool]] = None,
    reporter: Optional["ProgressReporter"] = None,
) -> SweepResult:
    """Run configured sweeps on a single dataset session.

    This is the core sweep logic - used by both single dataset and batch operations.

    Args:
        session: Loaded DatasetSession
        config: Which sweeps to run
        cancel_check: Optional cancellation check
        reporter: Optional progress reporter

    Returns:
        SweepResult with counts from each operation
    """
    result = SweepResult(dataset_name=session.dataset_path.name if session.dataset_path else "unknown")

    def is_cancelled() -> bool:
        if cancel_check and cancel_check():
            return True
        if reporter and reporter.is_cancelled():
            return True
        return False

    try:
        if config.restore_all and not is_cancelled():
            result.restored = session.restore_from_trash()

        if config.run_missing and not is_cancelled():
            session._auto_mark_missing_pairs()  # noqa: SLF001
            session.state.rebuild_reason_counts()
            session.mark_cache_dirty()

        if config.run_duplicates and not is_cancelled():
            result.duplicates = session.run_duplicate_sweep(
                cancel_check=is_cancelled,
                reporter=reporter,
            )

        if config.run_quality and not is_cancelled():
            result.blurry, result.motion = session.run_quality_sweep(
                cancel_check=is_cancelled,
                reporter=reporter,
            )

        if config.run_patterns and config.matcher and config.matcher.has_patterns and not is_cancelled():
            result.patterns = session.run_pattern_sweep(
                config.matcher,
                cancel_check=is_cancelled,
            )

        if config.delete_marked and not is_cancelled():
            outcome = session.delete_marked_entries()
            result.deleted = outcome.moved

    except Exception as e:
        result.success = False
        result.error = str(e)
        log_debug(f"Sweep error on {result.dataset_name}: {e}", "BATCH")

    return result


def run_batch_sweeps(
    dataset_paths: List[Path],
    config: SweepConfig,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    on_dataset_complete: Optional[Callable[[SweepResult], None]] = None,
    max_workers: int = 2,
    use_tqdm: bool = True,
) -> BatchSweepResult:
    """Run sweeps on multiple datasets in parallel.

    Args:
        dataset_paths: List of dataset directories to process
        config: Sweep configuration
        cancel_check: Optional cancellation check
        on_progress: Callback(completed, total, message) for UI updates
        on_dataset_complete: Callback(result) after each dataset
        max_workers: Max parallel workers
        use_tqdm: Whether to show tqdm progress bar

    Returns:
        BatchSweepResult with aggregated counts
    """
    from backend.services.dataset_session import DatasetSession
    from backend.services.cache_writer import write_cache_payload

    batch_result = BatchSweepResult(total_datasets=len(dataset_paths))

    if not dataset_paths:
        return batch_result

    # Thread-safe progress tracking
    lock = Lock()
    completed = 0

    # Optional tqdm
    pbar = None
    if use_tqdm:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(dataset_paths), desc="Batch sweep", unit="ds", leave=False)
        except ImportError:
            pass

    def process_dataset(path: Path) -> SweepResult:
        nonlocal completed

        if cancel_check and cancel_check():
            return SweepResult(dataset_name=path.name, success=False, error="Cancelled")

        # Load dataset
        session = DatasetSession()
        if not session.load(path):
            result = SweepResult(dataset_name=path.name, success=False, error="Load failed")
            with lock:
                completed += 1
                if pbar:
                    pbar.update(1)
            return result

        # Run sweeps
        result = run_sweeps_on_session(session, config, cancel_check=cancel_check)

        # Save cache
        payload = session.snapshot_cache_payload()
        if payload:
            write_cache_payload(payload)

        # Update progress
        with lock:
            completed += 1
            if pbar:
                pbar.update(1)
                pbar.set_postfix_str(", ".join(result.summary_parts()) or "ok")

        if on_progress:
            try:
                msg = f"{result.dataset_name}: " + (", ".join(result.summary_parts()) or "ok")
                on_progress(completed, len(dataset_paths), msg)
            except RuntimeError:
                pass

        if on_dataset_complete:
            try:
                on_dataset_complete(result)
            except RuntimeError:
                pass

        return result

    # Run in parallel
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_dataset, path): path for path in dataset_paths}

            for future in as_completed(futures):
                if cancel_check and cancel_check():
                    for f in futures:
                        f.cancel()
                    break

                try:
                    result = future.result()
                    batch_result.add(result)
                except Exception as e:
                    path = futures[future]
                    batch_result.add(SweepResult(
                        dataset_name=path.name,
                        success=False,
                        error=str(e),
                    ))
    finally:
        if pbar:
            pbar.close()

    return batch_result


def collect_dataset_paths(
    workspace_path: Path,
    *,
    _include_collections: bool = True,  # TODO: implement filtering by collection
    _recursive: bool = True,  # TODO: implement non-recursive scan
) -> List[Path]:
    """Collect all dataset paths from a workspace.

    Args:
        workspace_path: Root workspace directory
        _include_collections: Whether to include datasets within collections (not yet implemented)
        _recursive: Whether to search recursively (not yet implemented)

    Returns:
        List of dataset directory paths
    """
    from backend.services.workspace_inspector import scan_workspace

    infos = scan_workspace(workspace_path)

    # Extract paths - only leaf datasets (not collections themselves)
    paths = []
    for info in infos:
        # WorkspaceDatasetInfo has 'kind' field: 'dataset' or 'collection'
        if info.kind == "dataset":
            paths.append(info.path)

    return sorted(paths)

