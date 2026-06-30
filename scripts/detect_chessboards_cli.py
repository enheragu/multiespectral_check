#!/usr/bin/env python3
"""Headless chessboard detection over a dataset (the GUI's "Detect chessboards" in CLI mode).

Runs the SAME detection pipeline the GUI uses (`analyze_pair_from_paths`) on every
lwir/visible image of a dataset and writes the corners with `save_corners` — the exact
same per-image `calibration/<base>.yaml` files the GUI produces. When the dataset is later
opened in the GUI, `DatasetSession.load()` auto-marks every base that has a corner file
(bottom-up consistency), so everything hooks up: marks, corner overlays and calibration views.

Only bases with at least one channel detected get a corner file (mirrors the GUI, which does
not store a file when nothing is found). A channel with no PNG is handled (detection returns
None for it), so single-channel frames are fine.

Usage:
    python scripts/detect_chessboards_cli.py --dataset /path/to/dataset [--workers 6] [--limit N]
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Make the project source importable when run standalone.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backend.services.calibration_corners_io import save_corners  # noqa: E402
from backend.utils.calibration import analyze_pair_from_paths  # noqa: E402
from config import get_config  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def _list_bases(dataset: Path) -> list[str]:
    """All frame bases that have an lwir and/or visible PNG (union of both channels)."""
    bases: set[str] = set()
    for ch in ("lwir", "visible"):
        ch_dir = dataset / ch
        if ch_dir.is_dir():
            bases.update(p.stem for p in ch_dir.glob("*.png"))
    return sorted(bases)


def _channel_path(dataset: Path, channel: str, base: str):
    p = dataset / channel / f"{base}.png"
    return p if p.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Dataset dir (must contain lwir/ and/or visible/)")
    parser.add_argument("--workers", type=int, default=6, help="Parallel detection workers")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N bases (0 = all)")
    parser.add_argument("--overwrite", action="store_true", help="Re-detect bases that already have a corner file")
    args = parser.parse_args()

    dataset: Path = args.dataset
    if not dataset.is_dir():
        print(f"ERROR: dataset not found: {dataset}", file=sys.stderr)
        return 2

    pattern = get_config().chessboard_size
    calib_dir = dataset / "calibration"
    existing = {p.stem for p in calib_dir.glob("*.yaml")} if calib_dir.is_dir() else set()

    bases = _list_bases(dataset)
    if not args.overwrite:
        bases = [b for b in bases if b not in existing]
    if args.limit:
        bases = bases[: args.limit]

    print(f"Dataset:  {dataset}")
    print(f"Pattern:  {pattern}  |  workers: {args.workers}  |  already-detected (kept): {len(existing)}")
    print(f"To process: {len(bases)} bases")
    if not bases:
        print("Nothing to do.")
        return 0

    def detect(base: str):
        lw = _channel_path(dataset, "lwir", base)
        vs = _channel_path(dataset, "visible", base)
        results, corners = analyze_pair_from_paths(base, lw, vs, pattern)
        return base, results, corners

    stats = {"both": 0, "lwir_only": 0, "visible_only": 0, "none": 0, "saved": 0}
    start = time.perf_counter()
    bar = tqdm(total=len(bases), unit="img", ncols=100) if tqdm else None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(detect, b) for b in bases]
        for fut in as_completed(futures):
            base, results, corners = fut.result()
            lw_ok = results.get("lwir") is True
            vs_ok = results.get("visible") is True
            if lw_ok and vs_ok:
                stats["both"] += 1
            elif lw_ok:
                stats["lwir_only"] += 1
            elif vs_ok:
                stats["visible_only"] += 1
            else:
                stats["none"] += 1
            if lw_ok or vs_ok:
                # Same save the GUI does (corners + per-channel meta + image_size).
                save_corners(dataset, base, corners, image_sizes=corners.get("image_size"))
                stats["saved"] += 1
            if bar:
                bar.update(1)
                bar.set_postfix(both=stats["both"], saved=stats["saved"], none=stats["none"])
    if bar:
        bar.close()

    elapsed = time.perf_counter() - start
    print(
        f"\nDone in {elapsed:.1f}s — saved {stats['saved']} corner files "
        f"(both={stats['both']}, lwir_only={stats['lwir_only']}, visible_only={stats['visible_only']}, "
        f"none={stats['none']})"
    )
    print(f"Corner files written to: {calib_dir}")
    print("Open the dataset in the GUI — bases with corner files are auto-marked as calibration candidates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
