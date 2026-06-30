# Chessboard Calibration Pipeline

This document covers the full calibration pipeline: first **detection** — finding chessboard corners in each image — then the **solve** — recovering each camera's intrinsics and the LWIR↔Visible extrinsic transform. It works with any chessboard pattern size and any image resolution; most sections end with notes on behaviours specific to our dataset.

---

## Detection Pipeline

Each image goes through the following stages in order. Execution stops at the first success.

```
1. findChessboardCornersSB (primary)
   ↓ fails
2. Fast pre-check (2-level, downscaled + CLAHE)
   → Both fail: skip enhancers, go to WithMeta fallback directly
   → At least one passes: run enhancer pipeline
3. Enhancer pipeline (parallel)
   CLAHE | bilateral | unsharp-mask | gamma | LAB-eq | HSV-eq | ...
   ↓ all fail
4. ROI-based SB fallback
   → Old algo locates board → crop → SB on crop (+ stronger CLAHE/histEq)
   ↓ fails
5. WithMeta fallback (interpolation allowed)
   → CLAHE | bilateral | bilateral+CLAHE-gray
   → Refinement: if WithMeta finds board, attempt SB on tight ROI to avoid interpolation
   ↓ fails
Return: not found
```

### Why this order

**SB (stage 1)** is the fastest path and fails fast on no-pattern images. It requires the full N×M grid to be detectable simultaneously.

**Fast pre-check (stage 2)** avoids the expensive enhancer pipeline for images with no board structure. It runs two levels on a downscaled (640-edge) image with CLAHE applied:
- Level 1: `CALIB_CB_FAST_CHECK` — gradient scan, very fast (~5ms).
- Level 2: `CALIB_CB_ADAPTIVE_THRESH` — slower but catches low-contrast boards that FAST_CHECK misses.

Boards too small or too tilted to be detected at 640px still reach WithMeta as a last resort.

**Enhancer pipeline (stage 3)** runs in parallel — any single success stops the rest. Handles various image degradations (blur, poor contrast, overexposure, sensor noise, etc.).

**ROI-based SB (stage 4)** handles a specific failure mode: SB can fail on images with complex structured backgrounds (e.g., glass facades, tiled surfaces) because it builds a saddle-point connectivity graph over the *entire* image and gets confused by false saddle points in the background. The classic `findChessboardCorners` algorithm is robust to background noise. It locates the board and provides bounds for a tight ROI crop (margin ≈ one square width). SB on the cropped region then succeeds.

**WithMeta + ROI refinement (stage 5)** uses `findChessboardCornersSBWithMeta`, which can interpolate missing corners from a partially-detected grid. After finding a valid result (with some interpolated corners), the pipeline immediately retries SB on the tight ROI using stronger contrast preprocessing. This refinement step often converts an interpolated result into a fully-direct one — see *Corner-block patterns* below.

---

## Corner-Block Patterns

For some board images, `findChessboardCornersSBWithMeta` detects the interior corners directly but interpolates the outermost row and/or column. This is called a **corner-block pattern**.

This can happen for several reasons:
- The board's outer frame introduces a slight brightness gradient, weakening the saddle-point response at edge corners.
- On small or distant boards, edge corners may not generate a strong enough local gradient.
- Perspective distortion can reduce contrast asymmetrically.

### Structural check

Not all interpolated patterns are accepted. `_has_bad_interpolation_structure()` enforces:
- At most **one** fully-interpolated row and **one** fully-interpolated column.
- Any such row/column must be on the **boundary** of the grid (first or last).
- Interior fully-interpolated rows/columns are rejected (they would bridge over missing observations with no anchor on either side, making the grid model unreliable).

### Upgrading corner-block to fully-direct

After WithMeta locates the board (36+13 or similar), the tight ROI is re-used to retry SB with stronger preprocessing. In practice, CLAHE with a higher clip limit applied to the ROI often makes the edge corners detectable directly, converting the result to 100% direct corners.

---

## Direct-Corner Ratio Threshold

`_META_MIN_DIRECT_RATIO` sets the minimum fraction of corners that must be directly detected (not interpolated) for a WithMeta result to be accepted.

`_min_direct_corners()` uses `math.ceil` intentionally. For example, with ratio=0.75 and 49 corners:
- `int(49 × 0.75) = 36` — wrong, effectively 0.735 (allows 13 interpolated)
- `ceil(49 × 0.75) = 37` — correct, maximum 12 interpolated

The current value (0.73) is calibrated to allow a single corner-block (last row + last column for our 7×7 pattern). If your pattern or use-case is different, adjust accordingly — the structural check provides an independent safety net.

---

## Why findChessboardCornersSB Fails on Background-Heavy Images

`findChessboardCornersSB` scans the *entire* image for saddle-like structures, builds a connectivity graph, and tries to fit an N×M regular grid. In images with structured backgrounds (glass facades, window grids, tiled floors), many false saddle points are added to the graph. The algorithm cannot isolate the real board and returns False even when the board itself is visually clear.

`findChessboardCorners` (classic algorithm) uses thresholding + contour detection and is less sensitive to background structure, making it a reliable board locator even in cluttered scenes.

---

## Detection Performance Notes

- **Parallel workers**: `max(2, min(8, cpu_count - 2))` enhancement workers; separate pool for workspace operations.
- **No-board images**: fast pre-check exits in ~0.3s; WithMeta fallback adds ~0.5–1.5s for images where the pre-check fails.
- **Board images (easy)**: SB direct ~0.05s.
- **Board images (hard)**: full pipeline (enhancers + ROI + WithMeta) up to ~2.5s.
- **Effective scan time** depends on worker count and dataset composition. As a rough guide, with 6 parallel workers and a mix of board and non-board images, expect ~3–5 minutes per 700-image dataset.

---

## Calibration Solve

Once corners are detected, calibration runs in two independent stages: **intrinsic** (per camera) and **extrinsic** (the rigid LWIR↔Visible transform). Both share an iterative outlier-rejection loop and an optional pre-solve view cap.

### Object points in unit squares

Object points are laid out as a unit grid (`np.mgrid[0:cols, 0:rows]`), **not** scaled by the physical square size. Intrinsics (focal length, principal point, distortion) are scale-invariant, so this is correct for the solve; the physical square size only matters when converting the extrinsic translation to metric units. Keeping unit squares also keeps any downstream board reconstruction consistent with the solver.

### View cap and priority selection

`cv2.calibrateCamera` and `cv2.stereoCalibrate` are **superlinear** in the number of views/pairs, while accuracy **saturates early** — intrinsic RMS flattens around 30–50 views, and the 6-DOF stereo transform needs even fewer *good* pairs. On large datasets (hundreds of views) the full set therefore costs a lot of time for no extra accuracy (an intrinsic solve on ~500 views can take ~1 hour).

At compute time a popup lets you cap how many views (per channel) or pairs to keep, or tick **Use all**. Defaults live in `config.py` (`intrinsic_max_views_default`, `extrinsic_max_pairs_default`).

The cap is a pure function (`calibration/view_selection.py`) over lightweight records — **no images are loaded**, only the already-detected corners. Candidates are ranked by priority:

1. **Fully-direct** corners are preferred over interpolated ones; interpolated views only top up the cap if direct views are insufficient. (A stereo pair counts as "direct" only if *neither* channel was interpolated.)
2. **Detected in both channels** — a small bonus for intrinsic (a single-channel view is still valid); always true for stereo pairs, so a no-op there.
3. **Coverage** — deterministic farthest-point sampling over an intrinsics-free feature vector derived from the board quad: centroid (image position), √area (apparent size → distance proxy) and opposite-side length skew (foreshortening → tilt proxy). This spreads the kept views across image position, distance and angle. When a prior camera matrix exists (re-calibration), true solvePnP pose can be appended.
4. **Sharpness** — optional tiebreak, **off by default** (measuring it would require loading every image, defeating the speed-up).

Indicative `stereoCalibrate` scaling (single thread, `FIX_INTRINSIC`): 50 pairs ≈ 2 s, 100 ≈ 8 s, 200 ≈ 19 s, 400 ≈ 80 s — roughly quadratic. The rejection loop runs the solve several times with the first (full-set) round dominating, so capping that first round is what saves the most.

### Intrinsic calibration

`cv2.calibrateCamera` is run **per channel** (LWIR and Visible), and the two channels solve **in parallel**. Each yields a camera matrix + distortion coefficients and per-view reprojection errors.

### Extrinsic (stereo) calibration

`cv2.stereoCalibrate` is run with `CALIB_FIX_INTRINSIC`: the per-channel intrinsics are held fixed and only the rigid rotation/translation between the two cameras is optimized. This is cheaper per pair than a full solve but still superlinear in pair count. It needs pairs detected on **both** cameras, which are scarcer than per-channel detections.

### Outlier rejection (shared)

Both stages iteratively drop views/pairs whose reprojection error is an outlier and refit, up to `*_reject_max_iters` rounds:

1. **Ceiling first** — anything above an absolute ceiling (`*_reject_ceiling_px`) is dropped *before* the robust statistics are computed, so gross outliers don't inflate the median/MAD and loosen the band.
2. **Robust band** — keep views with error ≤ `median + k · 1.4826 · MAD` (`*_reject_k_mad`), clamped between a per-view floor (`*_reject_floor_px`) and the ceiling.
3. **Stop** when a round removes nothing, when removing more would drop below the minimum (`intrinsic_reject_min_views` / `extrinsic_reject_min_pairs`), or at the max iteration.

The MAD test is *relative*, so the absolute ceiling is what guarantees obviously-bad views/pairs are always dropped even when the whole distribution is poor; the floor prevents over-pruning a good fit. All thresholds are in `config.py`.

### Progress

Each rejection round reports progress: a full line to the terminal (`[calib lwir] iter 3/5 · 268 views · RMS 0.341px`, `[stereo] …`) and a short label in the GUI progress bar. Intrinsic channels report independently as they solve in parallel.

---

## Our Dataset Notes

*These notes reflect the specific hardware and conditions of our LWIR+visible calibration setup. The algorithm is not limited to these parameters.*

- **Cameras**: LWIR at 640×480, visible at 1600×1200.
- **Pattern**: 7×7 inner corners (8×8 chessboard squares).
- **Common failure modes observed**:
  - Building glass facade backgrounds → fixed by ROI-based SB fallback.
  - Corner-block patterns (outermost row + column interpolated) → usually upgraded to fully-direct by ROI refinement.
  - Very small or strongly tilted boards → may only be detectable by WithMeta; the pre-check may reject them at low resolution.
- **Motion blur (LWIR)**: the thermal sensor's long exposure (~33 ms, vs ~0.16 ms for visible) blurs the board when it moves, shifting the detected LWIR corners roughly uniformly relative to the visible ones. This barely affects the per-camera intrinsic fit but inflates the **stereo** reprojection error, so the extrinsic outlier ceiling (`extrinsic_reject_ceiling_px`) is set to drop these blurred pairs.
