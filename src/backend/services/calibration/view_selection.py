"""Priority view selection for intrinsic calibration (pure-python, no Qt).

``cv2.calibrateCamera`` cost is superlinear in the number of views, while the
intrinsic precision (RMS) saturates around 30-50 views.  Solving on ~500 views
therefore takes ~1 hour for no extra accuracy.  This module picks the best
``max_views`` views PER CHANNEL *before* the solve so that the same precision is
reached in a couple of minutes.

Selection is a PURE function over lightweight records (no image loading, no
intrinsics required) so it is trivially testable and does not couple to the Qt
layer nor to the ``CalibrationSample`` dataclass.

Priority (per project request, cheapest signals first):
  1. Fully-DIRECT corners (no interpolation) are preferred over interpolated
     ones — interpolated views are only used to top up the cap if direct views
     are insufficient.
  2. Detected in BOTH channels — a soft quality bonus, never a filter (for
     intrinsic a single-channel view is still valid).
  3. COVERAGE + pose diversity — greedy farthest-point sampling (FPS) over a
     standardized feature vector so the kept views spread across image position,
     apparent board size (distance proxy) and foreshortening (tilt proxy).
  4. SHARPNESS — OPTIONAL tiebreak only, default OFF.  Measuring sharpness needs
     loading images (~30-60s for 500) which defeats the speed-up; if supplied it
     is applied as a small bonus on the boundary candidates only.

Feature vector per view, all derivable WITHOUT intrinsics and WITHOUT loading
images (corners are already in memory):
  - centroid_x, centroid_y : mean of the 4 board-quad corners (image position)
  - sqrt(quad_area)        : apparent board size -> monotone DISTANCE proxy
  - skew_a, skew_b         : opposite-side length differences -> foreshortening
                             (frontal board => equal opposite sides) => TILT proxy

When a prior camera matrix ``K`` exists (re-calibration) the caller may pass
pose features (tilt_x, tilt_y, distance) computed via the report-cache
``_compute_poses`` helper; they are appended to the feature vector.  On a fresh
solve ``K`` does not exist (solvePnP needs it), so the quad-only features above
are the primary, intrinsic-free signal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

Number = float
Point = Sequence[Number]


@dataclass(frozen=True)
class ViewCandidate:
    """Lightweight, framework-agnostic record describing one calibration view.

    Decoupled from ``CalibrationSample`` on purpose so the selection logic stays
    pure and testable.  ``key`` is any caller-defined identity (e.g. ``base`` or
    ``(base, channel)``) returned verbatim in the selection.
    """

    key: object
    corners: Sequence[Point]          # normalized [0,1] image corners
    image_size: Tuple[int, int] = (1, 1)  # (width, height); only aspect matters
    is_direct: bool = True            # True == no interpolation (preferred)
    in_both: bool = False             # detected in both channels (soft bonus)
    pose: Optional[Tuple[float, float, float]] = None  # (tilt_x, tilt_y, dist), optional


@dataclass
class _Feature:
    candidate: ViewCandidate
    vec: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Quad / feature extraction (intrinsics-free)
# ---------------------------------------------------------------------------

def _board_quad(corners: Sequence[Point], pattern_size: Optional[Tuple[int, int]]) -> Optional[List[Point]]:
    """Return the 4 outer board corners as a quad, or ``None`` if unavailable.

    Mirrors ``calibration_report_cache._extract_chessboard_quads`` but keeps the
    per-view association (it is computed per candidate, not over a flat list).
    """
    n = len(corners)
    if n < 4:
        return None
    cols = rows = None
    if pattern_size and len(pattern_size) == 2:
        cols, rows = int(pattern_size[0]), int(pattern_size[1])
    if cols and rows and n == cols * rows:
        return [corners[0], corners[cols - 1], corners[-1], corners[-cols]]
    # Fallback when the grid shape is unknown: spread 4 points across the order.
    return [corners[0], corners[n // 2 - 1], corners[-1], corners[n // 2]]


def _polygon_area(quad: Sequence[Point]) -> float:
    """Shoelace area of a 4-point polygon (normalized^2 units)."""
    area = 0.0
    for i in range(len(quad)):
        x1, y1 = float(quad[i][0]), float(quad[i][1])
        x2, y2 = float(quad[(i + 1) % len(quad)][0]), float(quad[(i + 1) % len(quad)][1])
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _side_len(p: Point, q: Point) -> float:
    return math.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))


def _quad_features(
    candidate: ViewCandidate, pattern_size: Optional[Tuple[int, int]]
) -> Optional[List[float]]:
    """Build the intrinsics-free feature vector for one candidate.

    Aspect-corrects x by the image aspect ratio so position distances in x and y
    are comparable in pixels rather than in normalized units.
    """
    quad = _board_quad(candidate.corners, pattern_size)
    if quad is None:
        return None
    w, h = candidate.image_size
    aspect = (float(w) / float(h)) if h else 1.0

    cx = sum(float(p[0]) for p in quad) / 4.0
    cy = sum(float(p[1]) for p in quad) / 4.0
    area = _polygon_area(quad)
    size = math.sqrt(area)

    # Quad vertices are ordered [tl, tr, br, bl] for a regular grid, so opposite
    # sides are (0-1, 3-2) and (1-2, 0-3). Difference of opposite-side lengths is
    # a foreshortening / tilt proxy (frontal board => differences ~0).
    s01 = _side_len(quad[0], quad[1])
    s32 = _side_len(quad[3], quad[2])
    s12 = _side_len(quad[1], quad[2])
    s03 = _side_len(quad[0], quad[3])
    skew_a = abs(s01 - s32)
    skew_b = abs(s12 - s03)

    vec = [cx * aspect, cy, size, skew_a, skew_b]
    if candidate.pose is not None:
        # Re-calibration path: append true pose (tilt deg, tilt deg, distance).
        vec.extend(float(v) for v in candidate.pose)
    return vec


def _standardize(features: List[_Feature]) -> None:
    """Z-score each feature dimension in place so all dims are comparable."""
    if not features:
        return
    dims = len(features[0].vec)
    for d in range(dims):
        col = [f.vec[d] for f in features]
        mean = sum(col) / len(col)
        var = sum((x - mean) ** 2 for x in col) / len(col)
        std = math.sqrt(var)
        if std < 1e-9:
            # Degenerate dimension carries no information; zero it out.
            for f in features:
                f.vec[d] = 0.0
        else:
            for f in features:
                f.vec[d] = (f.vec[d] - mean) / std


def _sq_dist(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Greedy farthest-point sampling with priority bonuses
# ---------------------------------------------------------------------------

def _fps_select(
    features: List[_Feature],
    count: int,
    *,
    both_bonus: float,
    sharpness: Dict[object, float],
    sharpness_bonus: float,
) -> List[_Feature]:
    """Greedy farthest-point sampling of ``count`` features.

    Maximises spread in the standardized feature space.  The ``in_both`` and
    optional sharpness signals are added as small additive bonuses to the
    farthest-point score so they act as tie-breakers among comparably-diverse
    candidates without overriding coverage.  Deterministic: the first seed is the
    candidate farthest from the centroid (no randomness).
    """
    if count >= len(features):
        return list(features)
    if count <= 0:
        return []

    dims = len(features[0].vec) if features[0].vec else 0
    centroid = [sum(f.vec[d] for f in features) / len(features) for d in range(dims)]

    def static_bonus(f: _Feature) -> float:
        bonus = both_bonus if f.candidate.in_both else 0.0
        if sharpness:
            bonus += sharpness_bonus * sharpness.get(f.candidate.key, 0.0)
        return bonus

    # Deterministic seed: farthest from centroid, bonuses break exact ties.
    seed_idx = max(
        range(len(features)),
        key=lambda i: (_sq_dist(features[i].vec, centroid) + static_bonus(features[i])),
    )
    selected = [seed_idx]
    selected_set = {seed_idx}
    # min squared distance from each candidate to the current selected set
    min_d = [_sq_dist(f.vec, features[seed_idx].vec) for f in features]

    while len(selected) < count:
        best_i = -1
        best_score = -math.inf
        for i, f in enumerate(features):
            if i in selected_set:
                continue
            score = min_d[i] + static_bonus(f)
            if score > best_score:
                best_score = score
                best_i = i
        if best_i < 0:
            break
        selected.append(best_i)
        selected_set.add(best_i)
        chosen_vec = features[best_i].vec
        for i, f in enumerate(features):
            if i in selected_set:
                continue
            d = _sq_dist(f.vec, chosen_vec)
            if d < min_d[i]:
                min_d[i] = d

    return [features[i] for i in selected]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_calibration_views(
    candidates: Sequence[ViewCandidate],
    max_views: int,
    *,
    pattern_size: Optional[Tuple[int, int]] = None,
    both_bonus: float = 0.25,
    sharpness_fn: Optional[Callable[[ViewCandidate], float]] = None,
    sharpness_bonus: float = 0.15,
) -> List[ViewCandidate]:
    """Pick the best ``max_views`` views from ``candidates`` (single channel).

    Returns the selected candidates in the SAME representation received (a subset
    of the input list), preserving each candidate's ``key``.  Pure and
    deterministic — same input always yields the same output.

    Algorithm:
      1. If ``len(candidates) <= max_views`` return them all unchanged.
      2. Partition into fully-direct vs interpolated.  Fill the cap from direct
         views first via FPS; only if direct views are fewer than the cap do we
         top up with FPS over the interpolated views.
      3. FPS spreads the picks over a standardized (position, size, skew[, pose])
         feature vector for maximum coverage / pose diversity.  ``in_both`` adds a
         small bonus; ``sharpness_fn`` (default ``None``) is an OPTIONAL tiebreak
         only applied to the boundary candidates, never forcing image loads here.

    Args:
        candidates: views for a SINGLE channel (intrinsic is per-channel).
        max_views: hard cap on returned views.
        pattern_size: ``(cols, rows)`` board shape; enables exact quad extraction.
        both_bonus: additive FPS bonus for ``in_both`` views (soft preference).
        sharpness_fn: optional ``candidate -> sharpness`` score; if given it is
            evaluated ONLY on the boundary candidates (those competing for the
            last slots) to avoid loading every image. Default ``None`` (off).
        sharpness_bonus: weight of the sharpness tiebreak.
    """
    candidates = list(candidates)
    if max_views <= 0 or len(candidates) <= max_views:
        return candidates

    # --- Build features, dropping any candidate whose quad cannot be formed. ---
    feats: List[_Feature] = []
    skipped: List[ViewCandidate] = []
    for c in candidates:
        vec = _quad_features(c, pattern_size)
        if vec is None:
            skipped.append(c)
        else:
            feats.append(_Feature(candidate=c, vec=vec))

    # Standardize across the whole candidate set so dims are comparable.
    _standardize(feats)

    direct = [f for f in feats if f.candidate.is_direct]
    interp = [f for f in feats if not f.candidate.is_direct]

    # Optional sharpness only on the boundary candidates (near the cap) to avoid
    # loading every image. We approximate the boundary as the pool we sample from.
    def boundary_sharpness(pool: List[_Feature], take: int) -> Dict[object, float]:
        if sharpness_fn is None or take >= len(pool):
            return {}
        scores: Dict[object, float] = {}
        for f in pool:
            try:
                scores[f.candidate.key] = float(sharpness_fn(f.candidate))
            except Exception:  # noqa: BLE001 - tiebreak must never break the solve
                scores[f.candidate.key] = 0.0
        # Normalize to [0,1] so the bonus magnitude is bounded.
        if scores:
            vals = list(scores.values())
            lo, hi = min(vals), max(vals)
            if hi - lo > 1e-9:
                scores = {k: (v - lo) / (hi - lo) for k, v in scores.items()}
            else:
                scores = {k: 0.0 for k in scores}
        return scores

    selected: List[_Feature] = []
    remaining = max_views

    # 1) Fill from fully-direct views first.
    if direct:
        take = min(remaining, len(direct))
        sharp = boundary_sharpness(direct, take)
        selected.extend(
            _fps_select(
                direct, take,
                both_bonus=both_bonus,
                sharpness=sharp,
                sharpness_bonus=sharpness_bonus,
            )
        )
        remaining = max_views - len(selected)

    # 2) Top up with interpolated views only if direct could not fill the cap.
    if remaining > 0 and interp:
        take = min(remaining, len(interp))
        sharp = boundary_sharpness(interp, take)
        selected.extend(
            _fps_select(
                interp, take,
                both_bonus=both_bonus,
                sharpness=sharp,
                sharpness_bonus=sharpness_bonus,
            )
        )

    result = [f.candidate for f in selected]

    # 3) If quad extraction skipped some and we still have room, include them
    #    (they are valid views; better a kept view than an empty slot).
    if len(result) < max_views and skipped:
        result.extend(skipped[: max_views - len(result)])

    return result
