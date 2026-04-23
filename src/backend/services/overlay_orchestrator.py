"""Overlay orchestrator for managing image overlays with caching and prefetching.

Coordinates rendering of calibration and label overlays on top of base images,
manages cache invalidation, and coordinates prefetching for smooth navigation.

Extracts overlay data from session/state and delegates rendering to OverlayWorkflow.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

from PyQt6.QtCore import QObject
from PyQt6.QtGui import QColor, QPixmap

from common.dict_helpers import get_dict_path
from common.reasons import format_reason_label
from common.log_utils import log_debug
from backend.utils.calibration import undistort_points

if TYPE_CHECKING:
    from backend.services.dataset_session import DatasetSession
    from backend.services.viewer_state import ViewerState
    from backend.services.overlays.overlay_prefetcher import OverlayPrefetcher
    from backend.services.overlays.overlay_workflow import OverlayWorkflow


class OverlayOrchestrator(QObject):
    """Orchestrates overlay rendering, caching, and prefetching.

    Extracts all overlay-related data from session/state and passes it to
    OverlayWorkflow.render(). This centralizes the data extraction logic
    that was previously scattered in ImageViewer.
    """

    def __init__(
        self,
        overlay_workflow: "OverlayWorkflow",
        overlay_prefetcher: "OverlayPrefetcher",
        session: "DatasetSession",
        state: "ViewerState",
        *,
        get_label_boxes: Callable[[str, str], List[Tuple[str, float, float, float, float, QColor, bool]]],
        get_label_signature: Callable[[str, str, List], Optional[Tuple[Any, ...]]],
        get_error_thresholds: Callable[[], Dict[str, float]],
        parent: Optional[QObject] = None,
    ) -> None:
        """Initialize overlay orchestrator.

        Args:
            overlay_workflow: Workflow for rendering and caching overlays
            overlay_prefetcher: Prefetcher for preloading nearby overlays
            session: Dataset session providing loader and corners
            state: Viewer state with cache_data
            get_label_boxes: Callback to get label boxes for (base, channel)
            get_label_signature: Callback to get label signature for cache
            get_error_thresholds: Callback to get calibration error thresholds
            parent: Optional parent QObject
        """
        super().__init__(parent)
        self.workflow = overlay_workflow
        self.prefetcher = overlay_prefetcher
        self.session = session
        self.state = state
        self._get_label_boxes = get_label_boxes
        self._get_label_signature = get_label_signature
        self._get_error_thresholds = get_error_thresholds

        # View settings (controlled by ImageViewer)
        self.view_rectified = False
        self.align_mode = "disabled"  # "disabled", "full", "fov_focus"
        self.grid_mode = "thirds"  # "off", "thirds", "detailed"
        self.show_labels = False
        self.show_overlays = True  # Info overlay (status text, calibration markers, corners)
        self.corner_display_mode = "subpixel"  # "original", "subpixel", "both"

        # Cached homography for stereo alignment (computed once per calibration)
        self._cached_homography: Optional[Any] = None
        self._cached_lwir_calib_size: Optional[Tuple[int, int]] = None
        self._cached_vis_calib_size: Optional[Tuple[int, int]] = None
        self._cached_extrinsic_id: Optional[int] = None  # Track if extrinsic changed
        self._cached_parallax_h: float = 0.0  # Track parallax used for cached H
        self._cached_parallax_v: float = 0.0

        # User-adjustable parallax correction (horizontal / vertical pixels)
        self.parallax_h: float = 0.0
        self.parallax_v: float = 0.0

        # Cached aligned pixmaps: {base: (mode, view_rectified, lwir_pixmap, vis_pixmap)}
        # Only cache the last one to avoid memory bloat
        self._aligned_cache: Optional[Tuple[str, str, bool, QPixmap, QPixmap, Any]] = None

    def render_pair(self, base: str) -> Tuple[Optional[QPixmap], Optional[QPixmap]]:
        """Render overlayed pair for given base.

        Extracts all necessary data from session/state and renders both channels.

        Args:
            base: Image base name

        Returns:
            Tuple of (lwir_pixmap, vis_pixmap), either can be None
        """
        # Extract mark/reason data (unified format: base -> {reason, auto})
        cd = self.state.cache_data
        mark_entry = cd["marks"].get(base)
        if isinstance(mark_entry, dict):
            reason = mark_entry.get("reason")
            auto_reason = mark_entry.get("auto", False)
        else:
            reason = mark_entry  # Legacy string format
            auto_reason = False

        reason_label = format_reason_label(reason, auto=auto_reason) if reason else None

        # Calibration data
        calibration_flag = base in self.state.calibration_marked
        calibration_auto = self.state.is_calibration_auto(base)
        calib_results = self.state.calibration_results.get(base, {})
        # Only load corners if we need to show them (calibration marked AND has detection)
        calib_corners: Dict[str, Any] = {}
        if calibration_flag and (calib_results.get("lwir") or calib_results.get("visible")):
            calib_corners = self.session.get_corners(base) or {}

        # Determine which corners to show based on corner_display_mode
        # For each channel, we may have "original", "subpixel", or "both"
        def get_corners_for_channel(channel: str) -> Tuple[
            Optional[List[List[float]]],  # primary corners
            Optional[List[List[float]]],  # secondary corners (only in "both" mode)
            bool,  # primary is subpixel?
        ]:
            original = calib_corners.get(channel)
            subpixel = calib_corners.get(f"{channel}_subpixel")

            if self.corner_display_mode == "original":
                return original, None, False
            elif self.corner_display_mode == "subpixel":
                # Prefer subpixel if available, fallback to original
                if subpixel:
                    return subpixel, None, True
                return original, None, False
            else:  # "both"
                # Primary = subpixel (if available), secondary = original
                if subpixel:
                    return subpixel, original, True
                return original, None, False

        lwir_corners_primary, lwir_corners_secondary, lwir_is_subpixel = get_corners_for_channel("lwir")
        vis_corners_primary, vis_corners_secondary, vis_is_subpixel = get_corners_for_channel("visible")

        # Error data
        thresholds = self._get_error_thresholds()
        calibration_errors: Dict[str, Optional[float]] = {
            "lwir": get_dict_path(cd, f"reproj_errors.lwir.{base}"),
            "visible": get_dict_path(cd, f"reproj_errors.visible.{base}"),
        }
        stereo_error = cd["extrinsic_errors"].get(base)

        # Label data (only if showing labels)
        # Format: (display_name, x_center, y_center, width, height, color, is_projected, is_auto)
        label_boxes_lwir: List[Tuple[str, float, float, float, float, QColor, bool, bool]] = []
        label_boxes_vis: List[Tuple[str, float, float, float, float, QColor, bool, bool]] = []
        label_sig_lwir: Optional[Tuple[Any, ...]] = None
        label_sig_vis: Optional[Tuple[Any, ...]] = None

        if self.show_labels:
            # Get direct labels for each channel (is_projected=False)
            raw_lwir = self._get_label_boxes(base, "lwir")
            raw_vis = self._get_label_boxes(base, "visible")

            # Add is_projected=False flag to direct labels, keep is_auto
            label_boxes_lwir = [(d, x, y, w, h, c, False, auto) for d, x, y, w, h, c, auto in raw_lwir]
            label_boxes_vis = [(d, x, y, w, h, c, False, auto) for d, x, y, w, h, c, auto in raw_vis]

            # Project labels from other channel (is_projected=True)
            # vis -> lwir projection
            projected_to_lwir = self._project_labels(
                raw_vis, "visible", "lwir", base
            )
            # lwir -> vis projection
            projected_to_vis = self._project_labels(
                raw_lwir, "lwir", "visible", base
            )

            # Append projected labels
            label_boxes_lwir.extend(projected_to_lwir)
            label_boxes_vis.extend(projected_to_vis)

            # Update signatures to include projected labels
            label_sig_lwir = self._get_label_signature(base, "lwir", raw_lwir)
            label_sig_vis = self._get_label_signature(base, "visible", raw_vis)
            # Include cross-channel info in signature for cache invalidation
            if label_sig_lwir and raw_vis:
                label_sig_lwir = (*label_sig_lwir, "proj", len(raw_vis))
            if label_sig_vis and raw_lwir:
                label_sig_vis = (*label_sig_vis, "proj", len(raw_lwir))

        # Get base pixmaps from session
        display_lwir, display_vis = self.session.prepare_display_pair(base, self.view_rectified)

        # Get ORIGINAL image sizes from disk for corner coordinate transformation
        # Corners are normalized relative to the original image file, not the display pixmap
        lwir_original_size = self.session.get_original_image_size(base, "lwir")
        vis_original_size = self.session.get_original_image_size(base, "visible")

        # Apply stereo alignment if enabled (any mode except "disabled")
        alignment_transform = None
        if self.align_mode != "disabled" and display_lwir and display_vis:
            # Check alignment cache
            cache = self._aligned_cache
            if (
                cache is not None
                and cache[0] == base
                and cache[1] == self.align_mode
                and cache[2] == self.view_rectified
                and cache[5] == self.parallax_h
                and cache[6] == self.parallax_v
            ):
                # Cache hit - use cached aligned pixmaps
                display_lwir, display_vis = cache[3], cache[4]
                alignment_transform = cache[7]
                log_debug(f"Alignment cache hit for {base}", "ALIGN")
            else:
                # Cache miss - compute alignment
                display_lwir, display_vis = self._apply_stereo_alignment(
                    display_lwir, display_vis, self.align_mode
                )
                # Get the transform that was stored during alignment
                alignment_transform = self.state.cache_data.get("_alignment_transform")
                # Cache the result
                self._aligned_cache = (
                    base, self.align_mode, self.view_rectified,
                    display_lwir, display_vis,
                    self.parallax_h, self.parallax_v,
                    alignment_transform,
                )
                log_debug(f"Alignment cache miss for {base}, cached", "ALIGN")

        # Get calibration matrices for corner transformation
        cd = self.state.cache_data
        matrices = cd.get("_matrices") or {}
        lwir_matrices = matrices.get("lwir") or {}
        vis_matrices = matrices.get("visible") or {}

        # When show_overlays is False, suppress info overlays (but keep grid and labels)
        if not self.show_overlays:
            reason = None
            reason_label = None
            calibration_flag = False
            calibration_auto = False
            calibration_errors = None
            stereo_error = None
            lwir_corners_primary = None
            lwir_corners_secondary = None
            vis_corners_primary = None
            vis_corners_secondary = None

        # Render LWIR with overlays
        # Pass corners based on display mode - transformation happens inside render
        display_lwir = self.workflow.render(
            base,
            "lwir",
            display_lwir,
            view_rectified=self.view_rectified,
            grid_mode=self.grid_mode,
            reason=reason,
            reason_label=reason_label,
            calibration=calibration_flag,
            calibration_auto=calibration_auto,
            calibration_detected=calib_results.get("lwir") if self.show_overlays else None,
            corner_points=lwir_corners_primary,
            corner_points_secondary=lwir_corners_secondary,
            corners_refined=lwir_is_subpixel if self.show_overlays else False,
            warning_text=None,
            calibration_errors=calibration_errors,
            stereo_error=stereo_error,
            thresholds=thresholds,
            label_boxes=label_boxes_lwir,
            label_sig=label_sig_lwir,
            alignment_transform=alignment_transform,
            original_size=lwir_original_size,
            camera_matrix=lwir_matrices.get("camera_matrix"),
            distortion=lwir_matrices.get("distortion"),
        )

        # Render Visible with overlays
        # Pass corners based on display mode - transformation happens inside render
        display_vis = self.workflow.render(
            base,
            "visible",
            display_vis,
            view_rectified=self.view_rectified,
            grid_mode=self.grid_mode,
            reason=reason,
            reason_label=reason_label,
            calibration=calibration_flag,
            calibration_auto=calibration_auto,
            calibration_detected=calib_results.get("visible") if self.show_overlays else None,
            corner_points=vis_corners_primary,
            corner_points_secondary=vis_corners_secondary,
            corners_refined=vis_is_subpixel if self.show_overlays else False,
            warning_text=None,
            calibration_errors=calibration_errors,
            stereo_error=stereo_error,
            thresholds=thresholds,
            label_boxes=label_boxes_vis,
            label_sig=label_sig_vis,
            alignment_transform=alignment_transform,
            original_size=vis_original_size,
            camera_matrix=vis_matrices.get("camera_matrix"),
            distortion=vis_matrices.get("distortion"),
        )

        return display_lwir, display_vis

    def ensure_cached(self, base: str) -> None:
        """Ensure overlay for base is in cache by rendering it."""
        self.render_pair(base)

    def prepare_prefetch(
        self,
        current_index: int,
        total_pairs: int,
        current_base: Optional[str],
        calibration_marked: Set[str],
        get_base: Callable[[int], Optional[str]],
    ) -> None:
        """Prepare prefetching for nearby images."""
        self.prefetcher.prepare(
            current_index=current_index,
            total_pairs=total_pairs,
            current_base=current_base,
            calibration_marked=calibration_marked,
            get_base=get_base,
        )

    def invalidate(self, base: Optional[str] = None) -> None:
        """Invalidate overlay cache."""
        self.workflow.invalidate(base)

    def invalidate_alignment_cache(self) -> None:
        """Invalidate cached homography and aligned pixmaps (call when calibration data changes)."""
        self._cached_homography = None
        self._cached_extrinsic_id = None
        self._cached_lwir_calib_size = None
        self._cached_vis_calib_size = None
        self._aligned_cache = None

    def _project_labels(
        self,
        labels: List[Tuple[str, float, float, float, float, QColor, bool]],
        source_channel: str,
        target_channel: str,
        base: str,
    ) -> List[Tuple[str, float, float, float, float, QColor, bool, bool]]:
        """Project labels from one channel to the other using calibration.

        Uses the homography computed for stereo alignment to project bbox coordinates.
        All returned labels have is_projected=True.

        Args:
            labels: List of (display_name, x_center, y_center, width, height, color, is_auto)
            source_channel: 'visible' or 'lwir' - where labels were annotated
            target_channel: 'visible' or 'lwir' - where to project
            base: Image base name

        Returns:
            List of projected labels with is_projected=True flag
        """
        if not labels:
            return []

        # Get calibration data for projection
        cd = self.state.cache_data
        matrices = cd.get("_matrices") or {}

        # Need both intrinsics and extrinsic
        # Note: _extrinsic is the key for extrinsic calibration data
        lwir_mat = matrices.get("lwir") or {}
        vis_mat = matrices.get("visible") or {}
        extrinsic = cd.get("_extrinsic") or {}

        if not lwir_mat or not vis_mat or not extrinsic:
            log_debug(
                f"No calibration data for label projection "
                f"(lwir={bool(lwir_mat)}, vis={bool(vis_mat)}, ext={bool(extrinsic)})",
                "LABELS"
            )
            return []

        # Get or compute homography for this projection direction
        homography = self._get_projection_homography(
            source_channel, target_channel, lwir_mat, vis_mat, extrinsic, base
        )

        if homography is None:
            return []

        # Get image sizes for normalization
        source_size = self.session.get_original_image_size(base, source_channel)
        target_size = self.session.get_original_image_size(base, target_channel)

        if not source_size or not target_size:
            log_debug(f"No image sizes for label projection", "LABELS")
            return []

        # Import here to avoid circular imports
        from backend.services.labels.bbox_transform import project_bbox_with_homography

        projected: List[Tuple[str, float, float, float, float, QColor, bool, bool]] = []

        for display_name, xc, yc, w, h, color, is_auto in labels:
            bbox = (xc, yc, w, h)
            proj_bbox = project_bbox_with_homography(
                bbox, homography, source_size, target_size
            )
            if proj_bbox:
                proj_xc, proj_yc, proj_w, proj_h = proj_bbox
                # Use same color but with is_projected=True
                projected.append((display_name, proj_xc, proj_yc, proj_w, proj_h, color, True, is_auto))

        if projected:
            log_debug(
                f"Projected {len(projected)}/{len(labels)} labels from {source_channel} to {target_channel}",
                "LABELS"
            )

        return projected

    def _get_projection_homography(
        self,
        source_channel: str,
        target_channel: str,
        lwir_mat: Dict[str, Any],
        vis_mat: Dict[str, Any],
        extrinsic: Dict[str, Any],
        base: str,
    ) -> Optional[Any]:
        """Get or compute homography for label projection between channels.

        Reuses compute_alignment_homography from stereo_alignment.py.
        """
        try:
            import numpy as np
            from backend.utils.stereo_alignment import compute_alignment_homography
        except ImportError:
            return None

        # Determine source and target matrices
        if source_channel == "visible":
            source_mat = vis_mat
            target_mat = lwir_mat
            source_is_lwir = False
        else:
            source_mat = lwir_mat
            target_mat = vis_mat
            source_is_lwir = True

        # Get image size for homography computation
        source_size = self.session.get_original_image_size(base, source_channel)
        if not source_size:
            return None

        try:
            homography = compute_alignment_homography(
                source_matrix=source_mat,
                target_matrix=target_mat,
                rotation=extrinsic.get("rotation") or extrinsic.get("R"),
                translation=extrinsic.get("translation") or extrinsic.get("T"),
                image_size=source_size,
                source_is_lwir=source_is_lwir,
                parallax_h=self.parallax_h,
                parallax_v=self.parallax_v,
            )
            return homography
        except Exception as e:
            log_debug(f"Failed to compute projection homography: {e}", "LABELS")
            return None

    def clear(self) -> None:
        """Clear all cached overlays and cancel prefetching."""
        self.prefetcher.clear()
        self.invalidate()
        self.invalidate_alignment_cache()

    def set_view_rectified(self, rectified: bool) -> None:
        """Set whether to show rectified images."""
        if rectified != self.view_rectified:
            self.view_rectified = rectified
            self._aligned_cache = None  # Invalidate aligned cache on view change
            self.invalidate()

    def set_align_mode(self, mode: str) -> None:
        """Set stereo alignment mode.

        Args:
            mode: One of:
                - 'disabled': No alignment, show original images
                - 'full': Full view with FOV highlighted
                - 'fov_focus': Crop to FOV region only
                - 'max_overlap': Show maximum overlap area (no green padding)
        """
        if mode not in ("disabled", "full", "fov_focus", "max_overlap"):
            mode = "disabled"
        if mode != self.align_mode:
            self.align_mode = mode
            self.invalidate()

    def set_parallax_correction(self, h: float, v: float) -> None:
        """Set the parallax correction in pixels (horizontal / vertical).

        Changing either component invalidates the cached homography so
        the next render recomputes it.
        """
        if h != self.parallax_h or v != self.parallax_v:
            self.parallax_h = h
            self.parallax_v = v
            # Force homography recomputation (parallax changes the matrix)
            self._cached_homography = None
            self._aligned_cache = None
            self.invalidate()

    def set_grid_mode(self, mode: str) -> None:
        """Set grid display mode: 'off', 'thirds', or 'detailed'."""
        if mode not in ("off", "thirds", "detailed"):
            mode = "off"
        if mode != self.grid_mode:
            self.grid_mode = mode
            self.invalidate()

    def set_show_labels(self, show: bool) -> None:
        """Set whether to show label box overlays."""
        if show != self.show_labels:
            self.show_labels = show
            self.invalidate()

    def set_show_overlays(self, show: bool) -> None:
        """Set whether to show info overlays (status text, calibration markers, corners)."""
        if show != self.show_overlays:
            self.show_overlays = show
            self.invalidate()

    def set_corner_display_mode(self, mode: str) -> None:
        """Set corner display mode.

        Args:
            mode: One of:
                - 'original': Show only original detected corners
                - 'subpixel': Show subpixel-refined corners (if available)
                - 'both': Show both original and subpixel for comparison
        """
        if mode not in ("original", "subpixel", "both"):
            mode = "subpixel"
        if mode != self.corner_display_mode:
            self.corner_display_mode = mode
            self.invalidate()

    def _transform_corners_for_rectified(
        self,
        corners: Optional[List[Tuple[float, float]]],
        channel: str,
        original_size: Optional[Tuple[int, int]],
    ) -> Optional[List[Tuple[float, float]]]:
        """Transform corner points if view_rectified is enabled.

        Corners are detected on the RAW (distorted) image but displayed on
        the undistorted image. This function transforms them using undistortPoints.

        Args:
            corners: Normalized corner coordinates (0-1 relative to original image)
            channel: "lwir" or "visible"
            original_size: (width, height) of original image

        Returns:
            Transformed corners (still normalized) or original if no transform needed
        """
        if not corners or not self.view_rectified or not original_size:
            return corners

        # Get calibration matrices for this channel
        cd = self.state.cache_data
        matrices = cd.get("_matrices", {}).get(channel)
        if not matrices:
            log_debug(f"_transform_corners_for_rectified [{channel}]: no matrices, skipping", "OVERLAY")
            return corners

        camera_matrix = matrices.get("camera_matrix")
        distortion = matrices.get("distortion")
        if camera_matrix is None or distortion is None:
            log_debug(f"_transform_corners_for_rectified [{channel}]: no camera/distortion, skipping", "OVERLAY")
            return corners

        orig_w, orig_h = original_size

        # Denormalize corners to pixel coordinates
        pixel_corners = [(u * orig_w, v * orig_h) for u, v in corners]

        # Transform through undistort
        transformed = undistort_points(
            pixel_corners, camera_matrix, distortion, original_size
        )

        # Re-normalize to 0-1 range
        # Note: after undistort, coordinates might be slightly outside 0-1
        result = [(x / orig_w, y / orig_h) for x, y in transformed]

        # Log sample transformation
        if corners:
            u0, v0 = corners[0]
            u1, v1 = result[0]
            log_debug(
                f"_transform_corners_for_rectified [{channel}]: "
                f"({u0:.4f},{v0:.4f}) -> ({u1:.4f},{v1:.4f})",
                "OVERLAY"
            )

        return result

    def _apply_stereo_alignment(
        self,
        lwir_pixmap: QPixmap,
        vis_pixmap: QPixmap,
        mode: str,
    ) -> Tuple[QPixmap, QPixmap]:
        """Apply stereo alignment based on mode.

        Args:
            lwir_pixmap: LWIR camera pixmap
            vis_pixmap: Visible camera pixmap
            mode: Alignment mode:
                - 'full': Show both images with FOV highlighted, visible area darkened outside FOV
                - 'fov_focus': Crop both images to show only the FOV region
                - 'max_overlap': Show maximum overlap area without green padding

        Returns:
            Tuple of (aligned_lwir, aligned_vis)
        """
        from backend.utils.stereo_alignment import (
            compute_alignment_homography,
            align_fov_with_padding,
            align_fov_crop,
            align_max_overlap,
        )

        log_debug(f"_apply_stereo_alignment called with mode={mode}", "ALIGN")

        # Get calibration data
        cd = self.state.cache_data
        lwir_matrices = cd.get("_matrices", {}).get("lwir")
        vis_matrices = cd.get("_matrices", {}).get("visible")
        extrinsic = cd.get("_extrinsic")

        log_debug(f"Matrices available: lwir={lwir_matrices is not None}, vis={vis_matrices is not None}, ext={extrinsic is not None}", "ALIGN")

        if not lwir_matrices or not vis_matrices or not extrinsic:
            log_debug("Missing calibration data for alignment", "ALIGN")
            return (lwir_pixmap, vis_pixmap)

        # Get calibration sizes
        lwir_size_raw = lwir_matrices.get("image_size")
        vis_size_raw = vis_matrices.get("image_size")

        lwir_calib_size = (int(lwir_size_raw[0]), int(lwir_size_raw[1])) if lwir_size_raw else (640, 480)
        vis_calib_size = (int(vis_size_raw[0]), int(vis_size_raw[1])) if vis_size_raw else (1600, 1200)

        # Use cached homography if valid (extrinsic data hasn't changed)
        extrinsic_id = id(extrinsic)
        if (
            self._cached_homography is not None
            and self._cached_extrinsic_id == extrinsic_id
            and self._cached_lwir_calib_size == lwir_calib_size
            and self._cached_vis_calib_size == vis_calib_size
            and self._cached_parallax_h == self.parallax_h
            and self._cached_parallax_v == self.parallax_v
        ):
            homography = self._cached_homography
            log_debug("Using cached homography", "ALIGN")
        else:
            # Compute and cache homography LWIR → Visible
            homography = compute_alignment_homography(
                lwir_matrices, vis_matrices,
                extrinsic.get("rotation") or extrinsic.get("R"),
                extrinsic.get("translation") or extrinsic.get("T"),
                lwir_calib_size,
                source_is_lwir=True,
                parallax_h=self.parallax_h,
                parallax_v=self.parallax_v,
            )
            if homography is not None:
                self._cached_homography = homography
                self._cached_extrinsic_id = extrinsic_id
                self._cached_lwir_calib_size = lwir_calib_size
                self._cached_vis_calib_size = vis_calib_size
                self._cached_parallax_h = self.parallax_h
                self._cached_parallax_v = self.parallax_v
                log_debug("Computed and cached new homography", "ALIGN")

        if homography is None:
            log_debug("Failed to compute alignment homography", "ALIGN")
            return (lwir_pixmap, vis_pixmap)

        # Apply alignment based on mode
        if mode == "fov_focus":
            # Crop to FOV only
            lwir_aligned, vis_aligned, transform = align_fov_crop(
                lwir_pixmap, vis_pixmap,
                homography,
                lwir_calib_size,
                vis_calib_size,
            )
        elif mode == "max_overlap":
            # Max overlap - no green padding, show only valid image data
            lwir_aligned, vis_aligned, transform = align_max_overlap(
                lwir_pixmap, vis_pixmap,
                homography,
                lwir_calib_size,
                vis_calib_size,
            )
        else:
            # Full view with FOV highlighted
            lwir_aligned, vis_aligned, transform = align_fov_with_padding(
                lwir_pixmap, vis_pixmap,
                homography,
                lwir_calib_size,
                vis_calib_size,
            )

        # Store transform for use in corner overlays and other GUI elements
        if transform is not None:
            self.state.cache_data["_alignment_transform"] = transform
            log_debug(f"Stored alignment transform: output_size={transform.output_size}", "ALIGN")

        return (lwir_aligned, vis_aligned)
