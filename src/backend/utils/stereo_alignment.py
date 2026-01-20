"""Stereo image alignment utilities using extrinsic calibration.

This module provides functions for aligning stereo image pairs using
the extrinsic calibration (R, T) between cameras, including:
- FOV calculation and comparison
- Homography computation for alignment
- Image transformation with overlap visualization
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore

from PyQt6.QtGui import QPixmap, QImage, QColor, QPainter

from common.log_utils import log_debug, log_warning


def calculate_fov(
    camera_matrix: Dict[str, Any],
    image_size: Optional[Tuple[int, int]] = None,
) -> Tuple[float, float]:
    """Calculate horizontal and vertical field of view in degrees.

    Args:
        camera_matrix: Camera intrinsic calibration dict containing:
            - 'camera_matrix': 3x3 matrix as list of lists
            - 'image_size': [width, height] (preferred, from calibration)
        image_size: Fallback image dimensions if not in camera_matrix dict

    Returns:
        Tuple of (horizontal_fov, vertical_fov) in degrees
    """
    if np is None:
        return (0.0, 0.0)

    # Extract focal lengths from camera matrix
    matrix_data = camera_matrix.get("data") or camera_matrix.get("camera_matrix")
    if isinstance(matrix_data, dict):
        matrix_data = matrix_data.get("data")

    if not matrix_data:
        log_warning("Cannot calculate FOV: missing camera matrix data", "ALIGN")
        return (0.0, 0.0)

    # Use image_size from calibration dict if available (more accurate)
    calib_image_size = camera_matrix.get("image_size")
    if isinstance(calib_image_size, list) and len(calib_image_size) >= 2:
        width, height = int(calib_image_size[0]), int(calib_image_size[1])
    elif image_size:
        width, height = image_size
    else:
        log_warning("Cannot calculate FOV: no image size available", "ALIGN")
        return (0.0, 0.0)

    # Camera matrix is 3x3: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    # or flattened: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    if isinstance(matrix_data, list):
        if len(matrix_data) == 3 and isinstance(matrix_data[0], list):
            # 2D list: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            fx = float(matrix_data[0][0])
            fy = float(matrix_data[1][1])
        elif len(matrix_data) >= 5:
            # Flattened list
            fx = float(matrix_data[0])
            fy = float(matrix_data[4])
        else:
            return (0.0, 0.0)
    elif hasattr(matrix_data, 'shape'):  # numpy array
        fx = float(matrix_data[0, 0]) if matrix_data.ndim == 2 else float(matrix_data[0])
        fy = float(matrix_data[1, 1]) if matrix_data.ndim == 2 else float(matrix_data[4])
    else:
        return (0.0, 0.0)

    # FOV = 2 * arctan(size / (2 * focal))
    fov_h = 2.0 * math.degrees(math.atan(width / (2.0 * fx))) if fx > 0 else 0.0
    fov_v = 2.0 * math.degrees(math.atan(height / (2.0 * fy))) if fy > 0 else 0.0

    log_debug(f"FOV calculation: fx={fx:.1f}, fy={fy:.1f}, size={width}x{height} -> H={fov_h:.1f}°, V={fov_v:.1f}°", "ALIGN")
    return (fov_h, fov_v)


def compute_alignment_homography(
    source_matrix: Dict[str, Any],
    target_matrix: Dict[str, Any],
    rotation: Dict[str, Any],
    translation: Dict[str, Any],
    image_size: Tuple[int, int],
    source_is_lwir: bool = True,
) -> Optional[Any]:
    """Compute homography to align source image to target image coordinate system.

    Uses stereo rectification to compute the transformation between cameras.
    The homography maps points from SOURCE coordinates to TARGET coordinates.

    Args:
        source_matrix: Source camera intrinsic matrix (larger FOV)
        target_matrix: Target camera intrinsic matrix (smaller FOV)
        rotation: Rotation matrix (3x3) from cam1 to cam2
        translation: Translation vector (3x1) from cam1 to cam2
        image_size: Image dimensions as (width, height)
        source_is_lwir: True if source is LWIR (helps determine R,T direction)

    Returns:
        3x3 homography matrix (numpy array) or None if computation fails
    """
    if cv2 is None or np is None:
        return None

    try:
        # Extract camera matrices
        K1 = _dict_to_matrix(source_matrix, (3, 3))
        K2 = _dict_to_matrix(target_matrix, (3, 3))
        R = _dict_to_matrix(rotation, (3, 3))
        T = _dict_to_matrix(translation, (3, 1))

        # Get distortion coefficients (may be None)
        D1 = _dict_to_matrix({"data": source_matrix.get("distortion")}, (5,))
        D2 = _dict_to_matrix({"data": target_matrix.get("distortion")}, (5,))
        if D1 is None:
            D1 = np.zeros(5)
        if D2 is None:
            D2 = np.zeros(5)

        if K1 is None or K2 is None or R is None or T is None:
            log_warning("Cannot compute homography: missing matrix data", "ALIGN")
            return None

        # Get image sizes from calibration
        size1_raw = source_matrix.get("image_size")
        size2_raw = target_matrix.get("image_size")

        if isinstance(size1_raw, list) and len(size1_raw) >= 2:
            size1 = (int(size1_raw[0]), int(size1_raw[1]))
        else:
            size1 = image_size

        if isinstance(size2_raw, list) and len(size2_raw) >= 2:
            size2 = (int(size2_raw[0]), int(size2_raw[1]))
        else:
            size2 = image_size

        log_debug(f"Alignment: K1 size={size1}, K2 size={size2}", "ALIGN")
        log_debug(f"Alignment: source_is_lwir={source_is_lwir}", "ALIGN")

        # The extrinsic calibration (R, T) transforms from camera1 to camera2.
        # We need to know which camera is which.
        # Typically LWIR is camera1 and visible is camera2 in our calibration.
        # If source is NOT LWIR, we need to invert R and T.
        if not source_is_lwir:
            # Invert: R' = R^T, T' = -R^T @ T
            R = R.T
            T = -R @ T

        # Use stereoRectify to get rectification transforms
        # This gives us homographies that map to a common rectified plane
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, D1, K2, D2, size1, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,  # 0 = crop to valid pixels
        )

        # The rectification homography for each camera:
        # H1 = K1_new @ R1 @ K1_inv  (maps source to rectified)
        # H2 = K2_new @ R2 @ K2_inv  (maps target to rectified)
        #
        # To map from source to target: H = H2_inv @ H1
        # But we want inv(H) to map target corners to source

        K1_inv = np.linalg.inv(K1)
        K2_inv = np.linalg.inv(K2)

        # New projection matrices (first 3x3 part)
        K1_new = P1[:, :3]
        K2_new = P2[:, :3]

        H1 = K1_new @ R1 @ K1_inv  # source -> rectified
        H2 = K2_new @ R2 @ K2_inv  # target -> rectified

        # Combined: source -> target = H2_inv @ H1
        H2_inv = np.linalg.inv(H2)
        H = H2_inv @ H1

        # Log the transformation details
        fx1, fy1 = K1[0, 0], K1[1, 1]
        fx2, fy2 = K2[0, 0], K2[1, 1]
        log_debug(f"Alignment: source_focal=({fx1:.0f},{fy1:.0f}), target_focal=({fx2:.0f},{fy2:.0f})", "ALIGN")
        log_debug(f"Alignment: ROI1={roi1}, ROI2={roi2}", "ALIGN")

        return H

    except Exception as e:
        log_warning(f"Failed to compute homography: {e}", "ALIGN")
        import traceback
        log_debug(f"Homography error traceback: {traceback.format_exc()}", "ALIGN")
        return None


def _dict_to_matrix(data: Dict[str, Any], shape: Tuple[int, ...]) -> Optional[Any]:
    """Convert a dict with 'data', 'camera_matrix', or raw data to numpy matrix.

    Supports multiple formats:
    - {"data": [[...], ...]} - wrapped matrix
    - {"camera_matrix": [[...], ...]} - calibration dict
    - [[...], ...] - raw list of lists
    - [a, b, c, ...] - flattened list
    """
    if np is None:
        return None

    # Extract raw data from various dict structures
    raw = None
    if isinstance(data, dict):
        raw = data.get("data") or data.get("camera_matrix")
        # If camera_matrix is itself a dict, unwrap it
        if isinstance(raw, dict):
            raw = raw.get("data")
    else:
        raw = data

    if raw is None:
        return None

    try:
        if isinstance(raw, list):
            arr = np.array(raw, dtype=np.float64)
        elif hasattr(raw, 'shape'):
            arr = np.array(raw, dtype=np.float64)
        else:
            return None

        return arr.reshape(shape)
    except Exception:
        return None


def compute_overlap_region(
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
    homography: Any,
) -> Optional[Tuple[int, int, int, int]]:
    """Compute the overlapping region after alignment.

    Args:
        source_size: Source image size (width, height)
        target_size: Target image size (width, height)
        homography: 3x3 homography matrix

    Returns:
        Bounding box of overlap (x, y, width, height) or None
    """
    if cv2 is None or np is None or homography is None:
        return None

    try:
        # Transform source corners through homography
        sw, sh = source_size
        tw, th = target_size

        source_corners = np.array([
            [0, 0],
            [sw, 0],
            [sw, sh],
            [0, sh],
        ], dtype=np.float32).reshape(-1, 1, 2)

        transformed = cv2.perspectiveTransform(source_corners, homography.astype(np.float32))
        transformed = transformed.reshape(-1, 2)

        # Find bounding box of transformed source
        min_x = max(0, int(np.floor(transformed[:, 0].min())))
        max_x = min(tw, int(np.ceil(transformed[:, 0].max())))
        min_y = max(0, int(np.floor(transformed[:, 1].min())))
        max_y = min(th, int(np.ceil(transformed[:, 1].max())))

        if max_x <= min_x or max_y <= min_y:
            return None

        return (min_x, min_y, max_x - min_x, max_y - min_y)

    except Exception as e:
        log_warning(f"Failed to compute overlap: {e}", "ALIGN")
        return None


# Chroma green - soft green for padding (not too bright)
CHROMA_GREEN = (86, 180, 86)  # BGR - soft chroma green


class AlignmentTransform:
    """Encapsulates the transformation parameters for stereo alignment.

    This class stores all the information needed to transform points from
    the original visible image coordinates to the aligned output coordinates,
    and vice versa. It also handles LWIR coordinate transformations.

    Attributes:
        vis_matrix: 2x3 affine transformation matrix for visible image
        vis_matrix_inv: Inverse transformation (output → original visible)
        lwir_offset: (x, y) offset of LWIR image in output
        lwir_scale: Scale factor applied to LWIR image
        output_size: (width, height) of the output images
        fov_corners_out: FOV corners in output coordinates (4x2 array)
        homography: Optional 3x3 homography matrix for LWIR→Visible mapping
        lwir_calib_size: LWIR calibration size for homography scaling
        vis_calib_size: Visible calibration size for homography scaling
        lwir_display_size: LWIR display size for homography scaling
        vis_display_size: Visible display size for homography scaling
    """

    def __init__(
        self,
        vis_matrix: Any,
        lwir_offset: Tuple[int, int],
        lwir_scale: float,
        output_size: Tuple[int, int],
        fov_corners_out: Any,
        lwir_crop_offset: Tuple[int, int] = (0, 0),
        homography: Any = None,
        lwir_calib_size: Optional[Tuple[int, int]] = None,
        vis_calib_size: Optional[Tuple[int, int]] = None,
        lwir_display_size: Optional[Tuple[int, int]] = None,
        vis_display_size: Optional[Tuple[int, int]] = None,
    ):
        self.vis_matrix = vis_matrix
        self.lwir_offset = lwir_offset
        self.lwir_scale = lwir_scale
        self.output_size = output_size
        self.fov_corners_out = fov_corners_out
        self.lwir_crop_offset = lwir_crop_offset
        self.homography = homography
        self.lwir_calib_size = lwir_calib_size
        self.vis_calib_size = vis_calib_size
        self.lwir_display_size = lwir_display_size
        self.vis_display_size = vis_display_size

        # Compute inverse matrix for visible
        if np is not None and vis_matrix is not None:
            # For 2x3 affine, add [0,0,1] row to make 3x3, invert, take top 2 rows
            M_3x3 = np.vstack([vis_matrix, [0, 0, 1]])
            M_inv_3x3 = np.linalg.inv(M_3x3)
            self.vis_matrix_inv = M_inv_3x3[:2, :]
        else:
            self.vis_matrix_inv = None

    def transform_vis_points(self, points: Any) -> Any:
        """Transform points from original visible coords to output coords.

        Args:
            points: Nx2 array of (x, y) points in original visible coordinates

        Returns:
            Nx2 array of transformed points in output coordinates
        """
        if np is None or self.vis_matrix is None:
            return points

        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)

        # Apply affine transform: [x', y'] = M @ [x, y, 1]
        ones = np.ones((pts.shape[0], 1))
        pts_h = np.hstack([pts, ones])
        transformed = (self.vis_matrix @ pts_h.T).T

        # Log first point transformation for debugging
        if len(pts) > 0:
            log_debug(
                f"transform_vis_points: ({pts[0,0]:.1f},{pts[0,1]:.1f}) -> "
                f"({transformed[0,0]:.1f},{transformed[0,1]:.1f}) "
                f"[vis_matrix[0,2]={self.vis_matrix[0,2]:.1f}, vis_matrix[1,2]={self.vis_matrix[1,2]:.1f}]",
                "ALIGN"
            )

        return transformed

    def transform_vis_points_inverse(self, points: Any) -> Any:
        """Transform points from output coords back to original visible coords.

        Args:
            points: Nx2 array of (x, y) points in output coordinates

        Returns:
            Nx2 array of points in original visible coordinates
        """
        if np is None or self.vis_matrix_inv is None:
            return points

        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)

        ones = np.ones((pts.shape[0], 1))
        pts_h = np.hstack([pts, ones])
        transformed = (self.vis_matrix_inv @ pts_h.T).T

        return transformed

    def transform_vis_corners_complete(
        self,
        normalized_corners: Any,
        original_size: Tuple[int, int],
        view_rectified: bool,
        camera_matrix: Optional[Any] = None,
        distortion: Optional[Any] = None,
    ) -> Any:
        """Transform Visible corners from normalized RAW coords to output coords.

        This method handles the complete transformation chain:
        1. Denormalize to pixel coords (relative to original_size)
        2. If view_rectified: apply undistort transformation
        3. Apply vis_matrix affine transform to get output coords

        Note: Unlike LWIR, visible doesn't have crop_offset since no border
        is cropped from the visible image.

        Args:
            normalized_corners: Nx2 array of (u, v) in 0-1 range
            original_size: (width, height) of original RAW image
            view_rectified: Whether undistort was applied to the display image
            camera_matrix: Camera intrinsic matrix (needed if view_rectified)
            distortion: Distortion coefficients (needed if view_rectified)

        Returns:
            Nx2 array of (x, y) in output pixel coordinates
        """
        if np is None or self.vis_matrix is None:
            return normalized_corners

        corners = np.asarray(normalized_corners, dtype=np.float32)
        if corners.ndim == 1:
            corners = corners.reshape(1, 2)

        orig_w, orig_h = original_size

        # Step 1: Denormalize to RAW pixel coordinates
        pixel_coords = corners.copy()
        pixel_coords[:, 0] = corners[:, 0] * orig_w
        pixel_coords[:, 1] = corners[:, 1] * orig_h

        # Step 2: If view_rectified, apply undistort to points
        if view_rectified and camera_matrix is not None and distortion is not None:
            try:
                import cv2
                cam = np.array(camera_matrix, dtype=np.float32)
                dist = np.array(distortion, dtype=np.float32).reshape(-1)

                # Get new camera matrix (same as undistort_pixmap uses)
                new_cam, _ = cv2.getOptimalNewCameraMatrix(
                    cam, dist, (orig_w, orig_h), 1, (orig_w, orig_h)
                )

                # undistortPoints expects (N, 1, 2) shape
                pts = pixel_coords.reshape(-1, 1, 2)
                undistorted = cv2.undistortPoints(pts, cam, dist, P=new_cam)
                pixel_coords = undistorted.reshape(-1, 2)

                log_debug(
                    f"transform_vis_corners_complete: undistort applied, "
                    f"first point ({corners[0,0]*orig_w:.1f},{corners[0,1]*orig_h:.1f}) -> "
                    f"({pixel_coords[0,0]:.1f},{pixel_coords[0,1]:.1f})",
                    "ALIGN"
                )
            except Exception as e:
                log_warning(f"Failed to undistort visible corners: {e}", "ALIGN")

        # Step 3: Apply vis_matrix affine transform
        ones = np.ones((pixel_coords.shape[0], 1))
        pts_h = np.hstack([pixel_coords, ones])
        output_coords = (self.vis_matrix @ pts_h.T).T

        # Log transformation summary
        if len(corners) > 0:
            log_debug(
                f"transform_vis_corners_complete: normalized({corners[0,0]:.4f},{corners[0,1]:.4f}) -> "
                f"output({output_coords[0,0]:.1f},{output_coords[0,1]:.1f}) "
                f"[rectified={view_rectified}]",
                "ALIGN"
            )

        return output_coords

    def transform_lwir_points(self, points: Any) -> Any:
        """Transform points from cropped LWIR coords to output coords.

        NOTE: This expects points in coordinates of the CROPPED LWIR image
        (after _crop_black_borders was applied). For points in original
        image coordinates, use transform_lwir_points_from_original().

        Args:
            points: Nx2 array of (x, y) points in cropped LWIR coordinates

        Returns:
            Nx2 array of transformed points in output coordinates
        """
        if np is None:
            return points

        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)

        # For cropped coords: just scale and translate (crop offset already applied)
        transformed = pts.copy()
        transformed[:, 0] = pts[:, 0] * self.lwir_scale + self.lwir_offset[0]
        transformed[:, 1] = pts[:, 1] * self.lwir_scale + self.lwir_offset[1]

        return transformed

    def transform_lwir_points_from_original(self, points: Any) -> Any:
        """Transform points from ORIGINAL LWIR coords (pre-crop) to output coords.

        Use this for calibration corners which are stored as normalized
        coordinates relative to the original image file on disk.

        Args:
            points: Nx2 array of (x, y) points in original LWIR coordinates
                   (before any crop/undistort was applied)

        Returns:
            Nx2 array of transformed points in output coordinates
        """
        if np is None:
            return points

        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)

        # For original image coords, we need to:
        # 1. Subtract crop_offset to get coordinates in the cropped image
        # 2. Multiply by lwir_scale (cropped image is scaled to output)
        # 3. Add lwir_offset (position of scaled LWIR in output canvas)
        transformed = pts.copy()
        transformed[:, 0] = (pts[:, 0] - self.lwir_crop_offset[0]) * self.lwir_scale + self.lwir_offset[0]
        transformed[:, 1] = (pts[:, 1] - self.lwir_crop_offset[1]) * self.lwir_scale + self.lwir_offset[1]

        # Log first point transformation for debugging
        if len(pts) > 0:
            log_debug(
                f"transform_lwir_points_from_original: ({pts[0,0]:.1f},{pts[0,1]:.1f}) -> "
                f"({transformed[0,0]:.1f},{transformed[0,1]:.1f}) "
                f"[crop_off={self.lwir_crop_offset}, scale={self.lwir_scale:.3f}, off={self.lwir_offset}]",
                "ALIGN"
            )

        return transformed

    def transform_lwir_points_inverse(self, points: Any) -> Any:
        """Transform points from output coords back to original LWIR coords.

        Args:
            points: Nx2 array of (x, y) points in output coordinates

        Returns:
            Nx2 array of points in original LWIR coordinates
        """
        if np is None:
            return points

        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)

        # Inverse: subtract offset, divide by scale, add crop offset
        transformed = pts.copy()
        transformed[:, 0] = (pts[:, 0] - self.lwir_offset[0]) / self.lwir_scale + self.lwir_crop_offset[0]
        transformed[:, 1] = (pts[:, 1] - self.lwir_offset[1]) / self.lwir_scale + self.lwir_crop_offset[1]

        return transformed

    def transform_lwir_corners_complete(
        self,
        normalized_corners: Any,
        original_size: Tuple[int, int],
        view_rectified: bool,
        camera_matrix: Optional[Any] = None,
        distortion: Optional[Any] = None,
    ) -> Any:
        """Transform LWIR corners from normalized RAW coords to output coords.

        This method handles the complete transformation chain:
        1. Denormalize to pixel coords (relative to original_size)
        2. If view_rectified: apply undistort transformation
        3. Subtract crop_offset (only meaningful if view_rectified)
        4. Apply scale and offset to get output coords

        Args:
            normalized_corners: Nx2 array of (u, v) in 0-1 range
            original_size: (width, height) of original RAW image
            view_rectified: Whether undistort was applied to the display image
            camera_matrix: Camera intrinsic matrix (needed if view_rectified)
            distortion: Distortion coefficients (needed if view_rectified)

        Returns:
            Nx2 array of (x, y) in output pixel coordinates
        """
        if np is None:
            return normalized_corners

        corners = np.asarray(normalized_corners, dtype=np.float32)
        if corners.ndim == 1:
            corners = corners.reshape(1, 2)

        orig_w, orig_h = original_size

        # Step 1: Denormalize to RAW pixel coordinates
        pixel_coords = corners.copy()
        pixel_coords[:, 0] = corners[:, 0] * orig_w
        pixel_coords[:, 1] = corners[:, 1] * orig_h

        # Step 2: If view_rectified, apply undistort to points
        if view_rectified and camera_matrix is not None and distortion is not None:
            try:
                import cv2
                cam = np.array(camera_matrix, dtype=np.float32)
                dist = np.array(distortion, dtype=np.float32).reshape(-1)

                # Get new camera matrix (same as undistort_pixmap uses)
                new_cam, _ = cv2.getOptimalNewCameraMatrix(
                    cam, dist, (orig_w, orig_h), 1, (orig_w, orig_h)
                )

                # undistortPoints expects (N, 1, 2) shape
                pts = pixel_coords.reshape(-1, 1, 2)
                undistorted = cv2.undistortPoints(pts, cam, dist, P=new_cam)
                pixel_coords = undistorted.reshape(-1, 2)

                log_debug(
                    f"transform_lwir_corners_complete: undistort applied, "
                    f"first point ({corners[0,0]*orig_w:.1f},{corners[0,1]*orig_h:.1f}) -> "
                    f"({pixel_coords[0,0]:.1f},{pixel_coords[0,1]:.1f})",
                    "ALIGN"
                )
            except Exception as e:
                log_warning(f"Failed to undistort corners: {e}", "ALIGN")

        # Step 3: Subtract crop offset (from undistort border removal)
        # This is (0,0) if view_rectified=False or no magenta borders found
        crop_x, crop_y = self.lwir_crop_offset
        cropped_coords = pixel_coords.copy()
        cropped_coords[:, 0] = pixel_coords[:, 0] - crop_x
        cropped_coords[:, 1] = pixel_coords[:, 1] - crop_y

        # Step 4: Apply scale and offset
        output_coords = cropped_coords.copy()
        output_coords[:, 0] = cropped_coords[:, 0] * self.lwir_scale + self.lwir_offset[0]
        output_coords[:, 1] = cropped_coords[:, 1] * self.lwir_scale + self.lwir_offset[1]

        # Log transformation summary
        if len(corners) > 0:
            log_debug(
                f"transform_lwir_corners_complete: normalized({corners[0,0]:.4f},{corners[0,1]:.4f}) -> "
                f"output({output_coords[0,0]:.1f},{output_coords[0,1]:.1f}) "
                f"[rectified={view_rectified}, crop=({crop_x},{crop_y}), scale={self.lwir_scale:.3f}, "
                f"off=({self.lwir_offset[0]},{self.lwir_offset[1]})]",
                "ALIGN"
            )

        return output_coords


def _crop_black_borders(img: Any) -> Tuple[Any, Tuple[int, int, int, int]]:
    """Crop border fill from an image (e.g., from undistortion).

    Detects the magenta fill color (255, 0, 255) used by undistort_pixmap
    to mark areas outside the original image. Does NOT detect black pixels
    as borders since LWIR thermal images may have legitimate dark areas.

    If no magenta pixels are found, returns the image unchanged (no crop).

    Returns:
        Tuple of (cropped_image, (x, y, w, h) of the crop region)
    """
    if img is None:
        return img, (0, 0, 0, 0)

    h, w = img.shape[:2]

    # Create mask of INVALID pixels (magenta border fill only)
    # Do NOT detect black as invalid - LWIR images have legitimate dark areas
    if len(img.shape) == 3:
        # Magenta in BGR: B=255, G=0, R=255
        is_magenta = (
            (img[:, :, 0] > 250) &  # B channel high
            (img[:, :, 1] < 10) &   # G channel low
            (img[:, :, 2] > 250)    # R channel high
        )
        invalid_mask = is_magenta
    else:
        # Grayscale: can't detect magenta, so no crop
        return img, (0, 0, w, h)

    # Check if there are any magenta pixels at all
    magenta_count = np.sum(invalid_mask)
    if magenta_count == 0:
        # No undistort borders detected, return full image
        log_debug("_crop_black_borders: no magenta borders found, using full image", "ALIGN")
        return img, (0, 0, w, h)

    # Valid pixels are NOT invalid
    valid_mask = ~invalid_mask

    # Find the largest inscribed rectangle by shrinking from all sides
    # Start with full image bounds
    top, bottom, left, right = 0, h, 0, w

    # Shrink from top: find first row with >95% valid pixels
    for row in range(h):
        row_valid = np.sum(valid_mask[row, :])
        if row_valid > 0.95 * w:
            top = row
            break

    # Shrink from bottom
    for row in range(h - 1, top, -1):
        row_valid = np.sum(valid_mask[row, :])
        if row_valid > 0.95 * w:
            bottom = row + 1
            break

    # Shrink from left
    for col in range(w):
        col_valid = np.sum(valid_mask[top:bottom, col])
        if col_valid > 0.95 * (bottom - top):
            left = col
            break

    # Shrink from right
    for col in range(w - 1, left, -1):
        col_valid = np.sum(valid_mask[top:bottom, col])
        if col_valid > 0.95 * (bottom - top):
            right = col + 1
            break

    # Validate we found a reasonable crop
    crop_w = right - left
    crop_h = bottom - top

    if crop_w < w * 0.5 or crop_h < h * 0.5:
        # Crop too aggressive, fall back to full image
        log_debug(f"_crop_black_borders: crop too small ({crop_w}x{crop_h}), using full image", "ALIGN")
        return img, (0, 0, w, h)

    # Add tiny margin to avoid edge artifacts
    margin = 1
    top = min(top + margin, bottom - 1)
    bottom = max(bottom - margin, top + 1)
    left = min(left + margin, right - 1)
    right = max(right - margin, left + 1)

    crop_w = right - left
    crop_h = bottom - top

    log_debug(f"_crop_black_borders: cropped to ({left},{top}) {crop_w}x{crop_h}", "ALIGN")

    cropped = img[top:bottom, left:right]
    return cropped, (left, top, crop_w, crop_h)


def _compute_optimal_output_size(
    fov_corners: Any,
    vis_corners: Any,
    vis_matrix: Any,
) -> Tuple[int, int, Any]:
    """Compute the optimal output size based on union of FOV and visible image.

    Calculates the bounding box that contains both the FOV rectangle and
    the rotated visible image, then adjusts the transformation matrix
    to ensure all content fits with minimal padding.

    Args:
        fov_corners: 4x2 array of FOV corners in visible display coords
        vis_corners: 4x2 array of visible image corners (before transform)
        vis_matrix: 2x3 affine matrix (rotation + translation)

    Returns:
        Tuple of (width, height, adjusted_matrix)
    """
    if np is None:
        return (640, 480, vis_matrix)

    # Transform FOV corners with current matrix
    ones = np.ones((4, 1))
    fov_h = np.hstack([fov_corners, ones])
    fov_out = (vis_matrix @ fov_h.T).T

    # Transform visible corners with current matrix
    vis_h = np.hstack([vis_corners, ones])
    vis_out = (vis_matrix @ vis_h.T).T

    # Find bounding box of union
    all_pts = np.vstack([fov_out, vis_out])
    min_x = all_pts[:, 0].min()
    max_x = all_pts[:, 0].max()
    min_y = all_pts[:, 1].min()
    max_y = all_pts[:, 1].max()

    # Add small padding (5 pixels)
    padding = 5
    min_x -= padding
    min_y -= padding
    max_x += padding
    max_y += padding

    # Output size
    out_w = int(max_x - min_x)
    out_h = int(max_y - min_y)

    # Adjust matrix to shift everything so min corner is at (0, 0)
    adjusted = vis_matrix.copy()
    adjusted[0, 2] -= min_x
    adjusted[1, 2] -= min_y

    return (out_w, out_h, adjusted)


def align_fov_with_padding(
    lwir_pixmap: QPixmap,
    vis_pixmap: QPixmap,
    homography: Any,
    lwir_calib_size: Tuple[int, int],
    vis_calib_size: Tuple[int, int],
) -> Tuple[QPixmap, QPixmap, Optional[AlignmentTransform]]:
    """Align visible image to LWIR FOV with rotation, chroma padding, and FOV overlay.

    The visible image is rotated and translated so that the LWIR FOV region
    becomes axis-aligned (horizontal rectangle). This includes:
    - Rotation to align FOV edges horizontally
    - Translation to fit both FOV and visible image optimally
    - Chroma green padding where no image data exists
    - Darkened areas outside the LWIR FOV but inside the visible image
    - Yellow rectangle showing the LWIR FOV boundary

    The LWIR image is cropped to remove black borders (from undistortion),
    then padded with chroma green to match output size.

    Args:
        lwir_pixmap: LWIR camera pixmap
        vis_pixmap: Visible camera pixmap
        homography: Homography matrix (LWIR→Visible mapping)
        lwir_calib_size: LWIR calibration size (width, height)
        vis_calib_size: Visible calibration size (width, height)

    Returns:
        Tuple of (aligned_lwir, aligned_vis, transform) where transform
        contains the coordinate transformation data for use in GUI overlays
    """
    if cv2 is None or np is None or homography is None:
        return (lwir_pixmap, vis_pixmap, None)

    try:
        # Convert to arrays
        lwir_arr = _pixmap_to_array(lwir_pixmap)
        vis_arr = _pixmap_to_array(vis_pixmap)

        if lwir_arr is None or vis_arr is None:
            return (lwir_pixmap, vis_pixmap, None)

        # === LWIR: Crop black borders ===
        lwir_cropped, crop_rect = _crop_black_borders(lwir_arr)
        lwir_h, lwir_w = lwir_cropped.shape[:2]

        log_debug(f"align_fov_with_padding: LWIR cropped from {lwir_arr.shape} to {lwir_cropped.shape}", "ALIGN")

        # Get display sizes
        vis_h, vis_w = vis_arr.shape[:2]

        log_debug(f"align_fov_with_padding: lwir={lwir_w}x{lwir_h}, vis={vis_w}x{vis_h}", "ALIGN")
        log_debug(f"align_fov_with_padding: calib lwir={lwir_calib_size}, vis={vis_calib_size}", "ALIGN")

        # Scale factors from calibration to display
        vis_scale_x = vis_w / vis_calib_size[0]
        vis_scale_y = vis_h / vis_calib_size[1]

        # LWIR corners in calibration coordinates (use cropped region)
        # Adjust for crop offset
        orig_lwir_w, orig_lwir_h = lwir_arr.shape[1], lwir_arr.shape[0]
        lwir_scale_x = orig_lwir_w / lwir_calib_size[0]
        lwir_scale_y = orig_lwir_h / lwir_calib_size[1]

        # Crop rect in calibration coords
        crop_x_calib = crop_rect[0] / lwir_scale_x
        crop_y_calib = crop_rect[1] / lwir_scale_y
        crop_w_calib = crop_rect[2] / lwir_scale_x
        crop_h_calib = crop_rect[3] / lwir_scale_y

        lwir_corners_calib = np.array([
            [crop_x_calib, crop_y_calib],
            [crop_x_calib + crop_w_calib, crop_y_calib],
            [crop_x_calib + crop_w_calib, crop_y_calib + crop_h_calib],
            [crop_x_calib, crop_y_calib + crop_h_calib],
        ], dtype=np.float32).reshape(-1, 1, 2)

        # Map LWIR corners to visible coordinates (in calibration space)
        fov_corners_calib = cv2.perspectiveTransform(
            lwir_corners_calib,
            homography.astype(np.float32)
        ).reshape(-1, 2)

        # Convert FOV corners to visible display coordinates
        fov_corners_display = fov_corners_calib.copy()
        fov_corners_display[:, 0] *= vis_scale_x
        fov_corners_display[:, 1] *= vis_scale_y

        log_debug(f"align_fov_with_padding: FOV corners in vis display: {fov_corners_display.tolist()}", "ALIGN")

        # Calculate rotation angle from top edge of FOV
        # Top edge is from corner 0 to corner 1
        dx = fov_corners_display[1][0] - fov_corners_display[0][0]
        dy = fov_corners_display[1][1] - fov_corners_display[0][1]
        angle = math.degrees(math.atan2(dy, dx))

        log_debug(f"align_fov_with_padding: FOV rotation angle = {angle:.2f}°", "ALIGN")

        # Get FOV center
        fov_center_x = fov_corners_display[:, 0].mean()
        fov_center_y = fov_corners_display[:, 1].mean()

        # Calculate FOV size (width along top edge, height along left edge)
        fov_w = math.sqrt(dx**2 + dy**2)
        dx2 = fov_corners_display[3][0] - fov_corners_display[0][0]
        dy2 = fov_corners_display[3][1] - fov_corners_display[0][1]
        fov_h = math.sqrt(dx2**2 + dy2**2)

        log_debug(f"align_fov_with_padding: FOV size {fov_w:.0f}x{fov_h:.0f} center ({fov_center_x:.0f}, {fov_center_y:.0f})", "ALIGN")

        # === Calculate rotation matrix around FOV center ===
        M_rot = cv2.getRotationMatrix2D((fov_center_x, fov_center_y), angle, 1.0)

        # Visible image corners
        vis_corners = np.array([
            [0, 0], [vis_w, 0], [vis_w, vis_h], [0, vis_h]
        ], dtype=np.float32)

        # Compute optimal output size based on union of FOV and visible image
        out_w, out_h, M_adjusted = _compute_optimal_output_size(
            fov_corners_display, vis_corners, M_rot
        )

        log_debug(f"align_fov_with_padding: optimal output size {out_w}x{out_h}", "ALIGN")

        # Transform visible image with chroma green background
        vis_transformed = cv2.warpAffine(
            vis_arr, M_adjusted, (out_w, out_h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=CHROMA_GREEN
        )

        # Transform FOV corners to output coordinates
        ones = np.ones((4, 1))
        fov_corners_h = np.hstack([fov_corners_display, ones])
        fov_corners_out = (M_adjusted @ fov_corners_h.T).T

        # Calculate the axis-aligned bounding box of the FOV in output coords
        # This is THE reference rectangle for both images
        fov_bbox_x1 = int(fov_corners_out[:, 0].min())
        fov_bbox_y1 = int(fov_corners_out[:, 1].min())
        fov_bbox_x2 = int(fov_corners_out[:, 0].max())
        fov_bbox_y2 = int(fov_corners_out[:, 1].max())
        fov_bbox_w = fov_bbox_x2 - fov_bbox_x1
        fov_bbox_h = fov_bbox_y2 - fov_bbox_y1

        log_debug(f"align_fov_with_padding: FOV bbox in output: ({fov_bbox_x1},{fov_bbox_y1}) {fov_bbox_w}x{fov_bbox_h}", "ALIGN")

        # Transform original visible corners to output coordinates
        vis_corners_h = np.hstack([vis_corners, np.ones((4, 1))])
        vis_corners_out = (M_adjusted @ vis_corners_h.T).T.astype(np.int32)

        # === VISIBLE: Create mask and darken outside FOV bbox ===
        # Use the axis-aligned bbox for consistent masking
        fov_bbox_mask = np.zeros((out_h, out_w), dtype=np.uint8)
        cv2.rectangle(fov_bbox_mask, (fov_bbox_x1, fov_bbox_y1), (fov_bbox_x2, fov_bbox_y2), 255, -1)

        vis_mask = np.zeros((out_h, out_w), dtype=np.uint8)
        cv2.fillPoly(vis_mask, [vis_corners_out], 255)

        # Darken areas outside FOV bbox but inside original image
        outside_fov = (vis_mask == 255) & (fov_bbox_mask == 0)
        vis_result = vis_transformed.copy()
        vis_result[outside_fov] = (vis_transformed[outside_fov] * 0.4).astype(np.uint8)

        # === Calculate Max Overlap region ===
        # Max Overlap = largest axis-aligned rectangle INSIDE both:
        #   1. The FOV bbox (yellow rectangle)
        #   2. The visible polygon (rotated image, no green)
        #
        # The visible polygon is a rotated rectangle. Its borders are lines.
        # We need to find where these borders intersect with the FOV bbox
        # to compute the inscribed rectangle.

        # Helper to find x on a line at a given y
        def line_x_at_y(p1: np.ndarray, p2: np.ndarray, y: float) -> float:
            """Calculate x on line from p1 to p2 at given y."""
            if abs(p2[1] - p1[1]) < 1e-6:  # Horizontal line
                return (p1[0] + p2[0]) / 2
            t = (y - p1[1]) / (p2[1] - p1[1])
            return p1[0] + t * (p2[0] - p1[0])

        # Helper to find y on a line at a given x
        def line_y_at_x(p1: np.ndarray, p2: np.ndarray, x: float) -> float:
            """Calculate y on line from p1 to p2 at given x."""
            if abs(p2[0] - p1[0]) < 1e-6:  # Vertical line
                return (p1[1] + p2[1]) / 2
            t = (x - p1[0]) / (p2[0] - p1[0])
            return p1[1] + t * (p2[1] - p1[1])

        # Visible corners in output coords (already computed as vis_corners_out)
        # Order: [top-left, top-right, bottom-right, bottom-left] in original,
        # but rotated in output space
        v = vis_corners_out.astype(np.float32)

        # The visible polygon has 4 edges:
        # Left edge: v[3] to v[0]
        # Right edge: v[1] to v[2]
        # Top edge: v[0] to v[1]
        # Bottom edge: v[3] to v[2]

        # For the inscribed rectangle within FOV bbox:
        # x_min is limited by: fov_bbox_x1 AND the left edge of visible polygon
        # x_max is limited by: fov_bbox_x2 AND the right edge of visible polygon
        # y_min is limited by: fov_bbox_y1 AND the top edge of visible polygon
        # y_max is limited by: fov_bbox_y2 AND the bottom edge of visible polygon

        # Left edge constraint: find x at fov_bbox_y1 and fov_bbox_y2
        left_x_top = line_x_at_y(v[3], v[0], fov_bbox_y1)
        left_x_bottom = line_x_at_y(v[3], v[0], fov_bbox_y2)
        left_x_constraint = max(left_x_top, left_x_bottom)  # Must be right of both

        # Right edge constraint
        right_x_top = line_x_at_y(v[1], v[2], fov_bbox_y1)
        right_x_bottom = line_x_at_y(v[1], v[2], fov_bbox_y2)
        right_x_constraint = min(right_x_top, right_x_bottom)  # Must be left of both

        # Top edge constraint: find y at fov_bbox_x1 and fov_bbox_x2
        top_y_left = line_y_at_x(v[0], v[1], fov_bbox_x1)
        top_y_right = line_y_at_x(v[0], v[1], fov_bbox_x2)
        top_y_constraint = max(top_y_left, top_y_right)  # Must be below both

        # Bottom edge constraint
        bottom_y_left = line_y_at_x(v[3], v[2], fov_bbox_x1)
        bottom_y_right = line_y_at_x(v[3], v[2], fov_bbox_x2)
        bottom_y_constraint = min(bottom_y_left, bottom_y_right)  # Must be above both

        # Max Overlap bounds: intersection of FOV bbox and visible polygon constraints
        max_overlap_x1 = int(max(fov_bbox_x1, left_x_constraint))
        max_overlap_x2 = int(min(fov_bbox_x2, right_x_constraint))
        max_overlap_y1 = int(max(fov_bbox_y1, top_y_constraint))
        max_overlap_y2 = int(min(fov_bbox_y2, bottom_y_constraint))

        # Clamp to valid values
        max_overlap_x1 = max(0, max_overlap_x1)
        max_overlap_y1 = max(0, max_overlap_y1)
        max_overlap_x2 = min(out_w, max_overlap_x2)
        max_overlap_y2 = min(out_h, max_overlap_y2)

        max_overlap_w = max(0, max_overlap_x2 - max_overlap_x1)
        max_overlap_h = max(0, max_overlap_y2 - max_overlap_y1)

        log_debug(f"align_fov_with_padding: FOV bbox ({fov_bbox_x1},{fov_bbox_y1})-({fov_bbox_x2},{fov_bbox_y2})", "ALIGN")
        log_debug(f"align_fov_with_padding: Vis polygon constraints: left_x={left_x_constraint:.0f}, right_x={right_x_constraint:.0f}", "ALIGN")
        log_debug(f"align_fov_with_padding: Max Overlap ({max_overlap_x1},{max_overlap_y1}) {max_overlap_w}x{max_overlap_h}", "ALIGN")

        # Draw yellow FOV rectangle on visible
        cv2.rectangle(
            vis_result,
            (fov_bbox_x1, fov_bbox_y1),
            (fov_bbox_x2, fov_bbox_y2),
            (0, 255, 255), 2  # Yellow in BGR
        )
        cv2.putText(
            vis_result, "FOV Focus",
            (fov_bbox_x1 + 5, fov_bbox_y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1
        )

        # Draw cyan Max Overlap rectangle on visible
        cv2.rectangle(
            vis_result,
            (max_overlap_x1, max_overlap_y1),
            (max_overlap_x2, max_overlap_y2),
            (255, 255, 0), 2  # Cyan in BGR
        )
        cv2.putText(
            vis_result, "Max Overlap",
            (max_overlap_x1 + 5, max_overlap_y1 + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1
        )

        # === LWIR image: scale FULL image (with black borders) to match FOV size ===
        # Use full LWIR image, not cropped, so we can show the black borders darkened
        orig_lwir_h, orig_lwir_w = lwir_arr.shape[:2]

        # Scale to match the FOV bbox size
        # The valid region (crop_rect) should map to the FOV bbox
        lwir_scale = min(fov_bbox_w / lwir_w, fov_bbox_h / lwir_h)

        # Scale full LWIR image
        new_full_lwir_w = int(orig_lwir_w * lwir_scale)
        new_full_lwir_h = int(orig_lwir_h * lwir_scale)
        lwir_scaled_full = cv2.resize(lwir_arr, (new_full_lwir_w, new_full_lwir_h), interpolation=cv2.INTER_AREA)

        # Calculate where the valid region (crop_rect) should be positioned
        # The valid region center should align with FOV bbox center
        crop_center_x = (crop_rect[0] + crop_rect[2] / 2) * lwir_scale
        crop_center_y = (crop_rect[1] + crop_rect[3] / 2) * lwir_scale
        fov_center_out_x = (fov_bbox_x1 + fov_bbox_x2) / 2
        fov_center_out_y = (fov_bbox_y1 + fov_bbox_y2) / 2

        # Position of full LWIR image in output
        lwir_x = int(fov_center_out_x - crop_center_x)
        lwir_y = int(fov_center_out_y - crop_center_y)

        # Create chroma canvas
        lwir_result = np.full((out_h, out_w, 3), CHROMA_GREEN, dtype=np.uint8)

        # Calculate paste region with bounds checking
        src_x1 = max(0, -lwir_x)
        src_y1 = max(0, -lwir_y)
        dst_x1 = max(0, lwir_x)
        dst_y1 = max(0, lwir_y)

        paste_w = min(new_full_lwir_w - src_x1, out_w - dst_x1)
        paste_h = min(new_full_lwir_h - src_y1, out_h - dst_y1)

        if paste_w > 0 and paste_h > 0:
            lwir_result[dst_y1:dst_y1 + paste_h, dst_x1:dst_x1 + paste_w] = \
                lwir_scaled_full[src_y1:src_y1 + paste_h, src_x1:src_x1 + paste_w]

        # Create mask for the valid (non-black-border) region in output coords
        valid_x1 = lwir_x + int(crop_rect[0] * lwir_scale)
        valid_y1 = lwir_y + int(crop_rect[1] * lwir_scale)
        valid_x2 = valid_x1 + int(crop_rect[2] * lwir_scale)
        valid_y2 = valid_y1 + int(crop_rect[3] * lwir_scale)

        # Create mask for full LWIR area
        lwir_full_mask = np.zeros((out_h, out_w), dtype=np.uint8)
        lwir_full_x2 = min(out_w, lwir_x + new_full_lwir_w)
        lwir_full_y2 = min(out_h, lwir_y + new_full_lwir_h)
        if dst_x1 < lwir_full_x2 and dst_y1 < lwir_full_y2:
            cv2.rectangle(lwir_full_mask, (dst_x1, dst_y1), (lwir_full_x2, lwir_full_y2), 255, -1)

        # Create mask for valid region
        valid_mask = np.zeros((out_h, out_w), dtype=np.uint8)
        cv2.rectangle(valid_mask, (valid_x1, valid_y1), (valid_x2, valid_y2), 255, -1)

        # Darken areas outside valid region but inside full LWIR
        outside_valid = (lwir_full_mask == 255) & (valid_mask == 0)
        lwir_result[outside_valid] = (lwir_result[outside_valid] * 0.4).astype(np.uint8)

        # Draw yellow FOV rectangle on LWIR (same position as visible)
        cv2.rectangle(
            lwir_result,
            (fov_bbox_x1, fov_bbox_y1),
            (fov_bbox_x2, fov_bbox_y2),
            (0, 255, 255), 2  # Yellow in BGR
        )
        cv2.putText(
            lwir_result, "FOV Focus",
            (fov_bbox_x1 + 5, fov_bbox_y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1
        )

        # Draw cyan Max Overlap rectangle on LWIR
        cv2.rectangle(
            lwir_result,
            (max_overlap_x1, max_overlap_y1),
            (max_overlap_x2, max_overlap_y2),
            (255, 255, 0), 2  # Cyan in BGR
        )
        cv2.putText(
            lwir_result, "Max Overlap",
            (max_overlap_x1 + 5, max_overlap_y1 + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1
        )

        # Create transformation object
        # NOTE: In full mode, we display the FULL LWIR image (with magenta borders),
        # NOT the cropped version. Therefore lwir_crop_offset should be (0,0)
        # because the transformation is: output = orig_coord * scale + offset
        # The crop_rect is only used for FOV calculation, not for the displayed image.
        transform = AlignmentTransform(
            vis_matrix=M_adjusted,
            lwir_offset=(lwir_x, lwir_y),
            lwir_scale=lwir_scale,
            output_size=(out_w, out_h),
            fov_corners_out=fov_corners_out,
            lwir_crop_offset=(0, 0),  # Full image shown, no crop offset needed
        )

        # Convert back to pixmaps
        lwir_pm = _array_to_pixmap(lwir_result)
        vis_pm = _array_to_pixmap(vis_result)

        log_debug(f"align_fov_with_padding: result lwir={lwir_pm.width()}x{lwir_pm.height()}, vis={vis_pm.width()}x{vis_pm.height()}", "ALIGN")

        return (lwir_pm or lwir_pixmap, vis_pm or vis_pixmap, transform)

    except Exception as e:
        log_warning(f"align_fov_with_padding failed: {e}", "ALIGN")
        import traceback
        log_debug(f"Traceback: {traceback.format_exc()}", "ALIGN")
        return (lwir_pixmap, vis_pixmap, None)


def align_fov_crop(
    lwir_pixmap: QPixmap,
    vis_pixmap: QPixmap,
    homography: Any,
    lwir_calib_size: Tuple[int, int],
    vis_calib_size: Tuple[int, int],
) -> Tuple[QPixmap, QPixmap, Optional[AlignmentTransform]]:
    """Crop both images to show only the overlapping FOV region.

    This mode shows only the content within the LWIR FOV, cropping out
    all areas outside. Both images are aligned and cropped to the same
    FOV rectangle, providing a clean comparison view.

    Args:
        lwir_pixmap: LWIR camera pixmap
        vis_pixmap: Visible camera pixmap
        homography: Homography matrix (LWIR→Visible mapping)
        lwir_calib_size: LWIR calibration size (width, height)
        vis_calib_size: Visible calibration size (width, height)

    Returns:
        Tuple of (cropped_lwir, cropped_vis, transform)
    """
    if cv2 is None or np is None or homography is None:
        return (lwir_pixmap, vis_pixmap, None)

    try:
        # Convert to arrays
        lwir_arr = _pixmap_to_array(lwir_pixmap)
        vis_arr = _pixmap_to_array(vis_pixmap)

        if lwir_arr is None or vis_arr is None:
            return (lwir_pixmap, vis_pixmap, None)

        # === LWIR: Crop black borders to get valid region ===
        lwir_cropped, crop_rect = _crop_black_borders(lwir_arr)
        lwir_h, lwir_w = lwir_cropped.shape[:2]

        log_debug(f"align_fov_crop: LWIR cropped from {lwir_arr.shape} to {lwir_cropped.shape}", "ALIGN")

        # Get display sizes
        vis_h, vis_w = vis_arr.shape[:2]

        # Scale factors from calibration to display
        vis_scale_x = vis_w / vis_calib_size[0]
        vis_scale_y = vis_h / vis_calib_size[1]

        # LWIR dimensions for calibration
        orig_lwir_w, orig_lwir_h = lwir_arr.shape[1], lwir_arr.shape[0]
        lwir_scale_x = orig_lwir_w / lwir_calib_size[0]
        lwir_scale_y = orig_lwir_h / lwir_calib_size[1]

        # Crop rect in calibration coords
        crop_x_calib = crop_rect[0] / lwir_scale_x
        crop_y_calib = crop_rect[1] / lwir_scale_y
        crop_w_calib = crop_rect[2] / lwir_scale_x
        crop_h_calib = crop_rect[3] / lwir_scale_y

        lwir_corners_calib = np.array([
            [crop_x_calib, crop_y_calib],
            [crop_x_calib + crop_w_calib, crop_y_calib],
            [crop_x_calib + crop_w_calib, crop_y_calib + crop_h_calib],
            [crop_x_calib, crop_y_calib + crop_h_calib],
        ], dtype=np.float32).reshape(-1, 1, 2)

        # Map LWIR corners to visible coordinates (in calibration space)
        fov_corners_calib = cv2.perspectiveTransform(
            lwir_corners_calib,
            homography.astype(np.float32)
        ).reshape(-1, 2)

        # Convert FOV corners to visible display coordinates
        fov_corners_display = fov_corners_calib.copy()
        fov_corners_display[:, 0] *= vis_scale_x
        fov_corners_display[:, 1] *= vis_scale_y

        log_debug(f"align_fov_crop: FOV corners in vis display: {fov_corners_display.tolist()}", "ALIGN")

        # Calculate rotation angle from top edge of FOV
        dx = fov_corners_display[1][0] - fov_corners_display[0][0]
        dy = fov_corners_display[1][1] - fov_corners_display[0][1]
        angle = math.degrees(math.atan2(dy, dx))

        # Get FOV center
        fov_center_x = fov_corners_display[:, 0].mean()
        fov_center_y = fov_corners_display[:, 1].mean()

        # Calculate FOV size
        fov_w = math.sqrt(dx**2 + dy**2)
        dx2 = fov_corners_display[3][0] - fov_corners_display[0][0]
        dy2 = fov_corners_display[3][1] - fov_corners_display[0][1]
        fov_h = math.sqrt(dx2**2 + dy2**2)

        log_debug(f"align_fov_crop: FOV size {fov_w:.0f}x{fov_h:.0f}, angle {angle:.2f}°", "ALIGN")

        # Output size is exactly the FOV size
        out_w = int(fov_w)
        out_h = int(fov_h)

        # === Visible: Rotate and translate to extract FOV region ===
        # Rotation around FOV center
        M_rot = cv2.getRotationMatrix2D((fov_center_x, fov_center_y), angle, 1.0)

        # After rotation, FOV center stays at same position
        # We want FOV center to be at output center
        out_center_x = out_w / 2
        out_center_y = out_h / 2

        # Add translation to move FOV center to output center
        M_rot[0, 2] += out_center_x - fov_center_x
        M_rot[1, 2] += out_center_y - fov_center_y

        # Transform visible image - crop to FOV size
        vis_cropped = cv2.warpAffine(
            vis_arr, M_rot, (out_w, out_h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=CHROMA_GREEN
        )

        # === LWIR: Scale cropped LWIR to match output size ===
        lwir_scale = min(out_w / lwir_w, out_h / lwir_h)
        new_lwir_w = int(lwir_w * lwir_scale)
        new_lwir_h = int(lwir_h * lwir_scale)
        lwir_scaled = cv2.resize(lwir_cropped, (new_lwir_w, new_lwir_h), interpolation=cv2.INTER_AREA)

        # Center LWIR in output
        lwir_result = np.full((out_h, out_w, 3), CHROMA_GREEN, dtype=np.uint8)
        lwir_x = (out_w - new_lwir_w) // 2
        lwir_y = (out_h - new_lwir_h) // 2
        lwir_result[lwir_y:lwir_y + new_lwir_h, lwir_x:lwir_x + new_lwir_w] = lwir_scaled

        # === Calculate Max Overlap region within this FOV crop ===
        # Transform visible corners through the rotation
        vis_corners = np.array([
            [0, 0], [vis_w, 0], [vis_w, vis_h], [0, vis_h]
        ], dtype=np.float32)
        vis_corners_h = np.hstack([vis_corners, np.ones((4, 1))])
        vis_corners_out = (M_rot @ vis_corners_h.T).T.astype(np.float32)

        # Helper functions for line intersection
        def line_x_at_y(p1: np.ndarray, p2: np.ndarray, y: float) -> float:
            if abs(p2[1] - p1[1]) < 1e-6:
                return (p1[0] + p2[0]) / 2
            t = (y - p1[1]) / (p2[1] - p1[1])
            return p1[0] + t * (p2[0] - p1[0])

        def line_y_at_x(p1: np.ndarray, p2: np.ndarray, x: float) -> float:
            if abs(p2[0] - p1[0]) < 1e-6:
                return (p1[1] + p2[1]) / 2
            t = (x - p1[0]) / (p2[0] - p1[0])
            return p1[1] + t * (p2[1] - p1[1])

        v = vis_corners_out
        # Left/right/top/bottom constraints from visible polygon
        left_x = max(line_x_at_y(v[3], v[0], 0), line_x_at_y(v[3], v[0], out_h))
        right_x = min(line_x_at_y(v[1], v[2], 0), line_x_at_y(v[1], v[2], out_h))
        top_y = max(line_y_at_x(v[0], v[1], 0), line_y_at_x(v[0], v[1], out_w))
        bottom_y = min(line_y_at_x(v[3], v[2], 0), line_y_at_x(v[3], v[2], out_w))

        # Max overlap within FOV crop bounds
        mo_x1 = int(max(0, left_x))
        mo_x2 = int(min(out_w, right_x))
        mo_y1 = int(max(0, top_y))
        mo_y2 = int(min(out_h, bottom_y))

        # Draw "FOV Focus" text on both images
        cv2.putText(
            vis_cropped, "FOV Focus",
            (5, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )
        cv2.putText(
            lwir_result, "FOV Focus",
            (5, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )

        # Draw cyan Max Overlap rectangle if valid
        if mo_x2 > mo_x1 and mo_y2 > mo_y1:
            cv2.rectangle(
                vis_cropped,
                (mo_x1, mo_y1),
                (mo_x2, mo_y2),
                (255, 255, 0), 2  # Cyan in BGR
            )
            cv2.putText(
                vis_cropped, "Max Overlap",
                (mo_x1 + 5, mo_y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1
            )
            cv2.rectangle(
                lwir_result,
                (mo_x1, mo_y1),
                (mo_x2, mo_y2),
                (255, 255, 0), 2  # Cyan in BGR
            )
            cv2.putText(
                lwir_result, "Max Overlap",
                (mo_x1 + 5, mo_y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1
            )

        # Create transform object for coordinate mapping
        transform = AlignmentTransform(
            vis_matrix=M_rot,
            lwir_offset=(lwir_x, lwir_y),
            lwir_scale=lwir_scale,
            output_size=(out_w, out_h),
            fov_corners_out=np.array([
                [0, 0], [out_w, 0], [out_w, out_h], [0, out_h]
            ], dtype=np.float32),
            lwir_crop_offset=(crop_rect[0], crop_rect[1]),
        )

        # Convert back to pixmaps
        lwir_pm = _array_to_pixmap(lwir_result)
        vis_pm = _array_to_pixmap(vis_cropped)

        log_debug(f"align_fov_crop: result {out_w}x{out_h}", "ALIGN")

        return (lwir_pm or lwir_pixmap, vis_pm or vis_pixmap, transform)

    except Exception as e:
        log_warning(f"align_fov_crop failed: {e}", "ALIGN")
        import traceback
        log_debug(f"Traceback: {traceback.format_exc()}", "ALIGN")
        return (lwir_pixmap, vis_pixmap, None)


def align_max_overlap(
    lwir_pixmap: QPixmap,
    vis_pixmap: QPixmap,
    homography: Any,
    lwir_calib_size: Tuple[int, int],
    vis_calib_size: Tuple[int, int],
) -> Tuple[QPixmap, QPixmap, Optional[AlignmentTransform]]:
    """Align images showing only the maximum overlap area with NO green padding.

    This mode crops both images to show exactly the overlapping region:
    - FOV Focus = LWIR valid region projected to visible
    - Max Overlap = intersection of FOV Focus with visible image bounds
    - Both images are cropped to this region, no padding

    Args:
        lwir_pixmap: LWIR camera pixmap
        vis_pixmap: Visible camera pixmap
        homography: Homography matrix (LWIR→Visible mapping)
        lwir_calib_size: LWIR calibration size (width, height)
        vis_calib_size: Visible calibration size (width, height)

    Returns:
        Tuple of (aligned_lwir, aligned_vis, transform)
    """
    if cv2 is None or np is None or homography is None:
        return (lwir_pixmap, vis_pixmap, None)

    try:
        # Convert to arrays
        lwir_arr = _pixmap_to_array(lwir_pixmap)
        vis_arr = _pixmap_to_array(vis_pixmap)

        if lwir_arr is None or vis_arr is None:
            return (lwir_pixmap, vis_pixmap, None)

        # === LWIR: Crop black borders to get valid region ===
        lwir_cropped, crop_rect = _crop_black_borders(lwir_arr)
        lwir_h, lwir_w = lwir_cropped.shape[:2]

        log_debug(f"align_max_overlap: LWIR cropped {lwir_w}x{lwir_h}", "ALIGN")

        # Get display sizes
        vis_h, vis_w = vis_arr.shape[:2]

        # Scale factors from calibration to display
        vis_scale_x = vis_w / vis_calib_size[0]
        vis_scale_y = vis_h / vis_calib_size[1]

        orig_lwir_w, orig_lwir_h = lwir_arr.shape[1], lwir_arr.shape[0]
        lwir_scale_x = orig_lwir_w / lwir_calib_size[0]
        lwir_scale_y = orig_lwir_h / lwir_calib_size[1]

        # LWIR valid region corners in calibration coords
        crop_x_calib = crop_rect[0] / lwir_scale_x
        crop_y_calib = crop_rect[1] / lwir_scale_y
        crop_w_calib = crop_rect[2] / lwir_scale_x
        crop_h_calib = crop_rect[3] / lwir_scale_y

        lwir_corners_calib = np.array([
            [crop_x_calib, crop_y_calib],
            [crop_x_calib + crop_w_calib, crop_y_calib],
            [crop_x_calib + crop_w_calib, crop_y_calib + crop_h_calib],
            [crop_x_calib, crop_y_calib + crop_h_calib],
        ], dtype=np.float32).reshape(-1, 1, 2)

        # Project LWIR corners to visible (FOV Focus)
        fov_corners_calib = cv2.perspectiveTransform(
            lwir_corners_calib,
            homography.astype(np.float32)
        ).reshape(-1, 2)

        # Convert to visible display coordinates
        fov_corners_display = fov_corners_calib.copy()
        fov_corners_display[:, 0] *= vis_scale_x
        fov_corners_display[:, 1] *= vis_scale_y

        # FOV Focus bounding box
        fov_x1 = int(fov_corners_display[:, 0].min())
        fov_y1 = int(fov_corners_display[:, 1].min())
        fov_x2 = int(fov_corners_display[:, 0].max())
        fov_y2 = int(fov_corners_display[:, 1].max())

        # Max Overlap = intersection of FOV Focus with visible image bounds
        crop_x1 = max(0, fov_x1)
        crop_y1 = max(0, fov_y1)
        crop_x2 = min(vis_w, fov_x2)
        crop_y2 = min(vis_h, fov_y2)

        overlap_w = crop_x2 - crop_x1
        overlap_h = crop_y2 - crop_y1

        if overlap_w <= 0 or overlap_h <= 0:
            log_warning("align_max_overlap: no valid overlap region", "ALIGN")
            return (lwir_pixmap, vis_pixmap, None)

        # FOV dimensions in visible display space
        fov_w = fov_x2 - fov_x1
        fov_h = fov_y2 - fov_y1

        if fov_w <= 0 or fov_h <= 0:
            log_warning("align_max_overlap: invalid FOV dimensions", "ALIGN")
            return (lwir_pixmap, vis_pixmap, None)

        # Calculate how much of the FOV is clipped by image bounds
        # These are fractions (0.0 = no clip, 1.0 = fully clipped)
        clip_left = (crop_x1 - fov_x1) / fov_w  # How much is clipped on left
        clip_top = (crop_y1 - fov_y1) / fov_h   # How much is clipped on top
        clip_right = (fov_x2 - crop_x2) / fov_w  # How much is clipped on right
        clip_bottom = (fov_y2 - crop_y2) / fov_h  # How much is clipped on bottom

        # Calculate the corresponding crop region in LWIR
        # The FOV represents the full lwir_cropped, so we need to crop the same fractions
        lwir_crop_x1 = int(clip_left * lwir_w)
        lwir_crop_y1 = int(clip_top * lwir_h)
        lwir_crop_x2 = int(lwir_w - clip_right * lwir_w)
        lwir_crop_y2 = int(lwir_h - clip_bottom * lwir_h)

        # Ensure valid bounds
        lwir_crop_x1 = max(0, lwir_crop_x1)
        lwir_crop_y1 = max(0, lwir_crop_y1)
        lwir_crop_x2 = min(lwir_w, lwir_crop_x2)
        lwir_crop_y2 = min(lwir_h, lwir_crop_y2)

        lwir_visible_w = lwir_crop_x2 - lwir_crop_x1
        lwir_visible_h = lwir_crop_y2 - lwir_crop_y1

        if lwir_visible_w <= 0 or lwir_visible_h <= 0:
            log_warning("align_max_overlap: no visible LWIR region after crop", "ALIGN")
            return (lwir_pixmap, vis_pixmap, None)

        # Crop LWIR to only the visible portion
        lwir_visible = lwir_cropped[lwir_crop_y1:lwir_crop_y2, lwir_crop_x1:lwir_crop_x2]

        # Now calculate uniform scale to preserve aspect ratio
        lwir_aspect = lwir_visible_w / lwir_visible_h if lwir_visible_h > 0 else 1.0
        vis_aspect = overlap_w / overlap_h if overlap_h > 0 else 1.0

        # Output size: use the visible overlap dimensions, both images will match
        out_w = overlap_w
        out_h = overlap_h

        log_debug(f"align_max_overlap: FOV ({fov_x1},{fov_y1})-({fov_x2},{fov_y2})", "ALIGN")
        log_debug(f"align_max_overlap: LWIR visible crop ({lwir_crop_x1},{lwir_crop_y1})-({lwir_crop_x2},{lwir_crop_y2})", "ALIGN")
        log_debug(f"align_max_overlap: Max Overlap ({crop_x1},{crop_y1}) {out_w}x{out_h}", "ALIGN")

        # === VISIBLE: Crop to max overlap region ===
        vis_result = vis_arr[crop_y1:crop_y2, crop_x1:crop_x2].copy()

        # === LWIR: Scale the cropped visible portion to match visible output size ===
        lwir_result = cv2.resize(lwir_visible, (out_w, out_h), interpolation=cv2.INTER_AREA)

        # Calculate effective scale for transform
        uniform_scale = out_w / lwir_visible_w if lwir_visible_w > 0 else 1.0

        # Draw "Max Overlap" text on both images
        cv2.putText(
            vis_result, "Max Overlap",
            (5, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
        )
        cv2.putText(
            lwir_result, "Max Overlap",
            (5, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
        )

        # Create transform object with uniform scale
        # lwir_crop_offset now includes both the black border crop AND the visible crop
        total_lwir_offset_x = crop_rect[0] + lwir_crop_x1
        total_lwir_offset_y = crop_rect[1] + lwir_crop_y1
        transform = AlignmentTransform(
            vis_matrix=np.array([[1, 0, -crop_x1], [0, 1, -crop_y1]], dtype=np.float32),
            lwir_offset=(0, 0),
            lwir_scale=uniform_scale,
            output_size=(out_w, out_h),
            fov_corners_out=np.array([
                [0, 0], [out_w, 0], [out_w, out_h], [0, out_h]
            ], dtype=np.float32),
            lwir_crop_offset=(total_lwir_offset_x, total_lwir_offset_y),
        )

        # Convert back to pixmaps
        lwir_pm = _array_to_pixmap(lwir_result)
        vis_pm = _array_to_pixmap(vis_result)

        log_debug(f"align_max_overlap: result {out_w}x{out_h}", "ALIGN")

        return (lwir_pm or lwir_pixmap, vis_pm or vis_pixmap, transform)

    except Exception as e:
        log_warning(f"align_max_overlap failed: {e}", "ALIGN")
        import traceback
        log_debug(f"Traceback: {traceback.format_exc()}", "ALIGN")
        return (lwir_pixmap, vis_pixmap, None)


def draw_fov_overlay(
    lwir_pixmap: QPixmap,
    vis_pixmap: QPixmap,
    homography_inv: Any,
    source_channel: str,
    lwir_calib_size: Optional[Tuple[int, int]] = None,
    vis_calib_size: Optional[Tuple[int, int]] = None,
) -> Tuple[QPixmap, QPixmap]:
    """Draw FOV overlap indicator on the larger FOV camera.

    Instead of warping images, this draws a rectangle on the larger FOV
    camera showing which region is visible to the smaller FOV camera.
    This is more useful for visual comparison.

    Args:
        lwir_pixmap: LWIR camera pixmap (display size)
        vis_pixmap: Visible camera pixmap (display size)
        homography_inv: Inverse homography (target->source mapping, in calibration coordinates)
        source_channel: Which channel has larger FOV ("lwir" or "visible")
        lwir_calib_size: Original LWIR calibration size (width, height)
        vis_calib_size: Original visible calibration size (width, height)

    Returns:
        Tuple of (lwir_pixmap, vis_pixmap) with overlay drawn on larger FOV
    """
    log_debug(f"draw_fov_overlay: source={source_channel}, lwir={lwir_pixmap.size()}, vis={vis_pixmap.size()}", "ALIGN")
    log_debug(f"draw_fov_overlay: calib sizes lwir={lwir_calib_size}, vis={vis_calib_size}", "ALIGN")

    if cv2 is None or np is None or homography_inv is None:
        log_warning("draw_fov_overlay: missing cv2, np, or homography", "ALIGN")
        return (lwir_pixmap, vis_pixmap)

    try:
        # Determine which pixmap to draw on
        if source_channel == "lwir":
            # LWIR is larger FOV, draw on it showing visible's FOV
            source_pm = lwir_pixmap
            target_pm = vis_pixmap
            source_calib = lwir_calib_size or (lwir_pixmap.width(), lwir_pixmap.height())
            target_calib = vis_calib_size or (vis_pixmap.width(), vis_pixmap.height())
        else:
            # Visible is larger FOV, draw on it showing LWIR's FOV
            source_pm = vis_pixmap
            target_pm = lwir_pixmap
            source_calib = vis_calib_size or (vis_pixmap.width(), vis_pixmap.height())
            target_calib = lwir_calib_size or (lwir_pixmap.width(), lwir_pixmap.height())

        # Calculate scaling factors from calibration coords to display coords
        source_scale_x = source_pm.width() / source_calib[0]
        source_scale_y = source_pm.height() / source_calib[1]
        target_scale_x = target_pm.width() / target_calib[0]
        target_scale_y = target_pm.height() / target_calib[1]

        log_debug(f"draw_fov_overlay: source_scale={source_scale_x:.3f}x{source_scale_y:.3f}, target_scale={target_scale_x:.3f}x{target_scale_y:.3f}", "ALIGN")

        # Target corners in display coordinates - convert to calibration coords
        tw, th = target_pm.width(), target_pm.height()
        target_corners_display = np.array([
            [0, 0],
            [tw, 0],
            [tw, th],
            [0, th],
        ], dtype=np.float32)

        # Scale to calibration coordinates
        target_corners_calib = target_corners_display.copy()
        target_corners_calib[:, 0] /= target_scale_x
        target_corners_calib[:, 1] /= target_scale_y
        target_corners_calib = target_corners_calib.reshape(-1, 1, 2)

        # Map target corners (in calibration coords) to source (in calibration coords)
        source_corners_calib = cv2.perspectiveTransform(
            target_corners_calib,
            homography_inv.astype(np.float32)
        ).reshape(-1, 2)

        # Scale source corners to display coordinates
        source_corners_display = source_corners_calib.copy()
        source_corners_display[:, 0] *= source_scale_x
        source_corners_display[:, 1] *= source_scale_y

        log_debug(f"draw_fov_overlay: source corners (display): {source_corners_display.tolist()}", "ALIGN")

        # Draw the quadrilateral on the source image
        source_arr = _pixmap_to_array(source_pm)
        if source_arr is None:
            log_warning("draw_fov_overlay: failed to convert source pixmap to array", "ALIGN")
            return (lwir_pixmap, vis_pixmap)

        log_debug(f"draw_fov_overlay: source array shape {source_arr.shape}", "ALIGN")
        result = source_arr.copy()

        # Draw semi-transparent overlay outside the visible region
        # Create a mask for the visible region
        mask = np.zeros(result.shape[:2], dtype=np.uint8)
        pts = source_corners_display.astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)

        # Darken areas outside the visible region
        darkened = (result * 0.4).astype(np.uint8)
        result = np.where(mask[:, :, np.newaxis] > 0, result, darkened)

        # Draw the boundary with a bright color
        cv2.polylines(result, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

        # Add label - determine which camera we're showing the FOV of
        label = "LWIR FOV" if source_channel == "visible" else "Visible FOV"
        cv2.putText(
            result,
            label,
            (int(pts[0][0]) + 5, int(pts[0][1]) + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        result_pm = _array_to_pixmap(result)
        log_debug(f"draw_fov_overlay: result pixmap null={result_pm.isNull()}", "ALIGN")

        if source_channel == "lwir":
            return (result_pm or lwir_pixmap, vis_pixmap)
        else:
            return (lwir_pixmap, result_pm or vis_pixmap)

    except Exception as e:
        log_warning(f"Failed to draw FOV overlay: {e}", "ALIGN")
        return (lwir_pixmap, vis_pixmap)


def compute_stereo_rectification(
    lwir_matrices: Dict[str, Any],
    vis_matrices: Dict[str, Any],
    extrinsic: Dict[str, Any],
    display_size: Tuple[int, int],
) -> Optional[Dict[str, Any]]:
    """Compute stereo rectification maps and valid region for both cameras.

    The rectification is computed at the calibration resolution of each camera,
    then the maps are generated to output at display_size. This preserves the
    original calibration accuracy while allowing display at any size.

    Args:
        lwir_matrices: LWIR intrinsic calibration dict
        vis_matrices: Visible intrinsic calibration dict
        extrinsic: Extrinsic calibration (R, T between cameras)
        display_size: Target display size (width, height)

    Returns:
        Dict with rectification maps and valid regions, or None on failure
    """
    if cv2 is None or np is None:
        return None

    try:
        # Extract intrinsic matrices (at calibration resolution)
        K_lwir = _dict_to_matrix(lwir_matrices, (3, 3))
        K_vis = _dict_to_matrix(vis_matrices, (3, 3))

        # Get distortion coefficients
        D_lwir = _dict_to_matrix({"data": lwir_matrices.get("distortion")}, (5,))
        D_vis = _dict_to_matrix({"data": vis_matrices.get("distortion")}, (5,))
        if D_lwir is None:
            D_lwir = np.zeros(5)
        if D_vis is None:
            D_vis = np.zeros(5)

        # Get calibration sizes
        lwir_size_raw = lwir_matrices.get("image_size")
        vis_size_raw = vis_matrices.get("image_size")

        if isinstance(lwir_size_raw, list) and len(lwir_size_raw) >= 2:
            lwir_calib_size = (int(lwir_size_raw[0]), int(lwir_size_raw[1]))
        else:
            lwir_calib_size = display_size

        if isinstance(vis_size_raw, list) and len(vis_size_raw) >= 2:
            vis_calib_size = (int(vis_size_raw[0]), int(vis_size_raw[1]))
        else:
            vis_calib_size = display_size

        # Get extrinsic (R, T) - typically LWIR to Visible
        R = _dict_to_matrix({"data": extrinsic.get("rotation") or extrinsic.get("R")}, (3, 3))
        T = _dict_to_matrix({"data": extrinsic.get("translation") or extrinsic.get("T")}, (3, 1))

        if K_lwir is None or K_vis is None or R is None or T is None:
            log_warning("compute_stereo_rectification: missing matrix data", "ALIGN")
            return None

        log_debug(f"Stereo rectification: LWIR calib={lwir_calib_size}, Vis calib={vis_calib_size}", "ALIGN")
        log_debug(f"Stereo rectification: K_lwir=\n{K_lwir}", "ALIGN")
        log_debug(f"Stereo rectification: K_vis=\n{K_vis}", "ALIGN")

        # Compute stereo rectification at LWIR resolution (smaller camera)
        # We use LWIR size as the common output size since both cameras will display at that size
        # stereoRectify expects both cameras to have same resolution, so we use the smaller one
        common_size = lwir_calib_size

        # Scale K_vis to match LWIR resolution for stereoRectify
        scale_vis_x = common_size[0] / vis_calib_size[0]
        scale_vis_y = common_size[1] / vis_calib_size[1]

        K_vis_scaled = K_vis.copy()
        K_vis_scaled[0, 0] *= scale_vis_x  # fx
        K_vis_scaled[1, 1] *= scale_vis_y  # fy
        K_vis_scaled[0, 2] *= scale_vis_x  # cx
        K_vis_scaled[1, 2] *= scale_vis_y  # cy

        log_debug(f"Stereo rectification: K_vis_scaled=\n{K_vis_scaled}", "ALIGN")
        log_debug(f"Stereo rectification: R=\n{R}", "ALIGN")
        log_debug(f"Stereo rectification: T={T.flatten()}", "ALIGN")

        # Compute stereo rectification at common size
        # Use alpha=1.0 to keep all pixels (may have black borders)
        # alpha=0 would crop to valid region only but can give empty ROI
        R1, R2, P1, P2, Q, roi_lwir, roi_vis = cv2.stereoRectify(
            K_lwir, D_lwir, K_vis_scaled, D_vis,
            common_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=1.0,  # 1 = keep all pixels (may have black borders)
        )

        log_debug(f"Stereo rectification: roi_lwir={roi_lwir}, roi_vis={roi_vis}", "ALIGN")
        log_debug(f"Stereo rectification: P1=\n{P1}", "ALIGN")
        log_debug(f"Stereo rectification: P2=\n{P2}", "ALIGN")

        # Compute undistort+rectify maps
        # LWIR: input at lwir_calib_size, output at display_size
        # We need maps that account for the transformation from input to output

        # Scale output P matrices if display_size differs from common_size
        scale_out_x = display_size[0] / common_size[0]
        scale_out_y = display_size[1] / common_size[1]

        P1_display = P1.copy()
        P1_display[0, :] *= scale_out_x
        P1_display[1, :] *= scale_out_y

        P2_display = P2.copy()
        P2_display[0, :] *= scale_out_x
        P2_display[1, :] *= scale_out_y

        # For LWIR: input is at lwir_calib_size (same as common_size), output at display_size
        map1_lwir, map2_lwir = cv2.initUndistortRectifyMap(
            K_lwir, D_lwir, R1, P1_display, display_size, cv2.CV_32FC1
        )

        # For visible: input is at vis_calib_size, output at display_size
        # We need to use original K_vis (not scaled) because the input image is at original resolution
        # But we need the output to match LWIR's rectified space
        # The visible input needs to be scaled to match common_size first
        map1_vis, map2_vis = cv2.initUndistortRectifyMap(
            K_vis_scaled, D_vis, R2, P2_display, display_size, cv2.CV_32FC1
        )

        # Scale ROIs to display size
        roi_lwir_display = (
            int(roi_lwir[0] * scale_out_x),
            int(roi_lwir[1] * scale_out_y),
            int(roi_lwir[2] * scale_out_x),
            int(roi_lwir[3] * scale_out_y),
        )
        roi_vis_display = (
            int(roi_vis[0] * scale_out_x),
            int(roi_vis[1] * scale_out_y),
            int(roi_vis[2] * scale_out_x),
            int(roi_vis[3] * scale_out_y),
        )

        # Compute valid region intersection (common visible area)
        x1 = max(roi_lwir_display[0], roi_vis_display[0])
        y1 = max(roi_lwir_display[1], roi_vis_display[1])
        x2 = min(roi_lwir_display[0] + roi_lwir_display[2], roi_vis_display[0] + roi_vis_display[2])
        y2 = min(roi_lwir_display[1] + roi_lwir_display[3], roi_vis_display[1] + roi_vis_display[3])

        valid_roi = (x1, y1, max(0, x2 - x1), max(0, y2 - y1)) if x2 > x1 and y2 > y1 else None

        log_debug(f"Stereo rectification: ROI_LWIR={roi_lwir_display}, ROI_VIS={roi_vis_display}, valid={valid_roi}", "ALIGN")

        return {
            "map_lwir": (map1_lwir, map2_lwir),
            "map_vis": (map1_vis, map2_vis),
            "roi_lwir": roi_lwir_display,
            "roi_vis": roi_vis_display,
            "valid_roi": valid_roi,
            "lwir_calib_size": lwir_calib_size,
            "vis_calib_size": vis_calib_size,
            "common_size": common_size,
        }

    except Exception as e:
        log_warning(f"compute_stereo_rectification failed: {e}", "ALIGN")
        import traceback
        log_debug(f"Traceback: {traceback.format_exc()}", "ALIGN")
        return None


def apply_stereo_rectification(
    lwir_pixmap: QPixmap,
    vis_pixmap: QPixmap,
    rect_info: Dict[str, Any],
    show_valid_region: bool = True,
    border_color: Tuple[int, int, int] = (0, 255, 255),
) -> Tuple[QPixmap, QPixmap]:
    """Apply stereo rectification to align both images to a common plane.

    Both images are warped so that corresponding points lie on the same
    horizontal line (epipolar alignment). The valid region is highlighted.

    Args:
        lwir_pixmap: LWIR image
        vis_pixmap: Visible image
        rect_info: Rectification info from compute_stereo_rectification
        show_valid_region: Whether to highlight valid overlap region
        border_color: Color for valid region border (BGR)

    Returns:
        Tuple of (rectified_lwir, rectified_vis) pixmaps
    """
    if cv2 is None or np is None or not rect_info:
        return (lwir_pixmap, vis_pixmap)

    try:
        # Convert pixmaps to arrays
        lwir_arr = _pixmap_to_array(lwir_pixmap)
        vis_arr = _pixmap_to_array(vis_pixmap)

        if lwir_arr is None or vis_arr is None:
            log_warning("apply_stereo_rectification: failed to convert pixmaps", "ALIGN")
            return (lwir_pixmap, vis_pixmap)

        # Get rectification maps
        map_lwir = rect_info["map_lwir"]
        map_vis = rect_info["map_vis"]
        common_size = rect_info.get("common_size", (lwir_arr.shape[1], lwir_arr.shape[0]))

        # Log input sizes for debugging
        log_debug(f"apply_stereo_rectification: lwir_arr={lwir_arr.shape}, vis_arr={vis_arr.shape}", "ALIGN")
        log_debug(f"apply_stereo_rectification: map_lwir shape={map_lwir[0].shape}, map_vis shape={map_vis[0].shape}", "ALIGN")
        log_debug(f"apply_stereo_rectification: common_size={common_size}", "ALIGN")

        # The maps expect input at common_size (which is lwir_calib_size = 640x480)
        # Input pixmaps should already be at display_size which should match common_size
        # If sizes don't match, resize inputs to common_size
        expected_h, expected_w = map_lwir[0].shape[:2]

        if lwir_arr.shape[:2] != (expected_h, expected_w):
            log_debug(f"apply_stereo_rectification: resizing lwir from {lwir_arr.shape[:2]} to {(expected_h, expected_w)}", "ALIGN")
            lwir_arr = cv2.resize(lwir_arr, (expected_w, expected_h), interpolation=cv2.INTER_AREA)

        if vis_arr.shape[:2] != (expected_h, expected_w):
            log_debug(f"apply_stereo_rectification: resizing vis from {vis_arr.shape[:2]} to {(expected_h, expected_w)}", "ALIGN")
            vis_arr = cv2.resize(vis_arr, (expected_w, expected_h), interpolation=cv2.INTER_AREA)

        # Apply rectification using remap
        lwir_rect = cv2.remap(lwir_arr, map_lwir[0], map_lwir[1], cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))
        vis_rect = cv2.remap(vis_arr, map_vis[0], map_vis[1], cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))

        # Highlight valid region
        if show_valid_region:
            valid_roi = rect_info.get("valid_roi")
            if valid_roi and valid_roi[2] > 0 and valid_roi[3] > 0:
                x, y, w, h = valid_roi
                # Draw border around valid region on both images
                cv2.rectangle(lwir_rect, (x, y), (x + w, y + h), border_color, 2)
                cv2.rectangle(vis_rect, (x, y), (x + w, y + h), border_color, 2)

                # Darken outside valid region
                mask = np.zeros(lwir_rect.shape[:2], dtype=np.uint8)
                mask[y:y+h, x:x+w] = 255

                lwir_dark = (lwir_rect * 0.4).astype(np.uint8)
                vis_dark = (vis_rect * 0.4).astype(np.uint8)

                lwir_rect = np.where(mask[:, :, np.newaxis] > 0, lwir_rect, lwir_dark)
                vis_rect = np.where(mask[:, :, np.newaxis] > 0, vis_rect, vis_dark)

                # Re-draw border after darkening
                cv2.rectangle(lwir_rect, (x, y), (x + w, y + h), border_color, 2)
                cv2.rectangle(vis_rect, (x, y), (x + w, y + h), border_color, 2)

        # Convert back to pixmaps
        lwir_pm = _array_to_pixmap(lwir_rect)
        vis_pm = _array_to_pixmap(vis_rect)

        return (lwir_pm or lwir_pixmap, vis_pm or vis_pixmap)

    except Exception as e:
        log_warning(f"apply_stereo_rectification failed: {e}", "ALIGN")
        return (lwir_pixmap, vis_pixmap)


def align_image(
    source_pixmap: QPixmap,
    target_pixmap: QPixmap,
    homography: Any,
    show_non_overlap_grayscale: bool = True,
    overlap_color: Optional[QColor] = None,
) -> Tuple[QPixmap, QPixmap]:
    """Align source image to target coordinate system.

    The source image is warped using the homography to match the target's
    perspective. Optionally shows areas outside the overlap in grayscale
    and draws a rectangle around the aligned region.

    Args:
        source_pixmap: Source image to transform
        target_pixmap: Target image (used for size, optionally modified)
        homography: 3x3 homography matrix
        show_non_overlap_grayscale: If True, areas outside overlap are grayscale
        overlap_color: Color for overlap rectangle (e.g., red)

    Returns:
        Tuple of (aligned_source, modified_target) pixmaps
    """
    if cv2 is None or np is None or homography is None:
        return (source_pixmap, target_pixmap)

    try:
        # Convert source pixmap to numpy array
        source_arr = _pixmap_to_array(source_pixmap)
        if source_arr is None:
            log_warning("align_image: Could not convert source pixmap to array", "ALIGN")
            return (source_pixmap, target_pixmap)

        target_size = (target_pixmap.width(), target_pixmap.height())
        log_debug(f"align_image: Warping source {source_arr.shape} to target size {target_size}", "ALIGN")

        # Warp source image
        warped = cv2.warpPerspective(
            source_arr,
            homography.astype(np.float32),
            target_size,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # Create mask for valid pixels (non-black after warp)
        mask = (warped.sum(axis=2) > 0).astype(np.uint8)

        # Compute overlap region
        overlap = compute_overlap_region(
            (source_pixmap.width(), source_pixmap.height()),
            target_size,
            homography,
        )

        # Apply grayscale to non-overlap areas if requested
        if show_non_overlap_grayscale:
            # Convert non-overlapping areas to grayscale
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # Only keep color in overlap, grayscale outside
            if overlap:
                x, y, w, h = overlap
                result = gray_bgr.copy()
                result[y:y+h, x:x+w] = warped[y:y+h, x:x+w]
                warped = result

        # Draw overlap rectangle if color specified
        if overlap_color and overlap:
            x, y, w, h = overlap
            r, g, b = overlap_color.red(), overlap_color.green(), overlap_color.blue()
            cv2.rectangle(warped, (x, y), (x + w - 1, y + h - 1), (b, g, r), 2)

        aligned_source = _array_to_pixmap(warped)

        return (aligned_source, target_pixmap)

    except Exception as e:
        log_warning(f"Failed to align image: {e}", "ALIGN")
        return (source_pixmap, target_pixmap)


def _pixmap_to_array(pixmap: QPixmap) -> Optional[Any]:
    """Convert QPixmap to numpy BGR array."""
    if np is None:
        return None

    try:
        image = pixmap.toImage()
        image = image.convertToFormat(QImage.Format.Format_RGBA8888)

        width = image.width()
        height = image.height()

        # Qt6 approach: get constBits and convert to numpy array
        ptr = image.constBits()
        if ptr is None:
            log_warning("_pixmap_to_array: constBits() returned None", "ALIGN")
            return None

        # Try direct numpy conversion via memoryview
        bytes_per_line = image.bytesPerLine()
        expected_bytes = height * bytes_per_line

        try:
            # Qt6 PyQt6: ptr.asarray(size) returns a memoryview
            buffer = ptr.asarray(expected_bytes)
            arr = np.frombuffer(buffer, dtype=np.uint8).reshape((height, bytes_per_line))
            # Extract only the image data (width * 4 bytes per row)
            arr = arr[:, :width * 4].reshape((height, width, 4))
        except (TypeError, AttributeError):
            # Fallback: try direct conversion
            arr = np.array(ptr, dtype=np.uint8).reshape((height, bytes_per_line))[:, :width * 4]
            arr = arr.reshape((height, width, 4))

        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

    except Exception as e:
        log_warning(f"_pixmap_to_array failed: {e}", "ALIGN")
        return None


def _array_to_pixmap(arr: Any) -> QPixmap:
    """Convert numpy BGR array to QPixmap."""
    try:
        # Convert BGR to RGB for Qt
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]

        bytes_per_line = 3 * width
        image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        # Make a deep copy to avoid memory issues
        return QPixmap.fromImage(image.copy())

    except Exception:
        return QPixmap()


def get_alignment_info(
    lwir_matrices: Optional[Dict[str, Any]],
    visible_matrices: Optional[Dict[str, Any]],
    extrinsic: Optional[Dict[str, Any]],
    image_size: Tuple[int, int],
) -> Dict[str, Any]:
    """Get alignment information for the current calibration.

    Args:
        lwir_matrices: LWIR camera intrinsic calibration
        visible_matrices: Visible camera intrinsic calibration
        extrinsic: Extrinsic calibration between cameras
        image_size: Image size (width, height) - fallback for display

    Returns:
        Dict with alignment info (fov, homography, overlap, source_channel, calib sizes)
    """
    result: Dict[str, Any] = {
        "available": False,
        "source_channel": None,  # Channel to transform (larger FOV)
        "target_channel": None,  # Reference channel (smaller FOV)
        "fov_lwir": (0.0, 0.0),
        "fov_visible": (0.0, 0.0),
        "homography": None,
        "overlap": None,
        "lwir_calib_size": None,
        "vis_calib_size": None,
    }

    if not lwir_matrices or not visible_matrices or not extrinsic:
        return result

    # Extract calibration image sizes
    lwir_size = lwir_matrices.get("image_size")
    if isinstance(lwir_size, list) and len(lwir_size) >= 2:
        result["lwir_calib_size"] = (int(lwir_size[0]), int(lwir_size[1]))
    else:
        result["lwir_calib_size"] = image_size

    vis_size = visible_matrices.get("image_size")
    if isinstance(vis_size, list) and len(vis_size) >= 2:
        result["vis_calib_size"] = (int(vis_size[0]), int(vis_size[1]))
    else:
        result["vis_calib_size"] = image_size

    # Calculate FOV for both cameras (uses image_size from calibration if available)
    fov_lwir = calculate_fov(lwir_matrices, image_size)
    fov_visible = calculate_fov(visible_matrices, image_size)

    log_debug(f"FOV: LWIR={fov_lwir[0]:.1f}°x{fov_lwir[1]:.1f}°, Visible={fov_visible[0]:.1f}°x{fov_visible[1]:.1f}°", "ALIGN")

    result["fov_lwir"] = fov_lwir
    result["fov_visible"] = fov_visible

    # Determine which has larger FOV (will be transformed to match smaller)
    lwir_area = fov_lwir[0] * fov_lwir[1]
    visible_area = fov_visible[0] * fov_visible[1]

    if lwir_area <= 0 or visible_area <= 0:
        return result

    # Source = larger FOV camera, Target = smaller FOV camera
    source_is_lwir: bool
    if lwir_area > visible_area:
        result["source_channel"] = "lwir"
        result["target_channel"] = "visible"
        source_matrix = lwir_matrices
        target_matrix = visible_matrices
        source_is_lwir = True
    else:
        result["source_channel"] = "visible"
        result["target_channel"] = "lwir"
        source_matrix = visible_matrices
        target_matrix = lwir_matrices
        source_is_lwir = False

    # Get rotation and translation from extrinsic
    rotation = extrinsic.get("R") or extrinsic.get("rotation")
    translation = extrinsic.get("T") or extrinsic.get("translation")

    if not rotation or not translation:
        return result

    # Compute homography (pass source_is_lwir for correct R,T direction)
    homography = compute_alignment_homography(
        source_matrix,
        target_matrix,
        rotation if isinstance(rotation, dict) else {"data": rotation},
        translation if isinstance(translation, dict) else {"data": translation},
        image_size,
        source_is_lwir=source_is_lwir,
    )

    if homography is not None:
        result["homography"] = homography
        result["overlap"] = compute_overlap_region(image_size, image_size, homography)
        result["available"] = True

    return result
