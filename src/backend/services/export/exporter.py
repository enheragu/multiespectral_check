"""Dataset export orchestrator.

Walks selected datasets, applies the image and label pipelines, and
writes the result to a target directory mirroring the workspace layout.
A ``.export_info.yaml`` with full provenance (parameters, calibration,
homography, parallax) is written at the root of the export.

The orchestrator is GUI-agnostic: progress is reported via a callback
and cancellation is checked via another callback, so it can be driven
from a worker thread or a script.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore

from backend.dataset_loader import DatasetLoader
from backend.services.cache_service import DATASET_CACHE_FILENAME
from backend.services.export.image_pipeline import (
    RESOLUTION_DOWNSAMPLE,
    RESOLUTION_UPSAMPLE,
    AlignedPair,
    CalibrationBundle,
    TransformParams,
    compute_export_homography,
    transform_aligned_pair,
    transform_image_single,
)
from backend.services.export.label_pipeline import build_exported_labels
from backend.services.labels.label_storage import LabelStorage
from backend.services.workspace_config import get_workspace_config_service
from backend.utils.stereo_alignment import compute_auto_parallax
from common.log_utils import log_debug, log_info, log_warning
from common.yaml_utils import get_timestamp_fields, load_yaml, save_yaml
from config import get_config

from dataclasses import replace


ProgressCB = Callable[[str, int, int], None]   # message, current, total
CancelCB = Callable[[], bool]                  # returns True if user cancelled


@dataclass
class DatasetExportPlan:
    """One dataset to export; its path and its position inside the workspace tree."""
    dataset_path: Path
    relative_path: Path  # Path under the export root (e.g. "set_a/25-12-17_12-52")


@dataclass
class ExportRequest:
    """Full configuration for an export run."""
    workspace_path: Path
    output_dir: Path                           # root for the export
    datasets: List[DatasetExportPlan]
    channels: Tuple[str, ...]                  # subset of ("lwir", "visible")
    params: TransformParams = field(default_factory=TransformParams)


@dataclass
class DatasetReport:
    """Per-dataset stats included in the .export_info.yaml."""
    relative_path: str
    images_lwir: int = 0
    images_visible: int = 0
    labels_total: int = 0
    skipped_no_calibration: bool = False
    skipped_marked_for_deletion: int = 0
    error: Optional[str] = None


@dataclass
class ExportResult:
    """Aggregate result of an export run."""
    output_root: Path
    datasets: List[DatasetReport] = field(default_factory=list)
    total_images: int = 0
    total_labels: int = 0
    cancelled: bool = False


def run_export(
    request: ExportRequest,
    *,
    progress: Optional[ProgressCB] = None,
    cancelled: Optional[CancelCB] = None,
) -> ExportResult:
    """Execute an export request. Synchronous; suitable for a worker thread."""
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV/NumPy not available — cannot run export")

    output_root = _prepare_output_root(request.output_dir, request.workspace_path)
    result = ExportResult(output_root=output_root)

    total_datasets = len(request.datasets)
    total_image_units = _count_total_image_units(request)

    images_done = 0
    requires_alignment = (
        len(request.channels) == 2 and request.params.align_fov
    )

    # Track the calibration actually used (for export_info provenance).
    used_calibrations: List[Tuple[Path, CalibrationBundle, Optional[Any]]] = []

    for ds_idx, plan in enumerate(request.datasets):
        if cancelled and cancelled():
            result.cancelled = True
            break

        report = DatasetReport(relative_path=str(plan.relative_path))
        result.datasets.append(report)

        if progress:
            progress(
                f"[{ds_idx + 1}/{total_datasets}] {plan.relative_path}",
                images_done,
                total_image_units,
            )

        try:
            calib = _load_calibration_for(plan.dataset_path)
            if requires_alignment and calib is None:
                report.skipped_no_calibration = True
                log_warning(
                    f"Skipping {plan.relative_path}: alignment requested but no calibration",
                    "EXPORT",
                )
                continue

            # Per-dataset parallax: auto-compute from this dataset's own
            # calibration so the export is correct even when the dataset
            # was never opened in the GUI.
            ds_params = _resolve_per_dataset_params(request.params, calib)

            loader = DatasetLoader(str(plan.dataset_path))
            if not loader.load_dataset():
                report.error = "no images found"
                continue

            marked_for_deletion = _load_marked_bases(plan.dataset_path)
            target_dir = output_root / plan.relative_path
            label_storage = LabelStorage(plan.dataset_path)

            # Pre-compute homography for the export_info provenance.
            export_homography = None
            do_align = (
                calib is not None
                and ds_params.align_fov
                and len(request.channels) == 2
            )
            if do_align:
                export_homography = compute_export_homography(
                    calib, ds_params, source_is_lwir=True
                )

            if calib is not None:
                used_calibrations.append((plan.dataset_path, calib, export_homography))

            for base in loader.image_bases:
                if cancelled and cancelled():
                    result.cancelled = True
                    break

                if base in marked_for_deletion:
                    report.skipped_marked_for_deletion += 1
                    continue

                stats = _process_base(
                    base=base,
                    loader=loader,
                    label_storage=label_storage,
                    target_dir=target_dir,
                    request=request,
                    params=ds_params,
                    calib=calib,
                    aligned_export=do_align,
                )
                report.images_lwir += stats.images_lwir
                report.images_visible += stats.images_visible
                report.labels_total += stats.labels

                images_done += stats.images_lwir + stats.images_visible
                if progress and (images_done % 5 == 0 or images_done == total_image_units):
                    progress(
                        f"[{ds_idx + 1}/{total_datasets}] {plan.relative_path}: {base}",
                        images_done,
                        total_image_units,
                    )

        except Exception as e:
            log_warning(f"Export failure on {plan.relative_path}: {e}", "EXPORT")
            report.error = str(e)

    # Aggregate totals.
    result.total_images = sum(r.images_lwir + r.images_visible for r in result.datasets)
    result.total_labels = sum(r.labels_total for r in result.datasets)

    _write_export_info(
        output_root,
        request,
        result,
        used_calibrations,
    )

    if progress:
        progress("Export complete", total_image_units, total_image_units)
    log_info(
        f"Export finished: {result.total_images} images, {result.total_labels} labels "
        f"to {output_root}",
        "EXPORT",
    )
    return result


# ── per-base processing ──────────────────────────────────────────────────


@dataclass
class _PerBaseStats:
    images_lwir: int = 0
    images_visible: int = 0
    labels: int = 0


def _process_base(
    *,
    base: str,
    loader: DatasetLoader,
    label_storage: LabelStorage,
    target_dir: Path,
    request: ExportRequest,
    params: TransformParams,
    calib: Optional[CalibrationBundle],
    aligned_export: bool = False,
) -> _PerBaseStats:
    """Export one image-pair base across the requested channels.

    When ``aligned_export`` is True (both channels + ``align_fov``),
    runs the pair-wise affine crop+resize pipeline so both outputs share
    the same scene area at the same size. Otherwise each requested
    channel is processed independently (undistort only).
    """
    stats = _PerBaseStats()
    both_channels = len(request.channels) == 2

    # Always read both channels' images if available — we need both
    # native sizes to project labels even when only one channel is being
    # exported.
    images: Dict[str, Optional[Any]] = {"lwir": None, "visible": None}
    native_sizes: Dict[str, Optional[Tuple[int, int]]] = {"lwir": None, "visible": None}
    for channel in ("lwir", "visible"):
        path = loader.get_image_path(base, channel)
        if path is None:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            log_warning(f"Failed to read {path}", "EXPORT")
            continue
        images[channel] = img
        native_sizes[channel] = (img.shape[1], img.shape[0])

    # Decide which channels actually get written to disk.
    write_lwir = "lwir" in request.channels
    write_vis = "visible" in request.channels

    # ── Aligned pair path ───────────────────────────────────────────────
    if aligned_export and calib is not None:
        aligned = transform_aligned_pair(
            images.get("lwir"),
            images.get("visible"),
            calib=calib,
            params=params,
            keep_lwir=write_lwir,
            keep_vis=write_vis,
        )
        if aligned is None:
            log_warning(f"[{base}] aligned-pair transform produced no output", "EXPORT")
            return stats

        if write_lwir and aligned.lwir_image is not None:
            _write_image(target_dir, loader, base, "lwir", aligned.lwir_image)
            stats.images_lwir += 1
        if write_vis and aligned.vis_image is not None:
            _write_image(target_dir, loader, base, "visible", aligned.vis_image)
            stats.images_visible += 1

        # Labels: each exported channel uses its own affine crop; the
        # other channel's labels are projected via H to this channel's
        # native frame, then the same affine crop is applied.
        for channel in request.channels:
            own_crop = (
                aligned.lwir_crop_in_native
                if channel == "lwir"
                else aligned.vis_crop_in_native
            )
            other_crop = (
                aligned.vis_crop_in_native
                if channel == "lwir"
                else aligned.lwir_crop_in_native
            )
            stats.labels += _write_labels(
                target_dir=target_dir,
                loader=loader,
                base=base,
                channel=channel,
                params=params,
                calib=calib,
                output_size=aligned.output_size,
                native_sizes=native_sizes,
                label_storage=label_storage,
                both_channels=both_channels,
                own_crop=own_crop,
                other_crop=other_crop,
            )
        return stats

    # ── Per-channel path (single channel, or alignment off) ────────────
    for channel in request.channels:
        img = images.get(channel)
        if img is None:
            continue
        single = transform_image_single(
            img,
            channel=channel,
            calib=calib,
            params=params,
        )
        _write_image(target_dir, loader, base, channel, single.image)
        if channel == "lwir":
            stats.images_lwir += 1
        else:
            stats.images_visible += 1

        stats.labels += _write_labels(
            target_dir=target_dir,
            loader=loader,
            base=base,
            channel=channel,
            params=params,
            calib=calib,
            output_size=single.output_size,
            native_sizes=native_sizes,
            label_storage=label_storage,
            both_channels=both_channels,
            own_crop=None,
            other_crop=None,
        )

    return stats


def _write_image(
    target_dir: Path,
    loader: DatasetLoader,
    base: str,
    channel: str,
    image: Any,
) -> None:
    out_path = target_dir / channel / f"{base}{_extension_for(loader, base, channel)}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), image):
        log_warning(f"cv2.imwrite failed: {out_path}", "EXPORT")


def _write_labels(
    *,
    target_dir: Path,
    loader: DatasetLoader,
    base: str,
    channel: str,
    params: TransformParams,
    calib: Optional[CalibrationBundle],
    output_size: Tuple[int, int],
    native_sizes: Dict[str, Optional[Tuple[int, int]]],
    label_storage: LabelStorage,
    both_channels: bool,
    own_crop: Optional[Tuple[int, int, int, int]],
    other_crop: Optional[Tuple[int, int, int, int]],
) -> int:
    """Build and save the union label YAML for one exported channel.
    Returns the number of annotations written."""
    own = label_storage.load_labels(channel, base)
    other_channel = "visible" if channel == "lwir" else "lwir"
    other = label_storage.load_labels(other_channel, base)

    labels_obj = build_exported_labels(
        target_channel=channel,
        target_image_file=f"{channel}_{base}{_extension_for(loader, base, channel)}",
        target_image_size=output_size,
        own_native_size=native_sizes[channel],
        other_native_size=native_sizes[other_channel],
        own_labels=own,
        other_labels=other,
        calib=calib,
        params=params,
        both_channels_selected=both_channels,
        own_crop_in_native=own_crop,
        other_crop_in_native=other_crop,
    )
    if not labels_obj.annotations:
        return 0
    label_path = target_dir / "labels" / channel / f"{base}.yaml"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    save_yaml(label_path, labels_obj.to_dict(), sort_keys=False)
    return len(labels_obj.annotations)


def _extension_for(loader: DatasetLoader, base: str, channel: str) -> str:
    p = loader.get_image_path(base, channel)
    return p.suffix if p else ".png"


# ── calibration / marks / output prep ────────────────────────────────────


def _resolve_per_dataset_params(
    base_params: TransformParams,
    calib: Optional[CalibrationBundle],
) -> TransformParams:
    """Return a per-dataset copy of ``base_params`` with parallax filled in.

    The dialog only carries the user's intent (parallax checkbox on/off);
    the actual pixel shift depends on each dataset's calibration. This
    auto-computes it from K_vis and T at the configured default depth,
    so the export is correct without the user having to open the dataset.
    """
    if not base_params.parallax or calib is None:
        return base_params

    cfg = get_config()
    square_size = calib.square_size_mm or cfg.chessboard_square_size_mm
    depth_m = cfg.default_parallax_depth_m

    translation = calib.extrinsic.get("translation") or calib.extrinsic.get("T")
    if translation is None or not square_size:
        return base_params

    h, v = compute_auto_parallax(
        target_matrix=calib.vis_matrix,
        translation=translation,
        square_size_mm=square_size,
        depth_m=depth_m,
    )
    return replace(base_params, parallax_h=round(h, 1), parallax_v=round(v, 1))


def _load_calibration_for(dataset_path: Path) -> Optional[CalibrationBundle]:
    """Resolve calibration: dataset's own files first, then workspace default."""
    cfg = get_config()
    intrinsic_path = dataset_path / cfg.calibration_intrinsic_filename
    extrinsic_path = dataset_path / cfg.calibration_extrinsic_filename

    ws = get_workspace_config_service()
    if not intrinsic_path.exists():
        default = ws.get_default_calibration()
        if default and default.intrinsic_path and default.intrinsic_path.exists():
            intrinsic_path = default.intrinsic_path
    if not extrinsic_path.exists():
        default = ws.get_default_calibration()
        if default and default.extrinsic_path and default.extrinsic_path.exists():
            extrinsic_path = default.extrinsic_path

    if not intrinsic_path.exists() or not extrinsic_path.exists():
        return None

    try:
        intrinsic_data = load_yaml(intrinsic_path)
        extrinsic_data = load_yaml(extrinsic_path)
    except Exception as e:
        log_warning(f"Failed to load calibration for {dataset_path}: {e}", "EXPORT")
        return None

    channels = intrinsic_data.get("channels", {})
    lwir_mat = channels.get("lwir")
    vis_mat = channels.get("visible")
    if not isinstance(lwir_mat, dict) or not isinstance(vis_mat, dict):
        return None

    sq = intrinsic_data.get("square_size") or intrinsic_data.get("square_length")
    if sq is None:
        cfg_calib = ws.config.calibration
        sq = cfg_calib.square_size_mm

    return CalibrationBundle(
        lwir_matrix=lwir_mat,
        vis_matrix=vis_mat,
        extrinsic=extrinsic_data,
        square_size_mm=float(sq) if sq else None,
    )


def _load_marked_bases(dataset_path: Path) -> set:
    """Return set of bases marked for deletion in .image_labels.yaml."""
    path = dataset_path / DATASET_CACHE_FILENAME
    if not path.exists():
        return set()
    try:
        data = load_yaml(path)
    except Exception:
        return set()
    marks = data.get("marks", {})
    if not isinstance(marks, dict):
        return set()
    return set(str(k) for k in marks.keys())


def _prepare_output_root(output_dir: Path, workspace_path: Path) -> Path:
    """Create the root directory for the export under ``output_dir``."""
    name = f"{workspace_path.name}_export" if workspace_path.name else "export"
    target = output_dir / name
    target.mkdir(parents=True, exist_ok=True)
    return target


def _count_total_image_units(request: ExportRequest) -> int:
    """Approximate total image-write units for progress (for the bar denominator)."""
    total = 0
    for plan in request.datasets:
        loader = DatasetLoader(str(plan.dataset_path))
        if loader.load_dataset():
            marked = _load_marked_bases(plan.dataset_path)
            n = sum(1 for b in loader.image_bases if b not in marked)
            total += n * len(request.channels)
    return max(total, 1)


# ── export_info.yaml ─────────────────────────────────────────────────────


def _write_export_info(
    output_root: Path,
    request: ExportRequest,
    result: ExportResult,
    used_calibrations: List[Tuple[Path, CalibrationBundle, Optional[Any]]],
) -> None:
    """Write provenance + per-dataset stats + calibration to ``.export_info.yaml``."""
    info: Dict[str, Any] = {
        "version": 1,
        **get_timestamp_fields(),
        "tool": "multiespectral_check",
        "source_workspace": str(request.workspace_path),
        "parameters": {
            "channels": list(request.channels),
            "transforms": {
                "undistort": bool(request.params.undistort),
                "fov_alignment": bool(request.params.align_fov),
                "parallax": bool(request.params.parallax),
            },
            "resolution_mode": request.params.resolution_mode,
            "parallax_h_px": float(request.params.parallax_h),
            "parallax_v_px": float(request.params.parallax_v),
            "label_sources": ["manual", "reviewed"],
        },
        "totals": {
            "datasets": len(result.datasets),
            "images": result.total_images,
            "labels": result.total_labels,
        },
        "datasets": [
            {
                "path": r.relative_path,
                "images_lwir": r.images_lwir,
                "images_visible": r.images_visible,
                "labels_total": r.labels_total,
                "skipped_no_calibration": r.skipped_no_calibration,
                "skipped_marked_for_deletion": r.skipped_marked_for_deletion,
                "error": r.error,
            }
            for r in result.datasets
        ],
        "cancelled": bool(result.cancelled),
    }

    if used_calibrations:
        # Use the first dataset's calibration as the representative one.
        # (In practice a workspace default is shared across all of them.)
        ds_path, calib, H = used_calibrations[0]
        info["calibration"] = {
            "from_dataset": str(ds_path.name),
            "square_size_mm": calib.square_size_mm,
            "lwir": _serialize_matrix(calib.lwir_matrix),
            "visible": _serialize_matrix(calib.vis_matrix),
            "extrinsic": _serialize_extrinsic(calib.extrinsic),
        }
        if H is not None:
            info["calibration"]["homography_export"] = [
                [float(H[i, j]) for j in range(3)] for i in range(3)
            ]

    save_yaml(output_root / ".export_info.yaml", info, sort_keys=False)
    log_debug(f"Wrote export_info to {output_root}/.export_info.yaml", "EXPORT")


def _serialize_matrix(matrix: Dict[str, Any]) -> Dict[str, Any]:
    """Slim, YAML-friendly view of a calibration matrix."""
    out: Dict[str, Any] = {}
    if "image_size" in matrix:
        out["image_size"] = list(matrix["image_size"])
    K = matrix.get("camera_matrix")
    if K is not None:
        out["camera_matrix"] = K
    D = matrix.get("distortion")
    if D is not None:
        out["distortion"] = D
    return out


def _serialize_extrinsic(ext: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    R = ext.get("rotation") or ext.get("R")
    T = ext.get("translation") or ext.get("T")
    if R is not None:
        out["rotation"] = R
    if T is not None:
        out["translation"] = T
    return out


__all__ = [
    "DatasetExportPlan",
    "ExportRequest",
    "ExportResult",
    "DatasetReport",
    "run_export",
]
