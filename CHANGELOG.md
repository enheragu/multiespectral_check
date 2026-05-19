# Changelog

All notable changes to this project will be documented in this file.

The current runtime version is defined in [src/config.py](src/config.py).
Each release below is pinned to the commit that best represents that snapshot.

Versioning note: minor releases mark visible feature jumps; patch releases cover bugfixes and maintenance updates, including docs/help polish that does not change behavior.

## [Unreleased]
- Bugfix: prev and next boton in viewer were not working.
- Store GUI version in generated files for an easier tracking of issues.

## [0.9.1](https://github.com/enheragu/multiespectral_check/commit/18091fd) - 2026-05-13

- View › Labels display submenu: new toggle to show/hide projected (cross-channel) labels independently of the base labels toggle; preference persists across sessions.
- Calibration report updates: distortion map charts (barrel/pincushion per channel), FOV and focal length per channel, chessboard coverage visible when using workspace default calibration.
- Bugfix: stereo alignment FOV projection (all three modes) was feeding undistorted LWIR crop bounds to the homography instead of raw calibration corners, causing the FOV Focus box to appear ~4% too small per side; corrected draw order so the FOV Focus rectangle is always visible.
- Bugfix: autosave triggered correctly when accepting a label from the automated labelling toolchain.

## [0.9.0](https://github.com/enheragu/multiespectral_check/commit/89600bd) - 2026-05-06

- Dataset export system: new backend pipeline (`src/backend/services/export/`) with image processing (undistort, parallax correction, optional downscaling) and label export (COCO/YOLO converters).
- Export dialog (`src/frontend/widgets/export_dialog.py`): workspace tree selector, per-dataset toggles, progress bar, and background worker thread so the UI stays responsive during long exports.
- Stereo alignment snapshot (`src/backend/services/stereo_alignment_export.py`): writes `.stereo_alignment.yaml` at dataset level whenever calibration or parallax changes, so external tools can replay the same alignment without the GUI.
- About dialog (`AboutDialog`): dedicated popup showing app name, version, description, repository, and issue tracker links; reads all metadata from `config.py`. Banner image added to the help dialog header.
- Changed default auto-parallax reference depth from 30 m to 10 m.
- Minor fixes in Grounding DINO detector and stereo alignment utility.

## [0.8.2](https://github.com/enheragu/multiespectral_check/commit/cfbb0cc) - 2026-04-24

- Centralized app metadata in [src/config.py](src/config.py): `APP_NAME`, `APP_VERSION`, `SUPPORT_EMAIL`, `REPO_URL`, `ISSUES_URL`, `APP_DESCRIPTION`.
- Help and About dialogs now read version and contact data from `config.py` instead of hardcoded strings.
- Added application logo and banner assets under `src/frontend/resources/media/` (multiple sizes + `.ico`).
- Updated README with logo, banner, and a link to this changelog.

## [0.8.1](https://github.com/enheragu/multiespectral_check/commit/ff734e1) - 2026-04-23

- Fixed filename-extension casing in the image loader: images are resolved by stem so upper/lower-case variants (`.jpg`/`.JPG`) are treated consistently.
- Added `apply_parallax_correction` toggle and `effective_parallax` logic: correction can be disabled at runtime without removing calibration reprojection; the viewer persists the preference and reloads on change.
- Session robustness: decode-check and warning when image decoding fails to avoid silent errors during dataset processing.
- Extended help text and added a menu action to clarify the distinction between calibration reprojection and additive parallax correction.

## [0.8.0](https://github.com/enheragu/multiespectral_check/commit/00c8dc0) - 2026-04-23

- New stereo-alignment utility (`src/backend/utils/stereo_alignment.py`): computes fixed-distance parallax correction and exposes transform helpers consumed by viewer and export flows.
- Viewer controls for parallax: amount sliders, enable/disable toggle, and auto-parallax estimation; all settings persist in the workspace config.
- Export pipeline updated to apply parallax and undistort corrections so exported datasets reflect the active alignment.
- Updated UI, help text, and docs/screenshots to document the new alignment controls.
- Bugfix: image loader normalizes filename extensions (upper/lowercase); casing handling continued and finalized in 0.8.1.

## [0.7.1](https://github.com/enheragu/multiespectral_check/commit/b1f23e7) - 2026-03-10

- Updated README and the label configuration schema (`config/labels_multiespectral_dataset.yaml`) after dataset-driven validation.
- Minor docs and configuration polish; no behaviour-changing code beyond configuration adjustments.

## [0.7.0](https://github.com/enheragu/multiespectral_check/commit/f7edcb8) - 2026-03-10

- Label report dialog and backend `label_summary_derivation` pipeline: aggregates label counts, per-class stats, and export-ready summaries across the workspace.
- Per-class minimum confidence thresholds in the autolabelling pipeline; default thresholds tightened to reduce false positives during batch inference.
- Calibration reporting: chessboard visualization and reprojection-error plots added to help QA calibration quality.
- UI/Docs: labelling-focused screenshots and entries added to `docs/GUI_FUNCTIONALITIES.md`; help text updated.
- Stability: small fixes to workspace inspector and label storage for robustness on larger datasets.

## [0.6.1](https://github.com/enheragu/multiespectral_check/commit/e3124b1) - 2026-03-09

- Added x5 fast-forward and fast-reverse navigation buttons to the image viewer (jump 5 images at a time).
- Added a "Go to index" action so users can jump directly to a specific image pair by number.

## [0.6.0](https://github.com/enheragu/multiespectral_check/commit/278d0e9) - 2026-03-05

- Grounding DINO detector backend (`src/backend/services/labels/detection/grounding_dino.py`) for zero-shot large-model detections.
- Detector factory (`detector_factory.py`) and ensemble detector (`ensemble_detector.py`) to select and combine multiple detection backends at runtime.
- Label service updated to consume detection outputs: normalizes bounding boxes and inserts batch results into label storage for review.
- Batch inference path: run detection over a full workspace in batches with progress reporting via the UI progress queue.
- Small UX fixes in the label editor and marking controller to accommodate detection-assisted labelling flows; fixed early issues reported during external testing.

## [0.5.1](https://github.com/enheragu/multiespectral_check/commit/af96769) - 2026-03-04

- Dataset session: normalized loaded override types to avoid edge cases on workspace files written by older versions.
- Configuration: `default_dataset_dir` now defaults to an empty path to avoid embedding developer-specific paths in new installs.
- Label editing: class-selection validation in the label edit dialog; OK is enabled only when a valid class is selected.
- Viewer: removed fragile sweep-flag sync paths and improved missing-pair handling; runtime validation ensures users must pick a valid class when adding labels.
- Workspace selection: detect when the chosen root is actually a dataset directory (`lwir/` + `visible/`) and prompt the user to select the correct parent folder.
- Collected early bugfixes reported by external testers; tightened dialogs for more predictable behavior on varied datasets.

## [0.5.0](https://github.com/enheragu/multiespectral_check/commit/899436e) - 2026-02-13

Released with updated documentation and in-app help for external users:
- Expanded the help dialogs and user-facing messages in the app to match the new docs.
- Added the main project documentation set, including design philosophy and GUI functionality docs.
- Updated the README with installation and environment guidance.
- Unified the language across documentation.

## [0.4.0](https://github.com/enheragu/multiespectral_check/commit/d1bb725) - 2026-02-06

- Major refactor of label subsystem: centralized label types, storage, and service logic.
- New label service architecture: `label_service.py` (unified labelling API), `label_storage.py` (persistence), `label_types.py` (shared data structures).
- Bounding-box transform utilities: `bbox_transform.py` for format conversions and geometric operations.
- Format converters: added COCO and YOLO converters to support dataset export and import pipelines.
- Detection backends: added YOLOv8 detection integration (`src/backend/services/labels/detection/yolov8.py`) and detection factory infrastructure for extensibility.
- Label editing UI: new `label_edit_dialog.py` with rich attribute editing (class, occlusion, truncation, bbox, class-specific fields) and validation against label config.
- Image viewer integration: expanded viewer with labelling workflow support, annotation canvas, and label-list synchronization.
- Statistics and workspace improvements: refined stats panel and workspace inspector to correctly track and display label counts and validation state.
- Configuration: added `config/labels_multiespectral_dataset.yaml` defining label classes, attributes, and defaults for projects.

## [0.3.1](https://github.com/enheragu/multiespectral_check/commit/d055740) - 2026-01-20

- Consolidated YAML I/O utilities into `src/common/yaml_utils.py` to reduce boilerplate across the codebase.
- Refactored marks data format in cache service: unified manual and auto-marks into a single format to simplify storage and reduce redundancy.
- Cache consistency: simplified outlier/augmentation tracking and fixed edge cases in mark normalization and override conversion.
- Calibration workflow: improved intrinsic/extrinsic solver robustness and added logging for convergence and failure cases.
- Statistics and summary derivation: optimized stats computation and fixed display issues; improved robustness for edge-case datasets.
- View state controller: introduced new controller to centralize viewer state and cache-preference handling.
- Image viewer: stabilized annotation workflows and viewer/overlay synchronization; removed debug-specific plumbing.
- GUI polish: minor layout fixes, improved help text consistency, and cleaned up debug scripts.

## [0.3.0](https://github.com/enheragu/multiespectral_check/commit/bfbc869) - 2026-01-14

- Consolidated the separate tool areas into a single, integrated review surface (viewer + tool panels), simplifying navigation and session flow.
- Moved and reorganized core services into a `backend` package: dataset loading, session handling, workspace scanning, and workspace inspector.
- Added and refactored calibration subsystems (`calibration_controller`, `calibration_workflow`, `calibration_mixin`) to centralize intrinsic/extrinsic tooling and reporting.
- Introduced overlay orchestration and matching-FOV plumbing to support alignment previews and overlay workflows used by review and export.
- Integrated detection and filtering hooks into the GUI: `filter_controller`, `filter_workflow_mixin` and pattern-sweep support for batch operations.
- Improved workspace/dataset management with `dataset_handler`, `workspace_manager` and stabilized session/state handling used by the viewer.

## [0.2.0](https://github.com/enheragu/multiespectral_check/commit/0e70f80) - 2025-12-15

- Introduced the labelling toolchain: `label_workflow.py` and `labeling_controller.py` for annotation workflows, plus `config/labels_coco.yaml` schema for label configuration.
- Refactored code into a `services/` package structure: moved calibration, overlays, and signatures into separate modules for better maintainability.
- Added marking and progress management: `marking_controller.py` for mark operations and `ui/progress_queue.py` for async task progress reporting.
- Overlay improvements: new overlay prefetcher and workflow modules for efficient overlay caching and rendering.
- Signature scanning: added `signature_scan_manager.py` to support signature-based dataset scanning and organization.
- Image viewer refactor: streamlined viewer logic by extracting workflows into dedicated services; improved separation of concerns.
- Documentation: added `TESTS.md` with testing procedures and updated README with setup/usage guidance.

## [0.1.1](https://github.com/enheragu/multiespectral_check/commit/7c6cfb2) - 2025-12-15

- Visual polish: enhanced `src/widgets/style.py` with improved colors, spacing, and theming for a more cohesive look.
- Progress panel: refined progress panel rendering and status display for better clarity during long operations.
- Calibration dialogs: minor improvements to calibration check and outliers dialogs for better UX.
- Documentation: cleaned up and trimmed README for clarity.

## [0.1.0](https://github.com/enheragu/multiespectral_check/commit/99b5cff) - 2025-12-15

- Core image viewer with multispectral image display, navigation, and overlay support.
- Dataset loading and management: `dataset_loader.py` for discovering and loading multispectral dataset structures; `dataset_session.py` for session state.
- Calibration toolchain: integrated calibration controller, solvers (intrinsic/extrinsic), and refinement tools; chessboard detection and reprojection.
- Cache system: workspace and session caching with persistent writers.
- Overlay system: overlay orchestration, prefetching, and workflow support for alignment visualization.
- Utility modules: calibration helpers, duplicate detection, overlay math, filter modes, and progress tracking.
- UI framework: `ui_mainwindow.py` with menu/toolbar; dialog infrastructure (calibration check, outliers, help); panels for stats and progress.
- Styling and widgets: unified theming, zoom/pan controls, and responsive layout framework.
- Documentation: README with project overview and requirements tracking.
