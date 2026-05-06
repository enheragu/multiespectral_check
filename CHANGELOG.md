# Changelog

All notable changes to this project will be documented in this file.

The current runtime version is defined in [src/config.py](src/config.py).
Each release below is pinned to the commit that best represents that snapshot.

Versioning note: minor releases mark visible feature jumps; patch releases cover bugfixes and maintenance updates, including docs/help polish that does not change behavior.

## [Unreleased]
- Centralized app metadata in [src/config.py](src/config.py): app name, runtime version, repository URL, issues URL, description, and support email.
- Made the About dialog read version and contact data from configuration instead of shelling out to Git.
- Added a richer project changelog and linked it from [README.md](README.md).
- Added a stereo-alignment utility and parallax controls (`src/backend/utils/stereo_alignment.py`) and viewer wiring for parallax alignment experiments.
- Early exporter UI and backend scaffolding added (`src/backend/services/export/*`, `src/frontend/widgets/export_dialog.py`) to run dataset exports with transforms.
- Grounding DINO integration continued: detector factory, ensemble detector and Grounding DINO backend (`src/backend/services/labels/detection/grounding_dino.py`).
- Several UI/help-message updates and documentation screenshots (`src/frontend/widgets/help_dialog.py`, `src/frontend/utils/ui_messages.py`, `docs/GUI_FUNCTIONALITIES.md`).
- New application icons and resource files added under `src/frontend/resources/media/` (multiple sizes) and a resource package skeleton.
- Ongoing configuration and label-schema tweaks in `config/labels_multiespectral_dataset.yaml` and `src/config.py`.
- Misc unstaged edits to `src/frontend/image_viewer.py`, `src/frontend/ui_mainwindow.py`, and stereo/export plumbing.

## [0.8.1](https://github.com/enheragu/multiespectral_check/commit/ff734e1) - 2026-04-23

- Polished the parallax-correction UI and help text after the 0.8.0 alignment work.
- Fixed a filename-extension casing bug in the image loader (upper/lowercase mismatch).
- Minor adjustments to overlay and viewer wiring to keep export/viewer wording consistent with the new alignment flow.
- Dataset loader: resolve images by stem and accept extension case variants (e.g. .jpg/.JPG) to avoid missing files due to casing.
- Session robustness: added warnings and a decode-check when image decoding fails to avoid silent errors during dataset processing.
- Overlay orchestration: added `apply_parallax_correction` toggle and `effective_parallax` logic so the applied parallax can be disabled without removing calibration reprojection; caching keys updated to use effective parallax.
- Viewer integration: new `Apply Parallax Correction` action, persisted preference and a handler to toggle parallax at runtime (persists via `session.cache_service` and reloads the current image pair).
- UI/help: extended help text to explain the distinction between calibration reprojection and additive parallax correction, and added the menu action under stereo/alignment.

## [0.8.0](https://github.com/enheragu/multiespectral_check/commit/00c8dc0) - 2026-04-23

- Added parallax alignment based on a fixed distance, including a new stereo alignment utility (`src/backend/utils/stereo_alignment.py`) and viewer controls to configure parallax.
- Updated the parallax-correction UI and help text so the export and viewer flows describe the new behavior clearly.
- Fixed image-extension handling so upper-case and lower-case filenames are treated consistently.
- Extended workspace/config persistence so stereo-alignment and parallax settings are stored in workspace state and reflected in the viewer.
- Updated related viewer wiring and docs/screenshots to show the new alignment controls.
- New stereo-alignment utility: `src/backend/utils/stereo_alignment.py` implements fixed-distance parallax correction and exposes transform helpers used by viewer/export flows.
- Viewer controls: added parallax amount controls and a toggle to enable/disable parallax correction in the viewer UI; these controls persist in workspace config.
- Export integration: export pipeline aware of parallax/undistort options so exports can include aligned images with optional parallax correction.
- Persistence: stereo-alignment parameters and parallax-enabled state are saved in the workspace configuration and restored on load.
- Bugfix: image loader now normalizes filename extensions to avoid upper/lowercase mismatches across platforms.

## [0.7.1](https://github.com/enheragu/multiespectral_check/commit/b1f23e7) - 2026-03-10

- Updated README and the label configuration schema (`config/labels_multiespectral_dataset.yaml`) after dataset-driven validation.
- Minor docs and configuration polish; no behaviour-changing code beyond configuration adjustments.

## [0.7.0](https://github.com/enheragu/multiespectral_check/commit/f7edcb8) - 2026-03-10

- Added a label summary and a full label report dialog (frontend) with a backend summary derivation pipeline.
- Improved calibration reporting with chessboard visualization and reprojection-error plots in the calibration report.
- Refined autolabelling behavior with configurable per-class minimums and tighter default confidence handling.
- Updated GUI imagery and documentation screenshots; added supporting docs entries for the labelling flow.
- Label reporting: new `label_report_dialog` UI plus backend `label_summary_derivation` pipeline to aggregate label counts, per-class stats, and export-ready summaries.
- Autolabelling improvements: per-class minimum confidence thresholds added to the detection pipeline and default thresholds tightened to reduce false positives during batch inference.
- Calibration reporting: added chessboard visualization and reprojection-error plotting in calibration reports to help QA calibration quality.
- UI/Docs: added labelling-focused screenshots and doc entries in `docs/GUI_FUNCTIONALITIES.md`; updated help text in `src/frontend/widgets/help_dialog.py`.
- Stability: small fixes to workspace inspector and label storage to make summary derivation robust for larger datasets.

## [0.6.0](https://github.com/enheragu/multiespectral_check/commit/278d0e9) - 2026-03-05

- Added Grounding DINO detector implementation and integration (`src/backend/services/labels/detection/grounding_dino.py`).
- Introduced a detector factory and an ensemble detector to support multiple detection backends.
- Updated the label service and related plumbing to consume detection outputs and support batch inference.
- Fixed early issues found during external testing and adjusted a few UI label-edit touches.
- Grounding DINO backend: added `grounding_dino.py` implementation and wired it into the detection subsystem to enable large-model zero-shot detections.
- Detector factory: new `detector_factory.py` and `ensemble_detector.py` to allow selecting among detectors at runtime and combining outputs from multiple detectors.
- Label service updates: `label_service` modifications to accept detection outputs, normalize bounding boxes, and insert batch results into label storage for review.
- Batch inference paths: added plumbing for running detection over a workspace in batches and storing progress/results via the UI progress queue.
- UX touches: small adjustments in the label editor and marking controller to accommodate detection-assisted labelling flows.

## [0.5.1](https://github.com/enheragu/multiespectral_check/commit/af96769) - 2026-03-04

- Dataset session: guard and normalize loaded overrides by converting list inputs to `set` during load to avoid type-mismatch edge cases.
- Configuration: simplified `default_dataset_dir` to an empty path default to avoid embedding developer-specific paths in new installs.
- Label editing: added robust class-selection validation in the label edit dialog (`_validate_class_selection`, `_resolve_class_id`) and prevent accepting invalid classes; improved UX by enabling/disabling OK accordingly.
- Viewer: removed fragile sweep-flag sync paths and improved missing-pair handling; add runtime validation when adding labels so users must pick a valid class.
- Workspace selection: detect when the chosen workspace root is actually a dataset (contains `lwir/` and `visible/`) and show a helpful informational dialog guiding the user to select the correct parent folder.
- Misc: collected early bugfixes reported by external testers and tightened dialogs/workflow for more predictable behavior on varied datasets.

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
- Refactored marks data format in cache service: unified manual and auto-marks into a single format (`{base: {reason, auto}}`) to simplify storage and reduce redundancy.
- Cache consistency improvements: simplified outlier/augmentation tracking and fixed edge cases in mark normalization and override conversion.
- Calibration workflow refinements: improved intrinsic/extrinsic solver robustness and added better logging for convergence/failure cases.
- Statistics and summary derivation: optimized stats computation and fixed display issues in stats panel; improved robustness for edge-case datasets.
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
- Confirmed the core dataset review path and navigation were stable enough to continue evolving from a single surface.

## [0.2.0](https://github.com/enheragu/multiespectral_check/commit/0e70f80) - 2025-12-15

- Introduced the labelling toolchain: `label_workflow.py` and `labeling_controller.py` for annotation workflows, plus `config/labels_coco.yaml` schema for label configuration.
- Refactored code into a `services/` package structure: moved calibration, overlays, and signatures into separate modules for better maintainability.
- Added marking and progress management: `marking_controller.py` for mark operations and `ui/progress_queue.py` for async task progress reporting.
- Overlay improvements: new overlay prefetcher and workflow modules for efficient overlay caching and rendering.
- Signature scanning: added `signature_scan_manager.py` to support signature-based dataset scanning and organization.
- Image viewer refactor: streamlined viewer logic (~1172 lines) by extracting workflows into dedicated services; improved separation of concerns.
- Documentation: added `TESTS.md` with testing procedures and updated README with setup/usage guidance.
- UI utilities: added progress and state helpers to centralize common UI operations.

## [0.1.1](https://github.com/enheragu/multiespectral_check/commit/7c6cfb2) - 2025-12-15

- Visual polish: enhanced `src/widgets/style.py` with improved colors, spacing, and theming for a more cohesive look.
- Progress panel: refined progress panel rendering and status display for better clarity during long operations.
- Calibration dialogs: minor improvements to calibration check and outliers dialogs for better UX.
- Documentation: cleaned up and trimmed README for clarity.
- Widget layout touches: small tweaks to spacing and sizing in viewer and dialogs for better visual balance.

## [0.1.0](https://github.com/enheragu/multiespectral_check/commit/99b5cff) - 2025-12-15

- Core image viewer: `src/image_viewer.py` (1740 lines) with multispectral image display, navigation, and overlay support.
- Dataset loading and management: `dataset_loader.py` for discovering and loading multispectral dataset structures; `dataset_session.py` for session state.
- Calibration toolchain: integrated calibration controller, solvers (intrinsic/extrinsic), and refinement tools; chessboard detection and reprojection.
- Cache system: `cache_service.py` (623 lines) for workspace and session caching; cache writers for persistence.
- Overlay system: overlay orchestration, prefetching, and workflow support for alignment visualization.
- Utility modules: calibration helpers (629 lines), duplicate detection, overlay math, filter modes, and progress tracking.
- UI framework: `ui_mainwindow.py` with menu/toolbar; dialog infrastructure (calibration check, outliers, help); panels for stats and progress.
- Styling and widgets: unified theming, zoom/pan controls, and responsive layout framework.
- Documentation: README with project overview and requirements tracking.

