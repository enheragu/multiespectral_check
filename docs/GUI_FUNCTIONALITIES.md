# GUI Functionalities - Multiespectral Dataset Checker

GUI to handle multiespectral image datasets. Each dataset is composed of a workspace with a set of collections (groups of datasets) or single standalone datasets.

## Workspace Structure

```
workspace/
├── collection1/          # Groups related datasets
│   ├── dataset_a/       # Contains lwir/ and visible/
│   │   ├── lwir/
│   │   │   ├── *.png (images)
│   │   │   └── *.yaml (metadata)
│   │   └── visible/ (same structure as lwir)
│   └── dataset_b/
└── standalone_dataset/   # Dataset without collection
```

Standalone datasets are treated as collections (with a single dataset) for consistency. The workspace itself is never considered a collection and will never be loaded in edit mode in the second tab of the GUI.

## Main Window Layout

The GUI includes two tabs. Both have a title at the top-left and action buttons at the top-right. Below is the main view area. A status bar at the bottom shows progress (bar + cancel button, right side) and status messages (left side).

## Workspace Tab

Shows a table with a summary of all collections/datasets:
- Number of image pairs
- Number of deleted image pairs
- Number of images tagged to be deleted (by reason)
- Number of labels (total annotations, derived from disk on first scan and cached)
- Calibration candidates (disaggregated: both chessboards found, only one found, none found, discarded as outlier)
- Sweep status (which automated sweeps have been performed)

The table should update when changes are made (tags added/removed, images deleted/restored, calibration candidates marked/unmarked, sweeps executed, etc).
Under the table, a text box allows reading/writing notes for each collection or dataset.

Title format: "Workspace: {workspace path}"

Buttons: Open selected collection/dataset

## Dataset/Collection Viewer Tab

Allows inspecting the loaded dataset or collection.

Title format: "Collection: {name}" or "Dataset: {name}"

### Image Display

Both images (LWIR and Visible) appear side by side, each expanded to 50% of width independently of resolution (can differ between images).

**Stereo Alignment Modes** (requires extrinsic calibration):
- **Full View**: Each image shown in its original aspect ratio, no cropping
- **FOV Focus**: Centers on the LWIR field of view projected onto the visible image
- **Max Overlap**: Crops to the maximum overlapping region between cameras

See [Stereo Alignment — Technical Details](#stereo-alignment--technical-details) for the underlying geometry and parallax correction.

**Zoom & Pan**:
- Ctrl + Mouse Wheel: Zoom in/out
- Mouse drag: Pan image

**Overlay information on images**:
- Colored square around image indicating applied label/tag
- Upper-right corner text displaying label name (same color) and if its auto or manually annotated.
- Calibration images: detected corners displayed with reprojection error (intrinsic and extrinsic below the label name text). Outliers have red color in the corresponding reprojection error
- Tagged to be deleted images: X crossing the whole image

**View → Show image info overlay** (toggle, persisted):
- When disabled, hides all overlay information (status text, calibration markers, corners, reprojection errors)
- Grid and label overlays remain visible (controlled separately)
- Useful for taking screenshots without clutter

Under each image, a text box shows the metadata in YAML format.

Below that, a stats section displays the dataset summary: tagged images, calibration summary, etc. The stats should be updated when changes are made.

### Tagging System

Images can be tagged for deletion either automatically (via sweeps) or manually. Tag reasons include:

- **Missing pair**: One of the pair images doesn't exist
- **Duplicate**: Identical or near-identical images
- **Blurry**: Image out of focus
- **Motion blur**: Blur caused by camera/subject movement
- **Sync error**: Synchronization problem between LWIR and Visible
- **Pattern match**: Matches unwanted patterns (from config/patterns folder)
- **User marked**: Generic manual deletion

Tags can be applied via keyboard shortcuts or right-click context menu.

Additionally, images can be marked as **calibration candidates** (independent from delete tags).

### Navigation & Controls

**Buttons**:
- Previous ×5 (skip 5 images back)
- Previous image
- Go to… (jump to specific image number)
- Next image
- Next ×5 (skip 5 images forward)
- Delete selected (shows count)

**Keyboard navigation**:
- Left / Right Arrow: Navigate images
- Space: Next image
- Shift + Left / Shift + Right: Navigate ×5
- Ctrl+G: Go to image number
- C: Toggle calibration mark
- `[` / `]`: Adjust parallax correction ±1 px (stereo alignment modes only)
- Shift + `[` / `]`: Adjust parallax correction ±5 px
- F11: Toggle fullscreen
- Esc: Cancel manual label selection / exit fullscreen

**Tag shortcuts** (quick keys):
- Del: User marked (generic delete)
- Ctrl+Shift+D: Duplicate
- Ctrl+Shift+B: Blurry
- Ctrl+Shift+M: Motion blur
- Ctrl+Shift+S: Sync error
- Ctrl+Shift+P: Pattern match
- Ctrl+Shift+C: Toggle calibration candidate

**Labelling shortcuts**:
- Ctrl+L: Toggle manual labelling mode
- Ctrl+Shift+L: Toggle auto labelling mode

### Automated Sweeps

Sweeps automatically detect delete candidates. Can be executed at:
- **Workspace level**: Iterates over all collections/datasets
- **Collection/Dataset level**: Processes single collection or dataset

Images tagged by sweeps are marked as auto-detected (vs manual tagging).

### Context Menus (Right-Click)

**Image viewer** (right-click on image):
- Navigate (previous / next)
- Toggle delete marks by reason (manual, duplicate, blurry, motion, sync, missing, pattern)
- Toggle calibration candidate
- Re-run calibration detection (only for calibration-marked images)
- Enter manual or auto labelling mode
- Copy image path (LWIR / Visible)

**Labelling mode** (right-click on a label):
- Accept label (for auto-detected labels, individual or accept all on image)
- Edit label (change class or bounding box; edit dialog includes Delete with confirmation)
- Delete label
- Switch between manual / auto labelling mode
- Exit labelling mode

When multiple annotations of the same class overlap, the context menu shows compact bbox info (size and position) to distinguish them.

## Menu Structure

### File Menu

- Save current status (Ctrl+S)
- Exit (Ctrl+Q)

### Workspace Menu

- Open workspace…
- Refresh workspace
- Detect delete candidates → submenu: Duplicates, Blur/motion, Patterns, Missing pairs, Run all
- Untag delete candidates → submenu: all, or by category
- Delete marked → submenu: all, or by category
- Restore from trash → submenu: all, or by category
- Clear empty dataset folders
- Default Calibration → Set from current dataset, Clear default, Show calibration info
- Label report: Aggregated label statistics across all workspace datasets
- Reset selected dataset (dangerous)
- Reset workspace (dangerous)

### View Menu

Display options:

**Filter** (control which images appear in carousel, mutually exclusive):
- All images
- Calibration candidates (any, both found, partial, missing detections)
- Delete candidates (all, manual, duplicate, blurry, motion, sync, missing pair)

**Stereo Alignment** (requires calibration):
- Disabled / Full View / FOV Focus / Max Overlap

**Overlays**:
- Undistort images (toggle)
- Show grid (Off / Thirds / Detailed 9×9)
- Show labels (toggle)
- Show image info overlay (toggle)
- Corner display mode (Original / Subpixel / Both)

**Display**:
- Toggle fullscreen (F11)

### Dataset Menu

- Load dataset…
- Load recent (submenu with recently opened datasets)
- Detect delete candidates → submenu: Duplicate sweep, Blur/motion sweep, Pattern sweep
- Untag delete candidates → submenu: all, or by category
- Delete marked → submenu: all selected, or by category
- Restore from trash → submenu: all, or by category
- Reset dataset (dangerous)

### Calibration Menu

Operations for images marked as calibration candidates:

- **Auto search calibration candidates**: Scan the dataset for chessboard patterns
- **Detect chessboards**: Run corner detection on all calibration candidates
- **Refine chessboard corners**: Apply subpixel refinement for better accuracy
- **Use Subpixel Corners**: Toggle whether calibration uses subpixel-refined corners
- **Compute calibration matrices**: Solve LWIR and visible camera intrinsics
- **Compute extrinsic transform**: Estimate stereo transform between cameras
- **Check calibration report**: Inspect matrices, reprojection errors, and export options
- **Check calibration outliers**: Dialog with per-image reprojection errors. Automated outlier candidates highlighted in red. Checkboxes control which images participate in each calibration
- **Export calibration debug overlays**: Export debug images with drawn corners and reprojection overlays
- **Import calibration data**: Load calibration from external files

### Labelling Menu

Detection labelling toolchain for object detection workflows:

**Configuration**:
- **Configure model**: Select detection model. Supports Grounding DINO (base/tiny), YOLO26 (large/small), custom YOLO (.pt), and Ensemble (GDINO + YOLO26 with IoU fusion)
- **Configure labels YAML**: Select labels YAML defining object classes, colors, and attributes. Copied to the dataset for persistence
- **Reload labels config from source**: Re-read labels YAML from disk
- **Detection channel**: Toggle inference channel (Visible / LWIR)

**Inference operations**:
- **Run labelling on current**: Run detection on the currently displayed image
- **Run labelling on dataset**: Batch-run detection across all images for the active channel

**Labelling modes**:
- **Manual labelling mode** (Ctrl+L): Draw bounding boxes with crosshair guides (click two corners, pick class via autocomplete). Right-click labels to edit or delete. The edit dialog includes a Delete button with confirmation popup
- **Auto labelling mode** (Ctrl+Shift+L): Detection runs automatically on every image navigation. Labels appear as pending (AUTO) until accepted. Truncation attribute is auto-set when a bbox edge falls within 2% of the image boundary
- **Clear labels for current**: Remove all label files for the current pair

**Label acceptance workflow**:
- Auto-detected labels are marked as pending (AUTO) with a distinct visual style (dashed border, ⟳ icon)
- Right-click a label to accept it individually, or use "Accept all labels" to accept all pending labels on the image
- Editing an auto-detected label automatically accepts it
- Only accepted (REVIEWED) and manual labels are saved to disk; AUTO labels are ephemeral

**Reports**:
- **Label report**: Opens a dialog summarizing label statistics for the current dataset/collection — total annotations, images labeled, per-channel and per-source breakdown, per-class counts with attribute distributions. Includes a 2×2 chart grid (class histogram, bbox overlay, centre heatmap, w×h size heatmap) with per-class filter selector. Uses `.labels_summary_cache.yaml` for fast reload.

**Overlay display**: When labels are enabled (View → Show labels), bounding boxes appear on images with class name and color. Only class-specific attributes are shown in the overlay text (universal attributes like occlusion/truncation are hidden for clarity). Auto-detected labels show a dashed border and ⟳ suffix; accepted/manual labels show a solid border.

### Help Menu

- **See help** (Ctrl+H): Opens a help window describing panels, context menus, all menu options, keyboard shortcuts, and contact information

## Stereo Alignment — Technical Details

### FOV Projection

The LWIR and visible cameras have different fields of view (FOV).  The LWIR
FOV is smaller and represents a sub-region of the visible image.  To show both
images side-by-side with consistent spatial correspondence the system computes
where each LWIR pixel projects onto the visible image using stereo calibration.

The mapping uses the **extrinsic calibration** (rotation **R** and translation
**T** from LWIR to visible) together with the **intrinsic matrices** (K_lwir,
K_vis) of both cameras.

#### Homography via stereo rectification

`cv2.stereoRectify` is called with `CALIB_ZERO_DISPARITY` to compute
rectification transforms $R_1$, $R_2$ and new projection matrices $P_1$, $P_2$.
From these we derive per-camera homographies:

$$
H_1 = P_1^{3\times3} \cdot R_1 \cdot K_{\text{lwir}}^{-1} \qquad \text{(LWIR} \to \text{rectified)}
$$

$$
H_2 = P_2^{3\times3} \cdot R_2 \cdot K_{\text{vis}}^{-1} \qquad \text{(Visible} \to \text{rectified)}
$$

The combined LWIR → Visible mapping is:

$$
H = H_2^{-1} \cdot H_1
$$

This homography assumes **infinite scene depth** (equivalent to a pure
rotation between cameras).

### Parallax and Finite Depth

For objects at a finite depth $d$, the cameras see them from slightly different
viewpoints due to the physical baseline separation.  This causes a residual
translation (parallax) that the infinite-plane homography does not capture.

The plane-induced homography for a fronto-parallel plane at depth $Z$ (in
calibration units, i.e. chessboard squares) is:

$$
H_Z = K_{\text{vis}} \left( R + \frac{T \, n^\top}{Z} \right) K_{\text{lwir}}^{-1}
$$

where $\mathbf{n} = [0,\, 0,\, 1]^\top$ (fronto-parallel normal).  The parallax
residual relative to the infinite-depth homography is:

$$
\Delta H = H_Z - H_\infty = \frac{1}{Z} \, K_{\text{vis}} \, T \, n^\top \, K_{\text{lwir}}^{-1}
$$

For a point $\mathbf{p}$ in the LWIR image the pixel shift in the visible
image simplifies to:

$$
\Delta \mathbf{p} = \frac{K_{\text{vis}} \, T}{Z}
$$

(the scalar $n^\top K_{\text{lwir}}^{-1} \mathbf{p}$ equals 1 for any normalised point).

#### Conversion to physical units

The translation $T$ is expressed in **chessboard-square units** (the
calibration grid has unit square size).  To convert depth from metres to
square units:

$$
Z = \frac{d}{s}
$$

where $s$ is the physical square side length in metres.  Substituting:

$$
\Delta \mathbf{p} = \frac{K_{\text{vis}} \, T \cdot s}{d}
\qquad \Rightarrow \qquad
|\Delta| = \frac{\lVert \mathbf{e}_{xy} \rVert \cdot s}{d}
$$

The **direction** of the shift in pixel space is the 2D part of the
epipole $\mathbf{e} = K_{\text{vis}} \, T$, which is determined entirely by
the calibration.

The **magnitude** depends on three quantities:

| Symbol | Meaning | Typical value |
|--------|---------|---------------|
| $\lVert\mathbf{e}\rVert$ | Epipole norm (pixels) | 50–500 px |
| $s$ | Square size (metres) | 0.020–0.060 m |
| $d$ | Scene depth (metres) | 10–100 m |

#### Residual error at other depths

If the correction is tuned for a reference depth $d_0$, the residual error at
depth $d$ is:

$$
\varepsilon = \lVert\mathbf{e}\rVert \cdot s \cdot \left| \frac{1}{d} - \frac{1}{d_0} \right|
$$

For typical urban scenes (baseline ~5 cm, $f \approx 500$ px) with $d_0 = 30$ m:

| Object depth | Residual error |
|--------------|----------------|
| 15 m | ~0.8 px |
| 20 m | ~0.4 px |
| 50 m | ~0.3 px |
| 100 m | ~0.2 px |

### Parallax Correction in the Application

When any stereo alignment mode is activated for the first time, the system
auto-computes the parallax correction for a default depth of **30 m** using
the formula above.  The chessboard square size is read from `config.py`
(`chessboard_square_size_mm`, default 60 mm).  If for any reason the value is
not available, a dialog asks the user to enter it in millimetres.

The correction is applied as a translation matrix pre-multiplied onto $H$:

$$
H_{\text{adj}} = \begin{bmatrix} 1 & 0 & \Delta x \\ 0 & 1 & \Delta y \\ 0 & 0 & 1 \end{bmatrix} \cdot H
$$

where $[\Delta x,\, \Delta y] = \text{correction} \cdot \dfrac{\mathbf{e}}{\lVert\mathbf{e}\rVert}$.

#### Manual fine-tuning

After the auto-computation, the user can fine-tune with keyboard shortcuts:

| Key | Action |
|-----|--------|
| `]` | +1 px along baseline direction |
| `[` | −1 px along baseline direction |
| Shift + `]` | +5 px |
| Shift + `[` | −5 px |

The current value is displayed in the status bar and persisted per workspace.

The same correction is applied to **label projection** between channels, so
that bounding boxes projected from visible to LWIR (or vice versa) remain
consistent with the visual alignment.

## Data Persistence

All data is persistent and stored at dataset level in cache files:

- **`.image_labels.yaml`**: Marks, calibration flags, overrides
- **`.summary_cache.yaml`**: Stats summary for fast workspace table loading
- **`calibration/*.yaml`**: Detected corners and image sizes per image
- **`calibration_intrinsic.yaml`** / **`calibration_extrinsic.yaml`**: Clean calibration matrices (exportable)
- **`.calibration_errors_cached.yaml`**: Per-view reprojection errors (hidden cache)
- **`.workspace_config.yaml`**: Workspace-level settings (default calibration paths, chessboard square size)
- **`.labels_summary_cache.yaml`**: Label summary statistics for fast report loading
- **`labels/*.txt`**: YOLO format detection labels

Data ownership follows hierarchy: Dataset produces and stores → Collection aggregates → Workspace coordinates. Collections and workspace store minimal information, mostly consuming aggregated data from below.

When changes to the data are detected the cache has to be updated. To avoid too many write operations each datasets is in charge of updating the cache files and will have a timer that stores all pending updates when the dataset is stable (2-4 seconds without changes).

## Progress Tracking & Concurrency

All long-running operations are non-blocking and execute in background threads to keep the GUI responsive.

**Progress Display**:
- Progress bar appears at bottom-right of window showing current operation
- Cancel button next to progress bar allows stopping operations in progress
- Status messages appear at bottom-left (e.g., "Save pending", "Saving cache...", "Sweep complete")

**Multi-level TQDM bars** (for workspace/dataset operations):
- **Workspace level**: Operation title + dataset count (e.g., "Duplicate sweep: 5/24 datasets")
- **Operation level**: Image count within dataset (e.g., "[BremenSet03/25-12-17_12-52] 450/910 images")

Format for collection datasets in TQDM bar and other identification: `{CollectionName}/{DatasetName}`

**Concurrency behavior**:
- Multiple datasets processed in parallel (ThreadPoolExecutor with configurable workers)
- CPU-intensive operations (calibration detection, quality analysis) use limited parallelism to avoid overload
- I/O operations (workspace scan, cache writing) can run with higher parallelism
- Progress updates coordinated to avoid visual conflicts between TQDM bars and log messages
- Long operations can be cancelled mid-execution - cleanup performed gracefully
