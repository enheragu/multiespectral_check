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

The GUI includes two tabs. Both have a title in bold text at the left top, and a button space at the right. Below is the main view space. Under the view there's a status area: progress bar at the right side (with cancel button), and status messages at the left side (saving, stored, etc).

## Workspace Tab

Shows a table with a summary of all collections/datasets:
- Number of image pairs
- Number of deleted image pairs
- Number of images tagged to be deleted (by reason)
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

**Stereo Alignment Modes** (requires calibration):
- **Full View**: Each image shown in its original aspect ratio, no cropping
- **FOV Focus**: Centers on the LWIR field of view projected onto the visible image
- **Max Overlap**: Crops to the maximum overlapping region between cameras
- **Aligned View**: Warps LWIR to align with visible using stereo calibration

**Zoom & Pan**:
- Ctrl + Mouse Wheel: Zoom in/out
- Mouse drag: Pan image
- Double-click: Reset to fit mode

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
- Previous image
- Next image
- Delete selected (shows count)

**Keyboard navigation**:
- Left/Right Arrow: Navigate images
- C: Toggle calibration mark
- F11: Fullscreen mode

**Tag shortcuts for manual labelling** (quick keys):
- Ctrl+Shift+D: Duplicate
- Ctrl+Shift+B: Blurry
- Ctrl+Shift+M: Motion blur
- Ctrl+Shift+S: Sync error
- Del/supr: User marked

### Automated Sweeps

Sweeps automatically detect delete candidates. Can be executed at:
- **Workspace level**: Iterates over all collections/datasets
- **Collection/Dataset level**: Processes single collection or dataset

Images tagged by sweeps are marked as auto-detected (vs manual tagging).

## Menu Structure

### File Menu

- Force save cache (saves all pending changes)
- Exit

### Workspace Menu

- Open workspace
- Refresh workspace
- Sweep operations (workspace-wide automated detection)
- Delete all tagged images (worspace-wide)
- Delete by category (separate option for each tag reason workspace wide)
- Restore deleted images
- Restore deleted images by category
- Reset operations (remove tags by category)
- Reset selected collection (returns selected to its original state)
- Reset workspace (remove all GUI generated stuff, returns all Collecitions/datasets to its original state)

### View Menu

Display options:

**Filter** (control which images appear in carousel, options are mutually exclusive):
- Show all images (checkbox)
- Show calibration candidates only
- Show calibration by category (both chessboard found, outliers, only lwir chessboard, etc)
- Show delete candidates only
- Show delete candidates by category

**Alignment modes** (stereo view modes):
- Full view: Both images at native resolution, side by side
- FOV focus: Crop to overlapping region with padding for context
- Max overlap: Align images showing maximum common area
- Aligned view: Full stereo alignment with rectification

**Overlays**:
- View rectified images (checkbox)
- Show grid (checkbox for grid overlay) (default true)
- Show labels (checkbox)
- Corner display mode (Original/Subpixel/Both)

**Display**:
- Fullscreen toggle
- Zoom controls
- Show/hide metadata panels (default true)
- Show/hide stats panel (default true)

### Collection/Dataset Menu

- Load Collection
- Load recet (with sub-menut with 5 most recent collections)
**Tagging & Deletion**:
- Sweep operations (automated)
- Delete all tagged images
- Delete by category (separate option for each tag reason)
- Reset operations (remove tags by category)

**Management**:
- Restore deleted images
- Reset dataset (remove all cache/data, restore deleted images, untag all)
- Restore deleted images by category

### Calibration Menu

Operations for images marked as calibration candidates:

- **Detect chessboards**: Includes image transformations to enhance detection
- **Refine corners**: Refine to subpixel precision
- **Compute intrinsic calibration**: Per-camera calibration
- **Compute extrinsic calibration**: Stereo calibration between cameras
- **Check outliers**: Opens dialog showing reprojection errors per image with columns:
  - Image name
  - LWIR reprojection error
  - Visible reprojection error
  - Stereo reprojection error
  - Include LWIR (checkbox)
  - Include Visible (checkbox)
  - Include Stereo (checkbox)

  Automated outlier candidates appear with red cells. Checkboxes control which images participate in each calibration type.

- **Calibration report**: Shows calibration data (collection name, date, image count, matrices, parameters)
- **Import calibration**: Load external calibration from different session/dataset

### Labelling Menu

Detection labelling toolchain for object detection workflows (YOLO format):

**Configuration**:
- **Configure model**: Select YOLO model (.pt file) for inference
- **Configure labels**: Select labels YAML file defining object classes and colors

**Inference operations**:
- **Label current image**: Run detection on currently displayed image (uses channel selected in input mode)
- **Label all dataset**: Run detection across all images in dataset for selected channel

**Manual operations**:
- **Manual labelling mode**: Toggle mode to manually draw bounding boxes on images. Click and drag to create boxes, right-click existing boxes to delete them
- **Clear labels (current)**: Remove all labels from current image

**Storage**: Labels saved in YOLO format at `labels/{channel}/{channel}_{base}.txt`

**Overlay display**: When labels exist, bounding boxes appear on images with class name and color. Boxes can be interacted with in manual mode.

### Help Menu

- About (opens dialog with different sections. Title in bold followed by text box with the following sections:
  - GUI overview
  - Keyboard shortcuts reference
  - Menu options overview
  - Support information

## Data Persistence

All data is persistent and stored at dataset level in cache files:

- **`.image_labels.yaml`**: Marks, calibration flags, overrides
- **`.summary_cache.yaml`**: Stats summary for fast workspace table loading
- **`calibration/*.yaml`**: Detected corners and image sizes per image
- **`calibration_intrinsic.yaml`** / **`calibration_extrinsic.yaml`**: Clean calibration matrices (exportable)
- **`.calibration_errors_cached.yaml`**: Per-view reprojection errors (hidden cache)
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
