# Design Philosophy - Multiespectral Dataset Checker

## Overview
Management and validation system for multispectral datasets (LWIR + Visible) with calibration, labeling, and quality analysis tools.

This file should be compact with all the design principles and guidelines followed in the codebase, without extensive examples that are already covered in the philosophy explanation.
---

## Why Read This?

This documents the key decisions that make the codebase:
- **Maintainable**: Clear ownership, no scattered state
- **Testable**: Atomic operations, pure backend logic
- **Performant**: Smart caching, controlled concurrency
- **Debuggable**: Centralized logging, fail-fast validation


## 1. Architecture: Backend / Frontend / Common

**Clear separation between business logic and GUI**:
- **backend/**: Pure Python logic. NO UI widgets (QMessageBox, QDialog, etc). ONLY Qt threading (QObject, QRunnable, pyqtSignal).
- **frontend/**: Qt-dependent UI. Widgets, dialogs, user interactions. Delegates logic to backend.
- **common/**: Shared utilities and constants used by backend AND frontend:
  - `log_utils.py` - Centralized logging with timestamps
  - `dict_helpers.py` - Safe navigation of nested dicts
  - `timing.py` - Performance monitoring and decorators
  - `reasons.py` - DELETE_REASONS constants

**Rule**: If both frontend AND backend use it → goes in `common/`. If only backend → `backend/utils/`.

**Type Safety**: Backend has stricter type checking (mypy.ini). Frontend is more permissive due to Qt complexity.

---

## 2. Single Source of Truth: cache_data

**Problem**: Distributed state caused inconsistencies, O(n) conversions on every access.

**Solution**: `cache_data` dict is the only mutable source of truth:
```python
cache_data = {
    "marks": {},           # Dict[str, Dict] - base → {reason: str, auto: bool}
    "overrides": set(),    # Set[str] in memory, List in YAML
    "calibration": {},     # Nested dict with results, outliers, etc.
    "_detection_bins": {}  # Runtime-only (not persisted)
}
```

**Unified marks format**:
```yaml
# Unified format:
marks:
  base_001:
    reason: duplicate
    auto: true
  base_002:
    reason: blurry
    auto: false
```

**Rules**:
- Properties are **read-only views** computed from cache_data
- To modify: access `cache_data` directly, NEVER through properties
- **Set↔List conversion ONLY in cache.py**: `_normalize_dataset_entry()` (List→Set on load) and `serialize_dataset_entry()` (Set→List on save)
- mypy detects violations (property mutations generate errors)

---

## 3. Data Ownership: Who produces, manages

**Hierarchy**: Workspace (aggregates) → Collection (aggregates) → Dataset (produces)

**Dataset** (low level):
- ✅ Produces and stores: marks (with embedded auto flag), calibration corners, sweep_flags
- ✅ Files: `.image_labels.yaml`, `.summary_cache.yaml`, `calibration/*.yaml`

**Collection** (mid level):
- ✅ Aggregates child data with namespacing (`child/base`)
- ✅ Distributes changes back to children
- ❌ Does NOT have its own cache, does NOT produce data

**Workspace** (high level):
- ✅ Unified view, coordinates sweeps
- ❌ Does NOT modify child data directly

**Implementation**: See `backend/services/collection.py` (~656 lines) which encapsulates all aggregation/distribution logic.

**Key rule**: Delegate to the lowest level where the information lives. Workspace coordinates but does NOT execute; Dataset executes and stores.

---

## 4. Atomicity and Encapsulation

**Principle**: Complete, self-contained operations. Whoever handles the information, encapsulates the behavior.

**Atomicity**:
```python
# ✅ Atomic operation: mark, update counters, save
def mark_image(self, base: str, reason: str, auto: bool = False) -> None:
    self.cache_data["marks"][base] = {"reason": reason, "auto": auto}
    self.rebuild_reason_counts()
    self.mark_cache_dirty()

# ❌ NO: scattered steps that can be forgotten
marks[base] = reason
# ... 50 lines later...
rebuild_counts()  # Easy to forget
```

**Behavior encapsulation**:
```python
# ✅ Dataset encapsulates complete sweep
def run_duplicate_sweep(self) -> int:
    count = self._compute_duplicates()
    self.mark_cache_dirty()
    self.mark_sweep_done('duplicates')
    return count

# ❌ NO: logic scattered in caller
count = compute_duplicates(dataset)
dataset.mark_cache_dirty()
dataset.mark_sweep_done('duplicates')
```

**Rules**:
- ✅ One operation = one method that does EVERYTHING
- ✅ Whoever produces data, validates and persists it
- ❌ DO NOT split logically atomic operations
- Reuse existing functionality. Don't repeat code that could introduce errors when a tested function already exists.
---

## 5. Clean Code: Functions with Purpose

**Principle**: Every function should add value. No trivial getters/setters. Each class is atomic and responsible for its own data.

**✅ Functions with logic**: Validate, transform, encapsulate.
**✅ Useful properties**: Derived calculations, complex aggregations.
**❌ NO**: Getters/setters that only do `return self.x` or `self.x = value`.

**Rule**: If the function doesn't validate, doesn't transform, and doesn't encapsulate logic → it's probably unnecessary.

We aim to keep code from growing too large in any single file, encapsulating functionality into new objects/files when appropriate. We also avoid having too many files — find a middle ground.
---

## 6. Configuration: Single Source of Truth

**All configuration in `config.py`**: constants, paths, timeouts, limits, magic numbers.

```python
# ✅ Direct
from config import get_config
config = get_config()
self.size = config.chessboard_size

# ❌ DO NOT re-export
CHESSBOARD_SIZE = config.chessboard_size  # Unnecessary
```

**Rules**:
- ✅ Direct access via `config.property`
- ❌ DO NOT duplicate constants
- ❌ DO NOT re-export for "convenience"
- ✅ One `get_config()` per module, reuse the instance

**Examples in config.py**:
```python
@dataclass
class Config:
    default_dataset_dir: Path = Path.home() / "datasets"
    chessboard_size: Tuple[int, int] = (7, 7)
    calibration_intrinsic_filename: str = "calibration_intrinsic.yaml"
    calibration_extrinsic_filename: str = "calibration_extrinsic.yaml"
    overlay_cache_limit: int = 24
    signature_scan_timer_interval_ms: int = 100

    # Progress task IDs (for consistent tracking)
    progress_task_detection: str = "calibration-detect"
    progress_task_signatures: str = "signature-scan"
    progress_task_quality: str = "quality-scan"
```

---

## 7. Fail Fast: None Checks & Error Handling

**Principle**: Validate at the start, fail fast, don't mask errors.

**Pattern**:
```python
def process_data(obj: Optional[Foo]) -> Result:
    # ✅ Validate at the start
    if obj is None:
        log_error("obj is None, cannot process")
        return Result.error()

    # ✅ Rest of the code assumes obj is valid
    return obj.compute()
```

**Rules**:
- ✅ None checks at the start of functions
- ✅ If something critical is None: log_error and return
- ✅ If it "shouldn't be None": let it crash (detects programming bugs)
- ❌ DO NOT fill code with `if obj is not None` on every use
- ❌ DO NOT mask errors with silent default values

---

## 8. YAML-Compatible Types

**Prefer serializable types**:
- ✅ `list` over `tuple` (YAML serializable)
- ✅ `dict` over custom classes
- ✅ Primitive types: str, int, float, bool
- ⚠️ Numpy matrices: only where necessary (calibration)

**Conversion at boundaries**:
- Memory: Set[str] (efficiency in lookups)
- YAML: List[str] (serializable)
- Conversion ONLY in `cache.py`: `_normalize_dataset_entry()` (load) and `serialize_dataset_entry()` (save)

---

## 9. Cache Files & Persistence

**Storage Layers**:
- Dataset level: `.image_labels.yaml` (marks, calibration flags), `.summary_cache.yaml` (stats, sweep_flags)
- Calibration: `calibration/*.yaml` (corners, image_sizes), `calibration_intrinsic.yaml`, `calibration_extrinsic.yaml` (clean, exportable)
- Calibration cache: `.calibration_errors_cached.yaml` (per_view_errors, hidden)

**Separation principle**: Clean calibration files (exportable to other tools) vs hidden cache files (GUI/debug data).

**Dirty Tracking**: `session.mark_cache_dirty()` → `snapshot_cache_payload()` → async `write_cache_payload()`

---

## 10. Thread Safety & Concurrency

**Locks**:
- `_CACHE_WRITE_LOCK`: Serializa escrituras a disco
- `WorkspaceCacheCoordinator._lock`: Protege dirty_datasets
- `progress_lock`: Contadores compartidos en sweeps

**Patterns**:
- ThreadPoolExecutor por tipo de tarea (calibration, scan, sweep)
- `max_workers` basado en CPU cores y tipo de operación (I/O vs CPU-bound)
- DatasetSession: instancia por thread, NO compartir
- `as_completed()` para procesar resultados conforme llegan

**Progress Tracking**: 3 niveles tqdm (workspace → collection → operation) con posiciones fijas para evitar overlap

---

## 11. Sweep Ownership

**Principle**: Dataset executes sweeps on its own data, self-marks dirty and sweep_done.

**Pattern**: Workspace/Collection delegates → Dataset executes → Self-marks flags

**Handler Registry**: One DatasetHandler per path, persists in memory for reuse.

---

## 12. Code Cleanliness: No Legacy, No Cruft

**Principle**: Code should be current, not carry backward compatibility. Original functions should be modified to adapt to the new API; don't add wrappers that keep obsolete code alive and add extra code.

**❌ Remove**:
```python
# Legacy: combined dict for backwards compatibility
removed_by_reason = {}  # <-- Misleading "Legacy" if it's the current API

# Fallback to old scan_workspace for compatibility
results = scan_workspace()  # <-- There shouldn't be an "old" if it's the only version

# Keep complex properties for backward compatibility (will migrate in future)
@property  # <-- Migrate as soon as possible!
```

**✅ Do**:
```python
# Combine user and auto reasons for total count
removed_by_reason = {}  # <-- Clear description, no "Legacy"

results = scan_workspace()  # <-- No "fallback" if it's the only code

@property  # <-- No promises of "future migration"
def reason_counts(self) -> Dict[str, int]:
    return self.cache_data["reason_counts"]
```

**Rules**:
- ❌ NO dead code "just in case" - If it's not used, remove it
- ✅ If something IS legacy and must be kept: document WHY and when it will be removed
- ✅ When updating code: remove ALL references to old format
- ✅ Comments that explain WHAT the code does

**Example**: Remove compatibility with old formats when refactoring - code should be current, not carry legacy.

---

## 13. Code Quality & Tools

**Commands**: `mypy src/`, `pylint src/`, `pycln src/`

**Coverage**: `./scripts/run_debug.sh` (accumulates), `./scripts/coverage_report.sh` (view), `./scripts/coverage_html.sh` (HTML)

**Best Practices**:
- Imports at top, NO dispersed imports
- Centralized logging (`log_debug`, `log_perf`), NO wrapper functions checking env vars
- Type annotations on function signatures
- Magic numbers → `config.py`

---

## 14. Quick Reference

| Task | Module | Location | Notes |
|------|--------|----------|-------|
| Load dataset | `dataset_session.py` | backend/services | `load()`, `load_collection()` |
| Nested dict access | `dict_helpers.py` | **common/** | `get_dict_path()`, `set_dict_path()` |
| Logging | `log_utils.py` | **common/** | `log_debug()`, `log_info()`, etc. |
| Timing | `timing.py` | **common/** | `@timed`, `get_timestamp()` |
| Scan workspace | `workspace_inspector.py` | backend/services | `scan_workspace()` |
| Parallel tasks | `thread_pool_manager.py` | backend/services | `get_thread_pool_manager()` |
| Cache write | `cache_writer.py` | backend/services | `write_cache_payload()` |
| Delete reasons | `reasons.py` | **common/** | `REASON_*` constants |
---
