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

**Separación clara entre lógica de negocio y GUI**:
- **backend/**: Pure Python logic. NO UI widgets (QMessageBox, QDialog, etc). SOLO threading Qt (QObject, QRunnable, pyqtSignal).
- **frontend/**: Qt-dependent UI. Widgets, dialogs, user interactions. Delega lógica a backend.
- **common/**: Shared utilities y constantes usadas por backend Y frontend:
  - `log_utils.py` - Logging centralizado con timestamps
  - `dict_helpers.py` - Navegación segura de dicts anidados
  - `timing.py` - Performance monitoring y decorators
  - `reasons.py` - Constantes de DELETE_REASONS

**Regla**: Si frontend Y backend lo usan → va en `common/`. Si solo backend → `backend/utils/`.

**Type Safety**: Backend tiene type checking más estricto (mypy.ini). Frontend más permisivo por complejidad de Qt.

---

## 2. Single Source of Truth: cache_data

**Problema**: Estado distribuido causaba inconsistencias, conversiones O(n) en cada acceso.

**Solución**: `cache_data` dict es la única fuente de verdad mutable:
```python
cache_data = {
    "marks": {},           # Dict[str, Dict] - base → {reason: str, auto: bool}
    "overrides": set(),    # Set[str] in memory, List in YAML
    "calibration": {},     # Nested dict con results, outliers, etc.
    "_detection_bins": {}  # Runtime-only (no persiste)
}
```

**Formato unificado de marks**:
```yaml
# Nuevo formato (unificado):
marks:
  base_001:
    reason: duplicate
    auto: true
  base_002:
    reason: blurry
    auto: false
```

**Reglas**:
- Properties son **vistas read-only** computadas desde cache_data
- Para modificar: acceder `cache_data` directamente, NUNCA a través de properties
- **Conversión Set↔List SOLO en cache.py**: `_normalize_dataset_entry()` (List→Set al cargar) y `serialize_dataset_entry()` (Set→List al guardar)
- mypy detecta violaciones (mutaciones de properties generan errores)

---

## 3. Data Ownership: Quien produce, gestiona

**Jerarquía**: Workspace (agrega) → Collection (agrega) → Dataset (produce)

**Dataset** (nivel bajo):
- ✅ Produce y almacena: marks (con auto flag embebido), calibration corners, sweep_flags
- ✅ Archivos: `.image_labels.yaml`, `.summary_cache.yaml`, `calibration/*.yaml`

**Collection** (nivel medio):
- ✅ Agrega datos de hijos con namespacing (`child/base`)
- ✅ Distribuye cambios de vuelta a hijos
- ❌ NO tiene cache propio, NO produce datos

**Workspace** (nivel alto):
- ✅ Vista unificada, coordina sweeps
- ❌ NO modifica datos de hijos directamente

**Implementación**: Ver `backend/services/collection.py` (~656 líneas) que encapsula toda la lógica de aggregation/distribution.

**Regla clave**: Delegar al nivel más bajo donde vive la información. Workspace coordina pero NO ejecuta; Dataset ejecuta y almacena.

---

## 4. Atomicidad y Encapsulación

**Principio**: Operaciones completas, auto-contenidas. Quien maneja la información, encapsula el comportamiento.

**Atomicidad**:
```python
# ✅ Operación atómica: marca, actualiza contadores, guarda
def mark_image(self, base: str, reason: str, auto: bool = False) -> None:
    self.cache_data["marks"][base] = {"reason": reason, "auto": auto}
    self.rebuild_reason_counts()
    self.mark_cache_dirty()

# ❌ NO: pasos dispersos que se pueden olvidar
marks[base] = reason
# ... 50 líneas después...
rebuild_counts()  # Fácil olvidar
```

**Encapsulación de comportamiento**:
```python
# ✅ Dataset encapsula sweep completo
def run_duplicate_sweep(self) -> int:
    count = self._compute_duplicates()
    self.mark_cache_dirty()
    self.mark_sweep_done('duplicates')
    return count

# ❌ NO: lógica dispersa en llamador
count = compute_duplicates(dataset)
dataset.mark_cache_dirty()
dataset.mark_sweep_done('duplicates')
```

**Reglas**:
- ✅ Una operación = un método que hace TODO
- ✅ Quien produce datos, los valida y persiste
- ❌ NO dividir operaciones lógicamente atómicas
- Intentar reutilizar funcionalidades. No repetir código que pueda inducir a errores si se puede aprovechar una función que ya existe y ha sido probada.
---

## 5. Código Limpio: Funciones con Propósito

**Principio**: Cada función debe aportar valor. No getters/setters triviales. Cada clase es atómica y responsable de sus datos.

**✅ Funciones con lógica**: Validan, transforman, encapsulan.
**✅ Properties útiles**: Cálculos derivados, agregaciones complejas.
**❌ NO**: Getters/setters que solo hacen `return self.x` o `self.x = value`.

**Regla**: Si la función no valida, no transforma, y no encapsula lógica → probablemente sobra.

Intentamos que el código no sea demasiado extenso en ningún fichero, encapsulando funcionalidades en nuevos objetos/ficheros cuando sea posible y pertinente. Tampoco queremos tener demasiados ficheros, buscar punto medio.
---

## 6. Configuration: Single Source of Truth

**Toda configuración en `config.py`**: constantes, paths, timeouts, límites, números mágicos.

```python
# ✅ Directo
from config import get_config
config = get_config()
self.size = config.chessboard_size

# ❌ NO re-exportar
CHESSBOARD_SIZE = config.chessboard_size  # Innecesario
```

**Reglas**:
- ✅ Acceso directo via `config.property`
- ❌ NO duplicar constantes
- ❌ NO re-exportar para "conveniencia"
- ✅ Un solo `get_config()` por módulo, reutilizar la instancia

**Ejemplos de config.py**:
```python
@dataclass
class Config:
    default_dataset_dir: Path = Path.home() / "datasets"
    chessboard_size: Tuple[int, int] = (7, 7)
    calibration_intrinsic_filename: str = "calibration_intrinsic.yaml"
    calibration_extrinsic_filename: str = "calibration_extrinsic.yaml"
    overlay_cache_limit: int = 24
    signature_scan_timer_interval_ms: int = 100

    # Progress task IDs (para tracking consistente)
    progress_task_detection: str = "calibration-detect"
    progress_task_signatures: str = "signature-scan"
    progress_task_quality: str = "quality-scan"
```

---

## 7. Fail Fast: None Checks & Error Handling

**Principio**: Validar al inicio, fallar rápido, no enmascarar errores.

**Patrón**:
```python
def process_data(obj: Optional[Foo]) -> Result:
    # ✅ Validar al inicio
    if obj is None:
        log_error("obj is None, cannot process")
        return Result.error()

    # ✅ Resto del código asume obj válido
    return obj.compute()
```

**Reglas**:
- ✅ Checks de None al inicio de función
- ✅ Si algo crítico es None: log_error y retornar
- ✅ Si "no debería ser None": dejarlo petar (detecta bugs de programación)
- ❌ NO llenar código con `if obj is not None` en cada uso
- ❌ NO enmascarar errores con valores por defecto silenciosos

---

## 8. YAML-Compatible Types

**Preferir tipos serializables**:
- ✅ `list` sobre `tuple` (YAML serializable)
- ✅ `dict` sobre clases custom
- ✅ Tipos primitivos: str, int, float, bool
- ⚠️ Matrices numpy: solo donde necesario (calibración)

**Conversión en fronteras**:
- Memoria: Set[str] (eficiencia en lookups)
- YAML: List[str] (serializable)
- Conversión SOLO en `cache.py`: `_normalize_dataset_entry()` (load) y `serialize_dataset_entry()` (save)

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

**Principio**: Dataset ejecuta sweeps sobre sus datos, se auto-marca dirty y sweep_done.

**Pattern**: Workspace/Collection delegan → Dataset ejecuta → Auto-marca flags

**Handler Registry**: Un DatasetHandler por path, persiste en memoria para reutilización.

---

## 12. Code Cleanliness: No Legacy, No Cruft

**Principio**: El código debe ser actual, no arrastrar compatibilidad hacia atrás. Se modificarán las funciones originales para adaptarlas a la API nueva, no se añaden wrappers que mantienen el código obsoleto y añaden código extra.

**❌ Eliminar**:
```python
# Legacy: combined dict for backwards compatibility
removed_by_reason = {}  # <-- "Legacy" engañoso si es API actual

# Fallback to old scan_workspace for compatibility
results = scan_workspace()  # <-- NO debe haber "old" si es la única versión

# Keep complex properties for backward compatibility (will migrate in future)
@property  # <-- Migrar cuanto antes!
```

**✅ Hacer**:
```python
# Combine user and auto reasons for total count
removed_by_reason = {}  # <-- Descripción clara, sin "Legacy"

results = scan_workspace()  # <-- Sin "fallback" si es el único código

@property  # <-- Sin promesas de "future migration"
def reason_counts(self) -> Dict[str, int]:
    return self.cache_data["reason_counts"]
```

**Reglas**:
- ❌ NO código muerto "por si acaso" - Si no se usa, eliminarlo
- ✅ Si algo ES legacy y debe mantenerse: documentar POR QUÉ y cuándo se eliminará
- ✅ Al actualizar código: eliminar TODA referencia a formato antiguo
- ✅ Comentarios que explican QUÉ hace el código

**Ejemplo**: Eliminar compatibilidad con formatos antiguos al refactorizar - código debe ser actual, no arrastrar legacy.

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
