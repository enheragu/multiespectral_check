# Multiespectral Dataset Check GUI

Aplicación PyQt para revisar y etiquetar pares LWIR/visible: permite navegar datasets, marcar duplicados o descartes, revisar calibración, y hacer labelling automático o manual con overlays en tiempo real.

## Interfaz
- Ventana principal con dos vistas sincronizadas (LWIR y visible) y panel de progreso/estadísticas.
- Menús para cargar dataset, barrer duplicados, marcar razones de descarte, ejecutar detecciones/calibración y configurar el modelo/YAML de clases.
- Modo de etiquetado manual: banda elástica para cajas, autocompletado id:nombre desde YAML, clic derecho para borrar.
- Overlays: razones de descarte, cajas YOLO con colores por clase, aviso de esquinas sospechosas y estado de calibración.

## Estructura de código
- `src/main.py`: arranque de la app y bootstrap de la ventana principal.
- `src/image_viewer.py`: ventana principal; orquesta navegación, overlays, calibración y labelling apoyándose en servicios.
- `src/services/`: lógica reutilizable y sin UI.
	- `dataset_session.py`, `dataset_actions.py`: gestión de estado/cache del dataset y operaciones de marcado/restaurado.
	- `label_workflow.py`: IO de etiquetas YOLO, mapa de clases y caché de cajas.
	- `overlay_workflow.py`: renderizado y caché de overlays (razones, calibración, labels) sobre pixmaps.
	- `overlay_prefetcher.py`: cola y temporizador para precargar overlays vecinos.
	- `cancel_controller.py`: registra handlers cancelables y su estado para el UI.
	- `progress_queue.py`: contador de trabajos en cola con integración de cancelación.
	- `calibration_*`, `signature_controller.py`: flujo de calibración y firmas de duplicados.
	- `cache_service.py`, `cache_writer.py`: persistencia de estado en disco.
- `src/widgets/`: componentes Qt (panel de progreso, diálogos de calibración, zoom/pan, etc.).
- `src/utils/`: utilidades de overlays, estilos, textos de ayuda, razones de descarte, etc.
- `src/config/`: YAML de clases por defecto (COCO) utilizado para nombres de clase si el dataset no tiene `labels/labels.yaml`.

## Organización de datos
- Se espera un dataset con `lwir/` y `visible/` dentro de la carpeta seleccionada. Las etiquetas YOLO se guardan en `labels/<canal>/<canal>_<base>.txt` y el YAML activo se copia a `labels/labels.yaml`.

## Requisitos
- Python 3.10+ y las dependencias listadas en `requirements.txt` (PyQt6, numpy, yaml, etc.).

## Ejecución
```
pip install -r requirements.txt
python -m src.main
```