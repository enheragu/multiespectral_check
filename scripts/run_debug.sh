#!/bin/bash
# Script para lanzar la aplicación con debug, coverage y guardar logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
COVERAGE_DIR="$PROJECT_DIR/.coverage_data"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/debug_$TIMESTAMP.log"
LATEST_LOG="$LOG_DIR/latest.log"

# Crear directorios si no existen
mkdir -p "$LOG_DIR"
mkdir -p "$COVERAGE_DIR"

# Activar venv si existe
if [ -d "$PROJECT_DIR/../venv" ]; then
    echo "Activating venv..."
    source "$PROJECT_DIR/../venv/bin/activate"
fi

# Mostrar información
echo "======================================================"
echo "Launching multiespectral_check with debug logging + coverage"
echo "======================================================"
echo "Log file: $LOG_FILE"
echo "Latest log symlink: $LATEST_LOG"
echo "Coverage data: $COVERAGE_DIR/.coverage"
echo ""
echo "Press Ctrl+C to stop"
echo "======================================================"
echo ""

# Lanzar aplicación con coverage acumulado y todos los flags de debug
(cd "$PROJECT_DIR/src"
LOG_LEVEL="DEBUG" \
DEBUG_HANDLER=1 \
DEBUG_WORKSPACE=1 \
DEBUG_SESSION=1 \
DEBUG_CORNERS=1 \
DEBUG_CACHE=1 \
DEBUG_COLLECTION=1 \
DEBUG_MARKS=1 \
DEBUG_TIMING=1 \
DEBUG_STATS=1 \
DEBUG_PATTERN=1 \
python3 -m coverage run -a main.py 2>&1 | tee "$LOG_FILE")

# Coverage guarda automáticamente al salir
# Mover coverage al directorio de datos si existe
if [ -f ".coverage" ]; then
    mv .coverage "$COVERAGE_DIR/.coverage"
    echo "✓ Coverage data saved to $COVERAGE_DIR/.coverage"
fi

# Crear symlink al último log
ln -sf "$LOG_FILE" "$LATEST_LOG"

echo ""
echo "======================================================"
echo "Application closed"
echo "Log saved to: $LOG_FILE"
echo ""
echo "💡 Coverage tips:"
echo "  - View report: ./scripts/coverage_report.sh"
echo "  - Reset coverage: rm $COVERAGE_DIR/.coverage"
echo "  - HTML report: ./scripts/coverage_html.sh"
echo "======================================================"
