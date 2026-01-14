#!/bin/bash
# Script para lanzar la aplicación con debug y guardar logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/debug_$TIMESTAMP.log"
LATEST_LOG="$LOG_DIR/latest.log"

# Crear directorio de logs si no existe
mkdir -p "$LOG_DIR"

# Activar venv si existe
if [ -d "$SCRIPT_DIR/../venv" ]; then
    echo "Activating venv..."
    source "$SCRIPT_DIR/../venv/bin/activate"
fi

# Mostrar información
echo "======================================================"
echo "Launching multiespectral_check with debug logging"
echo "======================================================"
echo "Log file: $LOG_FILE"
echo "Latest log symlink: $LATEST_LOG"
echo ""
echo "Press Ctrl+C to stop"
echo "======================================================"
echo ""

# Lanzar aplicación con todos los flags de debug
cd "$SCRIPT_DIR/src" 
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
python3 main.py 2>&1 | tee "$LOG_FILE"

# Crear symlink al último log
ln -sf "$LOG_FILE" "$LATEST_LOG"

echo ""
echo "======================================================"
echo "Application closed"
echo "Log saved to: $LOG_FILE"
echo "======================================================"
