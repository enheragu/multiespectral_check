#!/bin/bash
# Script para ejecutar batería completa de tests con análisis de logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEST_LOG="$LOG_DIR/test_$TIMESTAMP.log"
LATEST_TEST_LOG="$LOG_DIR/latest_test.log"

# Crear directorio de logs si no existe
mkdir -p "$LOG_DIR"

# Activar venv si existe
if [ -d "$PROJECT_DIR/../venv" ]; then
    echo "Activating venv..."
    source "$PROJECT_DIR/../venv/bin/activate"
else
    echo "WARNING: venv not found at $PROJECT_DIR/../venv"
fi

echo "======================================================"
echo "Running multiespectral_check test battery"
echo "======================================================"
echo "Test log: $TEST_LOG"
echo "Project: $PROJECT_DIR"
echo ""

# Función para analizar logs
analyze_logs() {
    local log_file="$1"
    echo ""
    echo "======================================================"
    echo "LOG ANALYSIS"
    echo "======================================================"

    # Contar errores
    local errors=$(grep -c "ERROR" "$log_file" 2>/dev/null || echo "0")
    local warnings=$(grep -c "WARNING" "$log_file" 2>/dev/null || echo "0")
    local failures=$(grep -c "FAILED" "$log_file" 2>/dev/null || echo "0")
    local passed=$(grep -c "PASSED" "$log_file" 2>/dev/null || echo "0")

    echo "Test Results:"
    echo "  - Passed: $passed"
    echo "  - Failed: $failures"
    echo "  - Warnings: $warnings"
    echo "  - Errors: $errors"
    echo ""

    # Mostrar errores si existen
    if [ "$errors" -gt 0 ]; then
        echo "ERROR messages found:"
        grep "ERROR" "$log_file" | head -10
        echo ""
    fi

    # Mostrar fallos si existen
    if [ "$failures" -gt 0 ]; then
        echo "FAILED tests:"
        grep "FAILED" "$log_file"
        echo ""
    fi

    echo "======================================================"
}

# Ejecutar pytest con verbose y captura de logs
cd "$PROJECT_DIR"
echo "Running pytest..."
echo ""

pytest tests/ \
    -v \
    --tb=short \
    --log-cli-level=INFO \
    --capture=no \
    2>&1 | tee "$TEST_LOG"

EXIT_CODE=${PIPESTATUS[0]}

# Crear symlink al último log de test
ln -sf "$TEST_LOG" "$LATEST_TEST_LOG"

# Analizar logs
analyze_logs "$TEST_LOG"

# Resumen final
echo ""
echo "======================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ All tests PASSED"
else
    echo "✗ Some tests FAILED (exit code: $EXIT_CODE)"
fi
echo "======================================================"
echo "Full log: $TEST_LOG"
echo "Latest: $LATEST_TEST_LOG"
echo "======================================================"

exit $EXIT_CODE
