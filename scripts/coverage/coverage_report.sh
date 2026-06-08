#!/bin/bash
# Mostrar reporte de coverage en terminal

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COVERAGE_DIR="$PROJECT_DIR/.coverage_data"

if [ ! -f "$COVERAGE_DIR/.coverage" ]; then
    echo "❌ No coverage data found!"
    echo "Run the app with ./scripts/run_debug.sh first"
    exit 1
fi

cd "$PROJECT_DIR"
# Copiar temporalmente para que coverage lo encuentre
cp "$COVERAGE_DIR/.coverage" .coverage
python3 -m coverage report --sort=cover "$@"
rm .coverage

echo ""
echo "💡 Tips:"
echo "  - Show missing lines: ./scripts/coverage_report.sh --show-missing"
echo "  - HTML report: ./scripts/coverage_html.sh"
echo "  - Reset coverage: ./scripts/coverage_reset.sh"
