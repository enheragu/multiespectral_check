#!/bin/bash
# Generar reporte HTML de coverage

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COVERAGE_DIR="$PROJECT_DIR/.coverage_data"
HTML_DIR="$PROJECT_DIR/htmlcov"

if [ ! -f "$COVERAGE_DIR/.coverage" ]; then
    echo "❌ No coverage data found!"
    echo "Run the app with ./scripts/run_debug.sh first"
    exit 1
fi

echo "Generating HTML coverage report..."
cd "$PROJECT_DIR"
# Copiar temporalmente para que coverage lo encuentre
cp "$COVERAGE_DIR/.coverage" .coverage
python3 -m coverage html -d "$HTML_DIR"
rm .coverage

echo ""
echo "✅ HTML report generated!"
echo "📂 Location: $HTML_DIR/index.html"
echo ""
echo "Open with: xdg-open $HTML_DIR/index.html"
