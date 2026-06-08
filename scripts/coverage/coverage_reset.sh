#!/bin/bash
# Reset coverage data

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COVERAGE_DIR="$PROJECT_DIR/.coverage_data"

if [ -f "$COVERAGE_DIR/.coverage" ]; then
    rm "$COVERAGE_DIR/.coverage"
    echo "✅ Coverage data reset"
else
    echo "ℹ️  No coverage data to reset"
fi
