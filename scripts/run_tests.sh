#!/bin/bash
# Run the test suite.
# Usage: ./scripts/run_tests.sh [pytest args]
#
# Requires the venv to be set up first:  ./scripts/install.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ -d "$PROJECT_DIR/.venv" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
else
    echo "ERROR: .venv not found. Run ./scripts/install.sh first."
    exit 1
fi

export QT_QPA_PLATFORM=offscreen
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"

echo "Running tests..."
python -m pytest tests/ -v --tb=short "$@"
