#!/bin/bash
# Install script for Multispectral Dataset Viewer
#
# Creates a uv virtual environment, installs all dependencies, and registers
# the application as a .desktop entry so it appears in the system launcher
# and shows the correct taskbar icon under GNOME/Wayland.
#
# Usage:
#   ./scripts/install.sh [--no-desktop]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_DIR/src"
ICON_DIR="$PROJECT_DIR/src/frontend/resources/media"
DESKTOP_ID="multispectral-viewer"
DESKTOP_FILE="$HOME/.local/share/applications/${DESKTOP_ID}.desktop"

INSTALL_DESKTOP=true
for arg in "$@"; do
    case "$arg" in
        --no-desktop) INSTALL_DESKTOP=false ;;
    esac
done

echo "======================================================"
echo " Multispectral Dataset Viewer — installer"
echo "======================================================"
echo " Project: $PROJECT_DIR"
echo ""

# ── 1. Check uv ────────────────────────────────────────────────────────────────
if ! command -v uv &> /dev/null; then
    echo "ERROR: 'uv' is not installed."
    echo "Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "[1/3] uv $(uv --version) found."

# ── 2. Create venv and install dependencies ────────────────────────────────────
echo ""
echo "[2/3] Setting up virtual environment..."
cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
    uv venv .venv
    echo "      Created .venv"
else
    echo "      .venv already exists, skipping creation."
fi

echo "      Installing dependencies from requirements.txt..."
uv pip install --quiet -r requirements.txt
echo "      Dependencies installed."

# ── 3. Install .desktop entry ──────────────────────────────────────────────────
if [ "$INSTALL_DESKTOP" = true ]; then
    echo ""
    echo "[3/3] Installing .desktop entry..."
    mkdir -p "$HOME/.local/share/applications"

    # Resolve python interpreter inside the venv
    PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"

    cat > "$DESKTOP_FILE" <<DESKTOP
[Desktop Entry]
Version=1.0
Type=Application
Name=Multispectral Dataset Viewer
Comment=Multispectral image dataset inspection and labelling tool
Exec=${PYTHON_BIN} ${SRC_DIR}/main.py
Icon=${ICON_DIR}/logo.png
StartupWMClass=${DESKTOP_ID}
Categories=Science;Graphics;
Keywords=multispectral;infrared;dataset;calibration;
Terminal=false
DESKTOP

    echo "      Written: $DESKTOP_FILE"

    # Refresh desktop database if possible
    if command -v update-desktop-database &> /dev/null; then
        update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
    fi
    echo "      Desktop entry registered."
else
    echo "[3/3] Skipping .desktop installation (--no-desktop)."
fi

echo ""
echo "======================================================"
echo " Installation complete!"
echo ""
echo " Run the app:    ./scripts/run_debug.sh"
echo "   or directly:  .venv/bin/python src/main.py"
if [ "$INSTALL_DESKTOP" = true ]; then
    echo ""
    echo " The app should now appear in your system launcher"
    echo " and show the correct taskbar icon."
    echo " (Log out / log in if the icon still shows a gear)"
fi
echo "======================================================"
