#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)"
DESKTOP_FILE="$DIR/scview.desktop"
INSTALL_DIR="$HOME/.local/share/applications"

if [ ! -f "$DESKTOP_FILE" ]; then
    echo "ERROR: $DESKTOP_FILE not found"
    exit 1
fi

mkdir -p "$INSTALL_DIR"
cp "$DESKTOP_FILE" "$INSTALL_DIR/scview.desktop"

# Update desktop database if available
if command -v update-desktop-database &>/dev/null; then
    update-desktop-database "$INSTALL_DIR" 2>/dev/null || true
fi

echo "Desktop launcher installed to $INSTALL_DIR/scview.desktop"
echo "You should now see 'scView' in your application menu."
