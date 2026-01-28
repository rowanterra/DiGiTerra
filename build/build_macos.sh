#!/bin/bash
# Build script for macOS
# This script builds DiGiTerra as a macOS .app bundle

set -e

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

ensure_project_root

echo "Building DiGiTerra for macOS..."
echo

check_pyinstaller
run_build "build/DiGiTerra.spec"

echo
echo "Build successful!"
echo "The app bundle is located in: dist/DiGiTerra.app"
echo
echo "To create a .dmg for distribution:"
echo "  hdiutil create -volname DiGiTerra -srcfolder dist/DiGiTerra.app -ov -format UDZO dist/DiGiTerra.dmg"
echo
