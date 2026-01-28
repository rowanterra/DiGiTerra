#!/bin/bash
# Build script for Linux
# This script builds DiGiTerra as a Linux executable

set -e

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

ensure_project_root

echo "Building DiGiTerra for Linux..."
echo

check_pyinstaller
run_build "build/DiGiTerra_Linux.spec"

echo
echo "Build successful!"
echo "The executable is located in: dist/DiGiTerra/"
echo
echo "Make the executable: chmod +x dist/DiGiTerra/DiGiTerra"
echo
