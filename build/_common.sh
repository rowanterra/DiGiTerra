#!/bin/bash
# Common functions for build scripts

# Ensure we're in the project root
ensure_project_root() {
    if [ -f "../desktop_app.py" ]; then
        cd ..
    fi
}

# Check and install PyInstaller if needed
check_pyinstaller() {
    # Use 'python' to respect the active environment (conda, venv, etc.)
    if ! python -c "import PyInstaller" 2>/dev/null; then
        echo "PyInstaller is not installed. Installing..."
        python -m pip install pyinstaller
    else
        echo "PyInstaller is already installed."
    fi
}

# Run PyInstaller build
run_build() {
    local spec_file=$1
    echo
    echo "Running PyInstaller..."
    python -m PyInstaller "$spec_file"
    
    if [ $? -ne 0 ]; then
        echo
        echo "Build failed! Check the output above for errors."
        exit 1
    fi
}
