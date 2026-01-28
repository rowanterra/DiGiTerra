@echo off
REM Build script for Windows
REM This script builds DiGiTerra as a Windows executable
REM Run this script from the project root directory

echo Building DiGiTerra for Windows...
echo.

REM Change to project root if running from build directory
if exist "..\desktop_app.py" (
    cd ..
)

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo PyInstaller is not installed. Installing...
    pip install pyinstaller
)

REM Build the executable
echo.
echo Running PyInstaller...
pyinstaller build\DiGiTerra_Windows.spec

if errorlevel 1 (
    echo.
    echo Build failed! Check the output above for errors.
    pause
    exit /b 1
)

echo.
echo Build successful!
echo The executable is located in: dist\DiGiTerra\
echo.
echo You can distribute the entire DiGiTerra folder to users.
echo.
pause
