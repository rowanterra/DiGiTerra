# Cross-Platform Build Instructions

This document provides detailed instructions for building DiGiTerra desktop applications for macOS, Windows, and Linux.

## Prerequisites

All platforms require:
- Python 3.11 or higher
- pip (Python package manager)
- All dependencies from `requirements.txt`

### Platform-Specific Requirements

**macOS:**
- macOS 10.13 or higher
- Xcode Command Line Tools (for some dependencies)

**Windows:**
- Windows 10 or higher
- Visual C++ Redistributable (may be required for some Python packages)

**Linux:**
- A modern Linux distribution (Ubuntu 20.04+, Fedora 32+, etc.)
- Development tools: `build-essential` (Ubuntu/Debian) or equivalent
- GTK+ development libraries (for pywebview):
  - Ubuntu/Debian: `sudo apt-get install python3-dev libgtk-3-dev`
  - Fedora: `sudo dnf install python3-devel gtk3-devel`

## Building for macOS

### Quick Build
```bash
chmod +x build_macos.sh
./build_macos.sh
```

### Manual Build
```bash
pip install -r requirements.txt
pip install pyinstaller
pyinstaller DiGiTerra.spec
```

### Icon Setup (Optional)
macOS requires `.icns` format for app icons. To add an icon:

1. Create `static/Terra_Axe_Logo.icns` using one of these methods:
   - Online converter: https://cloudconvert.com/png-to-icns
   - Image2icon app (macOS App Store)
   - Command line: `iconutil -c icns Terra_Axe_Logo.iconset`

2. Update `DiGiTerra.spec`:
   ```python
   icon="static/Terra_Axe_Logo.icns",  # Uncomment this line
   # icon=None,  # Comment out this line
   ```

### Output
The build creates `dist/DiGiTerra.app` - a macOS application bundle that can be distributed directly or packaged in a `.dmg` file.

## Building for Windows

### Quick Build
Double-click `build_windows.bat` or run from Command Prompt:
```cmd
build_windows.bat
```

### Manual Build
```cmd
pip install -r requirements.txt
pip install pyinstaller
pyinstaller DiGiTerra_Windows.spec
```

### Icon Setup (Optional)
Windows uses `.ico` format for icons. To add an icon:

1. Create `static/Terra_Axe_Logo.ico` from your PNG logo:
   - Use an online converter: https://cloudconvert.com/png-to-ico
   - Or use ImageMagick: `magick convert Terra_Axe_Logo.png Terra_Axe_Logo.ico`

2. The spec file will automatically use it if it exists.

### Output
The build creates `dist/DiGiTerra/` folder containing:
- `DiGiTerra.exe` - The main executable
- Supporting DLLs and Python libraries
- `templates/`, `static/`, `python_scripts/` directories

**Distribution:** Distribute the entire `DiGiTerra` folder to users. They can run `DiGiTerra.exe` from anywhere.

## Building for Linux

### Quick Build
```bash
chmod +x build_linux.sh
./build_linux.sh
```

### Manual Build
```bash
pip3 install -r requirements.txt
pip3 install pyinstaller
pyinstaller DiGiTerra_Linux.spec
```

### Icon Setup (Optional)
Linux can use PNG format directly. The spec file will use `static/Terra_Axe_Logo.png` if it exists.

### Output
The build creates `dist/DiGiTerra/` folder containing:
- `DiGiTerra` - The main executable (make sure it's executable: `chmod +x DiGiTerra`)
- Supporting libraries and Python modules
- `templates/`, `static/`, `python_scripts/` directories

**Distribution:** Distribute the entire `DiGiTerra` folder to users. They should:
1. Extract the folder
2. Make the executable: `chmod +x DiGiTerra/DiGiTerra`
3. Run: `./DiGiTerra/DiGiTerra` or double-click in file manager

## Troubleshooting

### Common Issues

**"Module not found" errors:**
- Run `python scripts/check_requirements.py` to verify all dependencies are in `requirements.txt`
- Some packages may need to be added to `hiddenimports` in the spec file

**Large executable size:**
- This is normal - PyInstaller bundles Python and all dependencies
- macOS: ~500-800 MB
- Windows: ~400-700 MB
- Linux: ~400-700 MB

**App crashes on startup:**
- Enable debug mode: Set `DIGITERRA_DEBUG=1` (or `set DIGITERRA_DEBUG=1` on Windows)
- Check log files (see README.md for locations)
- Try building with `console=True` in the spec file to see error messages

**Linux: "No module named 'webview'":**
- Install GTK+ development libraries (see Prerequisites)
- Reinstall pywebview: `pip3 install --force-reinstall pywebview`

**Windows: Antivirus false positives:**
- PyInstaller executables are sometimes flagged by antivirus software
- This is a known issue - you may need to sign the executable or whitelist it

### Platform-Specific Notes

**macOS:**
- First launch may require right-clicking and selecting "Open" if macOS blocks unsigned apps
- To avoid this, codesign the app: `codesign --deep --force --verify --verbose --sign "Developer ID Application: Your Name" dist/DiGiTerra.app`

**Windows:**
- Windows Defender may flag the executable - this is a false positive
- Users may need to click "More info" → "Run anyway" on first launch
- Consider code signing for production releases

**Linux:**
- Different distributions may have different library requirements
- Test on target distributions before distribution
- Consider creating distribution-specific packages (.deb, .rpm) for easier installation

## Testing the Build

After building, test the application:

1. **macOS**: Double-click `dist/DiGiTerra.app`
2. **Windows**: Double-click `dist/DiGiTerra/DiGiTerra.exe`
3. **Linux**: Run `dist/DiGiTerra/DiGiTerra` from terminal

The application should:
- Launch without errors
- Display the DiGiTerra interface
- Allow file uploads
- Generate visualizations
- Save files to the appropriate platform-specific directories

## Distribution

### macOS
- Distribute `DiGiTerra.app` directly, or
- Create a `.dmg` file:
  ```bash
  hdiutil create -volname DiGiTerra -srcfolder dist/DiGiTerra.app -ov -format UDZO dist/DiGiTerra.dmg
  ```

### Windows
- Zip the entire `dist/DiGiTerra` folder
- Users extract and run `DiGiTerra.exe`
- Consider creating an installer using tools like Inno Setup or NSIS

### Linux
- Zip the entire `dist/DiGiTerra` folder
- Users extract, make executable, and run
- Consider creating `.deb` (Debian/Ubuntu) or `.rpm` (Fedora/RHEL) packages
