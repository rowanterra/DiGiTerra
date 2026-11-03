# Desktop build instructions

How to build the DiGiTerra desktop application on macOS, Windows, and Linux, and how that relates to the web deployment (Docker, gunicorn).

Every build command runs with the project root as the working directory (the directory with `app.py`, `requirements.txt`, and `build/`). The PyInstaller spec files resolve paths relative to the project root.

## Desktop and web

| Use | Command | Reference |
|-----|---------|-----------|
| Desktop app (macOS, Windows, Linux) | PyInstaller build, then distribute the `.app` or the `DiGiTerra/` folder | this file |
| Web app, development | `python app.py` at http://127.0.0.1:5000 | README.md |
| Web app, production | gunicorn with one worker (session state is in process memory) | docs/documentation.md |
| Web app, container | Docker or Helm, with gunicorn inside the image | deploy/README.md |

Under a URL prefix (`URL_PREFIX=digiterra` in Docker), the app serves both `/` and `/digiterra/`, so health checks and reverse-proxy mounts work at either path.

### Frontend bundle

The page loads `static/js/init.js`, `core.js`, `upload.js`, `preprocess.js`, `modeling.js`, `inference.js`, and `app.js`. With `DIGITERRA_USE_JS_BUNDLE=1` the template loads `static/js/app.bundle.js` instead. `npm run build` writes the bundle, and the bundle is committed, so a deployment does not need npm. `npm run check-bundle` in CI fails when the committed bundle differs from a fresh build.

## Prerequisites

Every platform needs Python 3.11 or 3.12, pip, and the packages in `requirements.txt` (`docs/DEPENDENCIES.md` lists the pins).

macOS: macOS 10.13 or later and the Xcode Command Line Tools.

Windows: Windows 10 or later. Some packages need the Visual C++ Redistributable.

Linux: a current distribution (Ubuntu 20.04 or later, Fedora 32 or later), `build-essential` or the equivalent, and the GTK 3 development libraries for pywebview (`sudo apt-get install python3-dev libgtk-3-dev` on Debian and Ubuntu, `sudo dnf install python3-devel gtk3-devel` on Fedora).

## macOS

Script:

```bash
chmod +x build/build_macos.sh
./build/build_macos.sh
```

Manual:

```bash
pip install -r requirements.txt
pip install pyinstaller
pyinstaller build/DiGiTerra.spec
```

Icon (optional): macOS needs an `.icns` file. Create `static/Terra_Axe_Logo.icns` with an online converter, the Image2icon app, or `iconutil -c icns Terra_Axe_Logo.iconset`, then set `icon="static/Terra_Axe_Logo.icns"` in `build/DiGiTerra.spec` in place of `icon=None`.

Output: `dist/DiGiTerra.app`, distributed as is or inside a `.dmg`.

## Windows

Script (double-click or run from a Command Prompt at the project root):

```cmd
build\build_windows.bat
```

Manual:

```cmd
pip install -r requirements.txt
pip install pyinstaller
pyinstaller build\DiGiTerra_Windows.spec
```

Icon (optional): Windows uses `.ico`. Create `static/Terra_Axe_Logo.ico` from the PNG logo with an online converter or `magick convert Terra_Axe_Logo.png Terra_Axe_Logo.ico`. The spec uses the file when the file exists.

Output: `dist/DiGiTerra/` with `DiGiTerra.exe`, the supporting DLLs and Python libraries, and the `templates/`, `static/`, and `python_scripts/` directories. The whole folder is distributed.

## Linux

Script:

```bash
chmod +x build/build_linux.sh
./build/build_linux.sh
```

Manual:

```bash
pip3 install -r requirements.txt
pip3 install pyinstaller
pyinstaller build/DiGiTerra_Linux.spec
```

Icon (optional): the spec uses `static/Terra_Axe_Logo.png` when the file exists.

Output: `dist/DiGiTerra/` with the `DiGiTerra` executable, the supporting libraries, and the `templates/`, `static/`, and `python_scripts/` directories. The whole folder is distributed. The end user extracts the folder, runs `chmod +x DiGiTerra/DiGiTerra`, and starts `./DiGiTerra/DiGiTerra`.

## Troubleshooting

"Module not found" at runtime: `python scripts/check_requirements.py` confirms that each package in `requirements.txt` imports. A package that PyInstaller does not detect goes in `hiddenimports` in the spec file.

Executable size: PyInstaller includes the interpreter and every dependency. Builds of this project have been 400 to 800 MB.

Crash at startup: set `DIGITERRA_DEBUG=1` (`set DIGITERRA_DEBUG=1` on Windows) and read the log file listed in README.md. Building with `console=True` in the spec file shows the error in a terminal.

Linux "No module named 'webview'": install the GTK 3 development libraries, then `pip3 install --force-reinstall pywebview`.

Antivirus on Windows: PyInstaller executables are sometimes reported as suspicious. Code signing or an allowlist entry resolves the report.

macOS first launch: an unsigned app may need a right-click and Open. Signing avoids the prompt: `codesign --deep --force --verify --verbose --sign "Developer ID Application: <name>" dist/DiGiTerra.app`.

Windows first launch: Windows Defender may show a SmartScreen prompt. More info, then Run anyway, starts the app. Code signing avoids the prompt.

Linux: library requirements differ between distributions, so the build is tested on each target distribution. A `.deb` or `.rpm` package is an option for installation.

## Checking the build

1. macOS: double-click `dist/DiGiTerra.app`.
2. Windows: double-click `dist/DiGiTerra/DiGiTerra.exe`.
3. Linux: run `dist/DiGiTerra/DiGiTerra`.

The application starts without an error, displays the page, accepts a CSV upload, produces a visualization, and writes files to the platform directories listed in README.md.

## Distribution

macOS: `DiGiTerra.app` as is, or a `.dmg`:

```bash
hdiutil create -volname DiGiTerra -srcfolder dist/DiGiTerra.app -ov -format UDZO dist/DiGiTerra.dmg
```

Windows: a zip of `dist/DiGiTerra`. An installer built with Inno Setup or NSIS is an option.

Linux: a zip of `dist/DiGiTerra`, or a `.deb` or `.rpm` package.
