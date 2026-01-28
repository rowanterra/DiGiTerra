# DiGiTerra

This project ships a Flask-based UI and a desktop wrapper so users can run the
app without opening Terminal or a browser tab. DiGiTerra supports **macOS, Windows, and Linux**.

**Developers / handoff:** See **`HANDOFF.md`** for repo layout, security notes, and tips for integrating into a website.

## End-user install (no Terminal)

### macOS
Provide a prebuilt `DiGiTerra.app` (or `.dmg` containing it). End users can:

1. Download the app from Releases (or your distribution link).
2. Drag `DiGiTerra.app` into `/Applications`.
3. Double-click to launch.

### Windows
Provide a prebuilt `DiGiTerra` folder. End users can:

1. Download the folder from Releases.
2. Extract the folder to their desired location (e.g., `C:\Program Files\DiGiTerra`).
3. Double-click `DiGiTerra.exe` to launch.

### Linux
Provide a prebuilt `DiGiTerra` folder. End users can:

1. Download the folder from Releases.
2. Extract the folder to their desired location (e.g., `~/Applications/DiGiTerra`).
3. Make the executable: `chmod +x DiGiTerra/DiGiTerra`
4. Double-click `DiGiTerra` or run from terminal: `./DiGiTerra/DiGiTerra`

**Note:** No Python, pip, or Terminal steps are required for end users when you distribute
a prebuilt application. The compiled application contains all Python dependencies inside it.

## Local development (browser)

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`.

**Example datasets:** The `examples/` directory contains 9 CSVs: 3 classification (Iris, Wine, Breast cancer), 3 regression (Diabetes, Synthetic, Linnerud), and 3 clustering (feature-only versions). See `examples/README.md` for usage.

## Local development (desktop window)

```bash
pip install -r requirements.txt
python desktop_app.py
```

## Building Desktop Applications

> These steps are for maintainers building the distributable.

### Prerequisites

1. Install build tooling and app dependencies:

```bash
pip install -r requirements.txt
pip install pyinstaller
```

### macOS

**Option 1: Using the build script (recommended):**
```bash
chmod +x build/build_macos.sh
./build/build_macos.sh
```

**Option 2: Manual build:**
```bash
pyinstaller build/DiGiTerra.spec
```

The resulting `DiGiTerra.app` will be in the `dist/` directory. Distribute that
`.app` (or wrap it in a `.dmg`) so users can double-click the app without
launching Terminal or installing dependencies.

**Note:** macOS requires `.icns` format for app icons. See `build/DiGiTerra.spec` for instructions on creating the icon file.

### Windows

**Option 1: Using the build script (recommended):**
```cmd
build\build_windows.bat
```

**Option 2: Manual build:**
```cmd
pyinstaller build\DiGiTerra_Windows.spec
```

The resulting `DiGiTerra` folder (containing `DiGiTerra.exe` and dependencies) will be in the `dist/` directory. Distribute the entire folder to users.

**Note:** Windows uses `.ico` format for icons. The spec file references `static/Terra_Axe_Logo.ico`.

### Linux

**Option 1: Using the build script (recommended):**
```bash
chmod +x build/build_linux.sh
./build/build_linux.sh
```

**Option 2: Manual build:**
```bash
pyinstaller build/DiGiTerra_Linux.spec
```

The resulting `DiGiTerra` folder (containing the `DiGiTerra` executable and dependencies) will be in the `dist/` directory. Distribute the entire folder to users.

**Note:** Make sure the executable has execute permissions: `chmod +x dist/DiGiTerra/DiGiTerra`

## Dependency audit (maintainers)

To avoid missing imports, run a quick dependency scan whenever you add new
Python imports:

```bash
python scripts/check_requirements.py
```

## Configuration

The desktop launcher uses a random open port by default. You can override the
port with:

**macOS/Linux:**
```bash
export DIGITERRA_PORT=5050
```

**Windows:**
```cmd
set DIGITERRA_PORT=5050
```

If the desktop app closes immediately, enable debug logging and check the log
file:

**macOS/Linux:**
```bash
export DIGITERRA_DEBUG=1
```

**Windows:**
```cmd
set DIGITERRA_DEBUG=1
```

**Log file locations:**
- **macOS**: `~/Library/Logs/DiGiTerra/digiterra.log`
- **Windows**: `%APPDATA%\DiGiTerra\Logs\digiterra.log`
- **Linux**: `~/.cache/DiGiTerra/logs/digiterra.log`

**Application data locations:**
- **macOS**: `~/Library/Application Support/DiGiTerra/user_visualizations/`
- **Windows**: `%APPDATA%\DiGiTerra\user_visualizations\`
- **Linux**: `~/.local/share/DiGiTerra/user_visualizations/`

## Deployment (Docker and Kubernetes)

Docker and Kubernetes (Helm) assets live under `deploy/`. See `deploy/README.md` for
Docker build/run and Helm install instructions.

## Accessibility

DiGiTerra is designed with accessibility in mind and includes comprehensive support for assistive technologies. The application implements ARIA attributes throughout the interface, provides full keyboard navigation support, and includes screen reader announcements for dynamic content updates. Form fields include accessible labels and help text, and all interactive elements have visible focus indicators. A skip link allows keyboard users to bypass navigation and jump directly to main content. For detailed information about accessibility features, see the Accessibility section in `docs/documentation.md`.
