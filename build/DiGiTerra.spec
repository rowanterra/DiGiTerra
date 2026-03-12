# -*- mode: python ; coding: utf-8 -*-
# macOS build. Run from project root: pyinstaller build/DiGiTerra.spec

from PyInstaller.utils.hooks import collect_submodules
import os

# Paths relative to spec file (build/); SPECPATH is the directory containing the spec
_SPEC_DIR = os.path.abspath(SPECPATH)
_PROJECT_ROOT = os.path.normpath(os.path.join(_SPEC_DIR, '..'))

hiddenimports = (
    collect_submodules("xlsxwriter")
    + collect_submodules("openpyxl")
    + collect_submodules("seaborn")
    + collect_submodules("shap")
)

datas = [
    (os.path.join(_PROJECT_ROOT, "templates"), "templates"),
    (os.path.join(_PROJECT_ROOT, "static"), "static"),
    (os.path.join(_PROJECT_ROOT, "python_scripts"), "python_scripts"),
]

block_cipher = None


a = Analysis(
    [os.path.join(_PROJECT_ROOT, "desktop_app.py")],
    pathex=[_PROJECT_ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DiGiTerra",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="DiGiTerra",
)

# Note: macOS requires .icns format for app icons (PNG will not work)
# To create Terra_Axe_Logo.icns:
#   1. Use an online converter: https://cloudconvert.com/png-to-icns
#   2. Or use Image2icon app (macOS App Store)
#   3. Save the .icns file to static/Terra_Axe_Logo.icns
#   4. Then set icon to the path below (optional)
_ICNS = os.path.join(_PROJECT_ROOT, "static", "Terra_Axe_Logo.icns")
app = BUNDLE(
    coll,
    name="DiGiTerra.app",
    icon=_ICNS if os.path.exists(_ICNS) else None,
    bundle_identifier=None,
)
