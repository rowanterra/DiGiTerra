# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules

hiddenimports = (
    collect_submodules("xlsxwriter")
    + collect_submodules("openpyxl")
    + collect_submodules("seaborn")
    + collect_submodules("shap")
)

datas = [
    ("../templates", "templates"),
    ("../static", "static"),
    ("../python_scripts", "python_scripts"),
]

block_cipher = None


a = Analysis(
    ["../desktop_app.py"],
    pathex=[],
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
#   4. Then uncomment the icon line below and comment out the icon=None line
app = BUNDLE(
    coll,
    name="DiGiTerra.app",
    icon=None,  # Set to "static/Terra_Axe_Logo.icns" after creating the .icns file
    bundle_identifier=None,
)
