# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for Windows

from PyInstaller.utils.hooks import collect_submodules
import platform
import os

hiddenimports = (
    collect_submodules("xlsxwriter")
    + collect_submodules("openpyxl")
    + collect_submodules("seaborn")
    + collect_submodules("shap")
)

datas = [
    ("templates", "templates"),
    ("static", "static"),
    ("python_scripts", "python_scripts"),
]

block_cipher = None

a = Analysis(
    ["desktop_app.py"],
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
    console=False,  # Set to True if you want to see console output for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="static/Terra_Axe_Logo.ico" if os.path.exists("static/Terra_Axe_Logo.ico") else None,  # Windows uses .ico format
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
