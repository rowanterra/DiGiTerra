"""Shared configuration and paths for DiGiTerra. Single source of truth for BASE_DIR, APP_SUPPORT_DIR, UPLOAD_DIR, LOG_DIR, VIS_DIR, URL_PREFIX."""
from pathlib import Path
import os
import platform

# Repo root (parent of python_scripts/)
BASE_DIR = Path(os.environ.get("DIGITERRA_BASE_DIR", Path(__file__).resolve().parent.parent))

# Platform-specific app support and log directories
if platform.system() == "Windows":
    _default_app_support = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "DiGiTerra"
    LOG_DIR = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "DiGiTerra" / "Logs"
elif platform.system() == "Linux":
    xdg_data_home = os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    _default_app_support = Path(xdg_data_home) / "DiGiTerra"
    LOG_DIR = Path(xdg_cache_home) / "DiGiTerra" / "logs"
else:
    _default_app_support = Path.home() / "Library" / "Application Support" / "DiGiTerra"
    LOG_DIR = Path.home() / "Library" / "Logs" / "DiGiTerra"

APP_SUPPORT_DIR = Path(os.environ.get("DIGITERRA_APP_SUPPORT_DIR", str(_default_app_support)))
UPLOAD_DIR = APP_SUPPORT_DIR / "uploads"

# URL prefix for deployment (e.g. /digiterra). Set env URL_PREFIX=digiterra.
_url_prefix_raw = os.environ.get("URL_PREFIX", "").strip().strip("/")
URL_PREFIX = ("/" + _url_prefix_raw) if _url_prefix_raw else ""

# Output directory for visualizations and generated files (plots, PDFs, predictions).
# app.py calls update_vis_dir() at startup to set this to APP_SUPPORT_DIR / "user_visualizations" or DIGITERRA_OUTPUT_DIR.
VIS_DIR = Path(os.environ.get("DIGITERRA_OUTPUT_DIR", str(APP_SUPPORT_DIR / "user_visualizations")))
VIS_DIR.mkdir(parents=True, exist_ok=True)


def update_vis_dir(new_path: Path):
    """Update VIS_DIR to the given path. Called at startup from app.py."""
    global VIS_DIR
    VIS_DIR = Path(new_path)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
