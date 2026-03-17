"""Shared helpers for DiGiTerra routes (paths, URLs, validation)."""
from pathlib import Path

from python_scripts import config


def ensure_user_vis_dir() -> Path:
    """Ensure VIS_DIR exists and return it."""
    config.VIS_DIR.mkdir(parents=True, exist_ok=True)
    return config.VIS_DIR


def with_prefix(path: str) -> str:
    """Return path with URL_PREFIX applied."""
    normalized = path if path.startswith("/") else f"/{path}"
    if not config.URL_PREFIX:
        return normalized
    return f"{config.URL_PREFIX}{normalized}"


def allowed_file(filename: str) -> bool:
    """Check if file has an allowed extension."""
    from core.constants import ALLOWED_EXTENSIONS
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def normalize_predict_preprocess_mode(mode: str) -> str:
    """Map training-time preprocessing modes to prediction-safe modes."""
    if mode in {"target", "indicatorAndTarget"}:
        return "indicator"
    return mode
