"""Shared configuration constants for DiGiTerra."""
from pathlib import Path
import os

# Output directory for visualizations and generated files (plots, PDFs, predictions).
# app.py sets DIGITERRA_OUTPUT_DIR and calls update_vis_dir() so we stay in sync.
# Override via env: DIGITERRA_OUTPUT_DIR.
VIS_DIR = Path(os.environ.get("DIGITERRA_OUTPUT_DIR", "static/userVisualizations"))
VIS_DIR.mkdir(parents=True, exist_ok=True)


def update_vis_dir(new_path: Path):
    """Update VIS_DIR to match the app's USER_VIS_DIR. Called at startup from app.py."""
    global VIS_DIR
    VIS_DIR = Path(new_path)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
