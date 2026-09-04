"""Shared helpers for DiGiTerra routes (paths, URLs, validation)."""

from pathlib import Path

from python_scripts import config


def ensure_user_vis_dir() -> Path:
    """Return this session's output directory, creating it if needed."""
    return config.VIS_DIR.current()


class FrameTooLarge(ValueError):
    """The file is larger than the reader is allowed to parse or keep."""


def read_csv_bounded(filepath, max_cells: int = 0, max_frame_mb: float = 0):
    """Read a CSV without letting the parse itself exceed the cell budget.

    pandas.read_csv over a whole file allocates before any size check can run, so
    the column count is taken from the header first and the row count is capped at
    the budget divided by that width. One extra row is read to tell a file at the
    limit from a file over it. The parsed frame is then measured, since object
    columns hold far more bytes per cell than numeric ones.
    """
    import pandas as pd

    header = pd.read_csv(filepath, nrows=0)
    n_cols = len(header.columns)
    if n_cols == 0:
        raise pd.errors.EmptyDataError("No columns to parse from file")

    if max_cells and n_cols > max_cells:
        raise FrameTooLarge(f"The file has {n_cols:,} columns; the limit is {max_cells:,} cells.")

    if max_cells:
        max_rows = max(1, max_cells // n_cols)
        data = pd.read_csv(filepath, nrows=max_rows + 1)
        if len(data) > max_rows:
            del data
            raise FrameTooLarge(
                f"The file has more than {max_rows:,} rows across {n_cols} columns; the limit is {max_cells:,} cells."
            )
    else:
        data = pd.read_csv(filepath)

    if max_frame_mb:
        frame_mb = float(data.memory_usage(index=True, deep=True).sum()) / (1024 * 1024)
        if frame_mb > max_frame_mb:
            del data
            raise FrameTooLarge(f"The file uses {frame_mb:.0f} MB once parsed; the limit is {max_frame_mb:.0f} MB.")
    return data


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
