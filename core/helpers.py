"""Shared helpers for DiGiTerra routes (paths, URLs, validation)."""

from pathlib import Path

from python_scripts import config


def ensure_user_vis_dir() -> Path:
    """Return this session's output directory, creating it if needed."""
    return config.VIS_DIR.current()


class FrameTooLarge(ValueError):
    """The file is larger than the reader is allowed to parse or keep."""


TABLE_EXTENSIONS = {"csv", "xlsx", "xlsm", "parquet"}


def _table_extension(filepath) -> str:
    ext = Path(str(filepath)).suffix.lower().lstrip(".")
    if ext not in TABLE_EXTENSIONS:
        raise ValueError(f"Unsupported file type '.{ext}'. Accepted types: .csv, .xlsx, .xlsm, .parquet.")
    return ext


def _check_width(n_cols: int, max_cells: int):
    if n_cols == 0:
        import pandas as pd

        raise pd.errors.EmptyDataError("No columns to parse from file")
    if max_cells and n_cols > max_cells:
        raise FrameTooLarge(f"The file has {n_cols:,} columns; the limit is {max_cells:,} cells.")


def _too_many_rows(max_rows: int, n_cols: int, max_cells: int) -> FrameTooLarge:
    return FrameTooLarge(
        f"The file has more than {max_rows:,} rows across {n_cols} columns; the limit is {max_cells:,} cells."
    )


def _read_csv(filepath, max_cells: int):
    import pandas as pd

    header = pd.read_csv(filepath, nrows=0)
    n_cols = len(header.columns)
    _check_width(n_cols, max_cells)
    if not max_cells:
        return pd.read_csv(filepath)
    max_rows = max(1, max_cells // n_cols)
    data = pd.read_csv(filepath, nrows=max_rows + 1)
    if len(data) > max_rows:
        del data
        raise _too_many_rows(max_rows, n_cols, max_cells)
    return data


def _read_excel(filepath, max_cells: int):
    """Read the first sheet of a workbook.

    openpyxl in read-only mode streams the sheet XML, so the header row is taken
    from the first row of ``iter_rows`` and pandas is then asked for at most
    ``max_rows + 1`` rows. pandas passes that count to openpyxl as
    ``file_rows_needed`` and stops the row iterator there (verified against
    pandas.io.excel._openpyxl.get_sheet_data). openpyxl still loads the shared
    string table of the workbook in full, so the bound covers rows and not the
    string table.
    """
    import openpyxl
    import pandas as pd

    book = openpyxl.load_workbook(filepath, read_only=True, data_only=True, keep_links=False)
    try:
        sheet = book.worksheets[0]
        first = next(sheet.iter_rows(min_row=1, max_row=1, values_only=True), ())
        n_cols = len([v for v in first if v is not None]) if first else 0
    finally:
        book.close()
    _check_width(n_cols, max_cells)
    if not max_cells:
        return pd.read_excel(filepath, sheet_name=0, engine="openpyxl")
    max_rows = max(1, max_cells // n_cols)
    data = pd.read_excel(filepath, sheet_name=0, engine="openpyxl", nrows=max_rows + 1)
    if len(data) > max_rows:
        del data
        raise _too_many_rows(max_rows, n_cols, max_cells)
    return data


def _read_parquet(filepath, max_cells: int):
    """Read a Parquet file, stopping at the cell budget.

    The column count comes from the file schema. Row groups are read one at a
    time and reading stops as soon as the rows read exceed the budget, so a file
    over the budget costs at most one row group past the limit.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(filepath)
    n_cols = pf.metadata.num_columns
    _check_width(n_cols, max_cells)
    if not max_cells:
        return pf.read().to_pandas()
    max_rows = max(1, max_cells // n_cols)
    if pf.metadata.num_rows > max_rows:
        raise _too_many_rows(max_rows, n_cols, max_cells)
    tables = []
    rows_read = 0
    for i in range(pf.num_row_groups):
        table = pf.read_row_group(i)
        rows_read += table.num_rows
        tables.append(table)
        if rows_read > max_rows:
            del tables
            raise _too_many_rows(max_rows, n_cols, max_cells)
    import pyarrow as pa

    return pa.concat_tables(tables).to_pandas() if tables else pf.read().to_pandas()


def read_table_bounded(filepath, max_cells: int = 0, max_frame_mb: float = 0):
    """Read a CSV, Excel, or Parquet file without letting the parse exceed the cell budget.

    A full read allocates before any size check can run, so the column count is
    taken from the header (CSV, Excel) or the schema (Parquet) first and the row
    count is capped at the budget divided by that width. One extra row is read
    to tell a file at the limit from a file over the limit. The parsed frame is then
    measured, since object columns take far more bytes per cell than numeric ones.
    """
    ext = _table_extension(filepath)
    if ext == "csv":
        data = _read_csv(filepath, max_cells)
    elif ext in ("xlsx", "xlsm"):
        data = _read_excel(filepath, max_cells)
    else:
        data = _read_parquet(filepath, max_cells)

    if max_frame_mb:
        frame_mb = float(data.memory_usage(index=True, deep=True).sum()) / (1024 * 1024)
        if frame_mb > max_frame_mb:
            del data
            raise FrameTooLarge(f"The file uses {frame_mb:.0f} MB once parsed; the limit is {max_frame_mb:.0f} MB.")
    return data


def read_csv_bounded(filepath, max_cells: int = 0, max_frame_mb: float = 0):
    """Alias of ``read_table_bounded`` kept for existing imports."""
    return read_table_bounded(filepath, max_cells=max_cells, max_frame_mb=max_frame_mb)


def with_prefix(path: str) -> str:
    """Return path with URL_PREFIX applied."""
    normalized = path if path.startswith("/") else f"/{path}"
    if not config.URL_PREFIX:
        return normalized
    return f"{config.URL_PREFIX}{normalized}"


def allowed_file(filename: str) -> bool:
    """Return True when the filename extension is one of ALLOWED_EXTENSIONS (csv, xlsx, xlsm, parquet)."""
    from core.constants import ALLOWED_EXTENSIONS

    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def normalize_predict_preprocess_mode(mode: str) -> str:
    """Map training-time preprocessing modes to prediction-safe modes."""
    if mode in {"target", "indicatorAndTarget"}:
        return "indicator"
    return mode
