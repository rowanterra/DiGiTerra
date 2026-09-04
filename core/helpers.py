"""Shared helpers for DiGiTerra routes (paths, URLs, validation)."""

from pathlib import Path

from python_scripts import config


def ensure_user_vis_dir() -> Path:
    """Return this session's output directory, creating it if needed."""
    return config.VIS_DIR.current()


class FrameTooLarge(ValueError):
    """The file is larger than the reader is allowed to parse or keep."""


TABLE_EXTENSIONS = {"csv", "xlsx", "xlsm", "parquet"}

# A compressed container is measured before it is opened. Its decoded size may be
# this many times the frame limit: sheet XML and Arrow buffers are larger than the
# pandas frame they become, so a tighter factor would refuse legitimate files.
DECODED_BUDGET_FACTOR = 8
# A zip member that inflates past this ratio is refused outright; ordinary
# spreadsheets compress 5 to 30 times, a crafted one thousands of times.
MAX_ZIP_RATIO = 200
MAX_ZIP_MEMBERS = 10_000


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


def estimate_decoded_bytes(filepath) -> int:
    """Bytes the file is expected to occupy once decoded, from metadata only.

    CSV: the file size, since a parsed frame is about the size of the text.
    Excel: the sum of the zip members' uncompressed sizes. Parquet: the sum of the
    row groups' uncompressed byte sizes from the footer. Nothing is decompressed.
    """
    ext = _table_extension(filepath)
    if ext == "csv":
        return int(Path(filepath).stat().st_size)
    if ext in ("xlsx", "xlsm"):
        import zipfile

        with zipfile.ZipFile(filepath) as z:
            return int(sum(info.file_size for info in z.infolist()))
    import pyarrow.parquet as pq

    md = pq.ParquetFile(filepath).metadata
    return int(sum(md.row_group(i).total_byte_size for i in range(md.num_row_groups)))


def _preflight_zip(filepath, max_decoded_bytes: int):
    """Refuse a workbook whose members would inflate past the budget, before opening it."""
    import zipfile

    with zipfile.ZipFile(filepath) as z:
        infos = z.infolist()
        if len(infos) > MAX_ZIP_MEMBERS:
            raise FrameTooLarge(f"The workbook has {len(infos):,} parts; the limit is {MAX_ZIP_MEMBERS:,}.")
        total = 0
        for info in infos:
            total += info.file_size
            if info.compress_size and info.file_size / info.compress_size > MAX_ZIP_RATIO and info.file_size > 1 << 20:
                raise FrameTooLarge(
                    f"Part {info.filename!r} of the workbook inflates {info.file_size / info.compress_size:.0f} times; "
                    f"the limit is {MAX_ZIP_RATIO}."
                )
        if max_decoded_bytes and total > max_decoded_bytes:
            raise FrameTooLarge(
                f"The workbook holds {total / (1024 * 1024):.0f} MB of sheet data once inflated; "
                f"the limit is {max_decoded_bytes / (1024 * 1024):.0f} MB."
            )


def _preflight_parquet(pf, max_decoded_bytes: int):
    """Refuse a Parquet file whose row groups decode past the budget, from the footer alone."""
    if not max_decoded_bytes:
        return
    md = pf.metadata
    total = sum(md.row_group(i).total_byte_size for i in range(md.num_row_groups))
    if total > max_decoded_bytes:
        mb, limit_mb = total / (1024 * 1024), max_decoded_bytes / (1024 * 1024)
        raise FrameTooLarge(f"The file decodes to {mb:.0f} MB; the limit is {limit_mb:.0f} MB.")


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


def _read_excel(filepath, max_cells: int, max_decoded_bytes: int = 0):
    """Read the first sheet of a workbook.

    The zip directory is checked first: member count, inflation ratio, and total
    inflated size against the decoded budget, so a workbook that would expand past
    the budget is refused without any part being inflated. openpyxl in read-only
    mode then streams the sheet XML; the header row is taken from the first row of
    ``iter_rows`` and pandas is asked for at most ``max_rows + 1`` rows, which it
    passes to openpyxl as ``file_rows_needed``. openpyxl still loads the shared
    string table in full, which is why the size check comes before the open.
    """
    import openpyxl
    import pandas as pd

    _preflight_zip(filepath, max_decoded_bytes)
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


def _read_parquet(filepath, max_cells: int, max_decoded_bytes: int = 0):
    """Read a Parquet file, stopping at the cell budget.

    The column count, row count, and uncompressed row-group sizes come from the
    footer, so a file over the cell budget or the decoded budget is refused with
    nothing decoded. Within budget, rows are read in batches of at most 65,536 and
    reading stops past the row cap, so one oversized row group cannot be
    materialized whole.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(filepath)
    n_cols = pf.metadata.num_columns
    _check_width(n_cols, max_cells)
    _preflight_parquet(pf, max_decoded_bytes)
    if not max_cells:
        return pf.read().to_pandas()
    max_rows = max(1, max_cells // n_cols)
    if pf.metadata.num_rows > max_rows:
        raise _too_many_rows(max_rows, n_cols, max_cells)
    batches = []
    rows_read = 0
    for batch in pf.iter_batches(batch_size=min(65_536, max_rows + 1)):
        rows_read += batch.num_rows
        batches.append(batch)
        if rows_read > max_rows:
            del batches
            raise _too_many_rows(max_rows, n_cols, max_cells)
    if not batches:
        return pf.schema_arrow.empty_table().to_pandas()
    return pa.Table.from_batches(batches).to_pandas()


def read_table_bounded(filepath, max_cells: int = 0, max_frame_mb: float = 0):
    """Read a CSV, Excel, or Parquet file without letting the parse exceed the cell budget.

    A full read allocates before any size check can run, so the column count is
    taken from the header (CSV, Excel) or the schema (Parquet) first and the row
    count is capped at the budget divided by that width. One extra row is read
    to tell a file at the limit from a file over the limit. The parsed frame is then
    measured, since object columns take far more bytes per cell than numeric ones.
    """
    ext = _table_extension(filepath)
    max_decoded = int(max_frame_mb * DECODED_BUDGET_FACTOR * 1024 * 1024) if max_frame_mb else 0
    if ext == "csv":
        data = _read_csv(filepath, max_cells)
    elif ext in ("xlsx", "xlsm"):
        data = _read_excel(filepath, max_cells, max_decoded)
    else:
        data = _read_parquet(filepath, max_cells, max_decoded)

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
