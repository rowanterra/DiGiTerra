"""Excel and Parquet import, Excel export of predictions.

The bounded reader in core/helpers.py dispatches on extension. These tests upload
each format through the Flask test client, check that a Parquet file over the cell
budget is refused without a full read, and check that the predictions export
route answers with a workbook openpyxl can open.
"""

import io
import warnings

import pandas as pd
import pytest

from python_scripts import config

warnings.filterwarnings("ignore")


@pytest.fixture
def client():
    from app import app

    app.config["TESTING"] = True
    return app.test_client()


def _frame(rows=20, cols=4):
    return pd.DataFrame({f"c{i}": range(rows) for i in range(cols)})


def _xlsx_bytes(rows=20, cols=4):
    buf = io.BytesIO()
    _frame(rows, cols).to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf


def _parquet_bytes(rows=20, cols=4):
    buf = io.BytesIO()
    _frame(rows, cols).to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)
    return buf


def test_allowed_file_accepts_the_four_formats():
    from core.helpers import allowed_file

    assert all(allowed_file(f"d.{ext}") for ext in ("csv", "xlsx", "xlsm", "parquet", "XLSX"))
    assert not allowed_file("d.txt") and not allowed_file("d")


def test_upload_xlsx(client):
    r = client.post("/upload", data={"file": (_xlsx_bytes(rows=25, cols=5), "d.xlsx")})
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body["numcols"] == 5 and body["rows"] == 25


def test_upload_parquet(client):
    r = client.post("/upload", data={"file": (_parquet_bytes(rows=30, cols=6), "d.parquet")})
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body["numcols"] == 6 and body["rows"] == 30


def test_upload_rejects_an_unknown_extension(client):
    r = client.post("/upload", data={"file": (io.BytesIO(b"a,b\n1,2\n"), "d.txt")})
    assert r.status_code == 400
    assert ".parquet" in r.get_json()["error"]


def test_read_csv_bounded_alias_still_works(tmp_path):
    from core.helpers import read_csv_bounded, read_table_bounded

    path = tmp_path / "d.csv"
    _frame(5, 3).to_csv(path, index=False)
    assert read_csv_bounded(str(path)).shape == read_table_bounded(str(path)).shape == (5, 3)


def test_bounded_reader_rejects_an_unsupported_extension(tmp_path):
    from core.helpers import read_table_bounded

    path = tmp_path / "d.txt"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="Unsupported file type"):
        read_table_bounded(str(path))


def test_bounded_parquet_reader_stops_at_the_budget(tmp_path):
    """A Parquet file over the budget is refused from the schema, not after a full read."""
    import tracemalloc

    from core.helpers import FrameTooLarge, read_table_bounded

    path = tmp_path / "wide.parquet"
    _frame(60_000, 10).to_parquet(path, index=False, engine="pyarrow", row_group_size=5_000)

    tracemalloc.start()
    with pytest.raises(FrameTooLarge, match="more than"):
        read_table_bounded(str(path), max_cells=10_000)
    bounded_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    tracemalloc.start()
    pd.read_parquet(path, engine="pyarrow")
    full_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    assert bounded_peak < full_peak / 4
    assert read_table_bounded(str(path), max_cells=10_000_000).shape == (60_000, 10)


def test_bounded_parquet_reader_rejects_too_many_columns(tmp_path):
    from core.helpers import FrameTooLarge, read_table_bounded

    path = tmp_path / "cols.parquet"
    _frame(2, 12).to_parquet(path, index=False, engine="pyarrow")
    with pytest.raises(FrameTooLarge, match="columns"):
        read_table_bounded(str(path), max_cells=10)


def test_bounded_excel_reader_stops_at_the_budget(tmp_path):
    """pandas passes nrows to openpyxl as file_rows_needed and the row iterator stops there."""
    import tracemalloc

    from core.helpers import FrameTooLarge, read_table_bounded

    path = tmp_path / "wide.xlsx"
    _frame(20_000, 10).to_excel(path, index=False, engine="openpyxl")

    tracemalloc.start()
    with pytest.raises(FrameTooLarge, match="more than"):
        read_table_bounded(str(path), max_cells=1_000)
    bounded_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    tracemalloc.start()
    pd.read_excel(path, engine="openpyxl")
    full_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    assert bounded_peak < full_peak / 2
    assert read_table_bounded(str(path), max_cells=10_000_000).shape == (20_000, 10)


def test_bounded_excel_reader_reads_the_first_sheet_at_the_limit(tmp_path):
    from core.helpers import read_table_bounded

    path = tmp_path / "two.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        _frame(10, 3).to_excel(writer, sheet_name="first", index=False)
        _frame(50, 8).to_excel(writer, sheet_name="second", index=False)
    assert read_table_bounded(str(path), max_cells=30).shape == (10, 3)


def test_predictions_xlsx_export(client):
    import uuid

    import openpyxl

    sid = uuid.uuid4().hex
    client.set_cookie("digiterra_session", sid)
    r = client.get("/download/predictions.xlsx")
    assert r.status_code == 404

    token = config.set_session_scope(sid)
    try:
        vis_dir = config.VIS_DIR.current()
        _frame(4, 2).assign(prediction=[0.5, 1.5, 2.5, 3.5]).to_csv(vis_dir / "predictions.csv", index=False)
    finally:
        config.reset_session_scope(token)

    r = client.get("/download/predictions.xlsx")
    assert r.status_code == 200, r.get_json()
    assert r.headers["Content-Disposition"].endswith("filename=predictions.xlsx")
    book = openpyxl.load_workbook(io.BytesIO(r.data), read_only=True)
    sheet = book.worksheets[0]
    rows = list(sheet.iter_rows(values_only=True))
    book.close()
    assert rows[0] == ("c0", "c1", "prediction")
    assert len(rows) == 5


def _zip_bomb_xlsx(tmp_path, inflated_mb: int):
    """A real workbook whose sheet XML is padded with a highly compressible run."""
    import zipfile

    src = tmp_path / "small.xlsx"
    _frame(5, 3).to_excel(src, index=False, engine="openpyxl")
    out = tmp_path / "bomb.xlsx"
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            payload = zin.read(info.filename)
            if info.filename.startswith("xl/worksheets/sheet"):
                payload = payload.replace(b"</worksheet>", b"<!--" + b"A" * (inflated_mb << 20) + b"--></worksheet>")
            zout.writestr(info.filename, payload)
    return out


def test_workbook_that_inflates_past_the_budget_is_refused_before_opening(tmp_path):
    import tracemalloc

    from core.helpers import FrameTooLarge, read_table_bounded

    path = _zip_bomb_xlsx(tmp_path, inflated_mb=40)
    assert path.stat().st_size < 1 << 20  # under a megabyte on disk
    tracemalloc.start()
    with pytest.raises(FrameTooLarge, match="inflates|inflated"):
        read_table_bounded(path, max_cells=1_000_000, max_frame_mb=1)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # The 40 MB comment was never inflated.
    assert peak < 8 << 20, f"peak {peak / 2**20:.1f} MB"


def test_parquet_over_the_decoded_budget_is_refused_from_the_footer(tmp_path):
    import tracemalloc

    import numpy as np

    from core.helpers import FrameTooLarge, estimate_decoded_bytes, read_table_bounded

    rows = 600_000
    df = pd.DataFrame({"a": np.arange(rows, dtype="int64"), "b": np.arange(rows, dtype="float64")})
    path = tmp_path / "wide.parquet"
    df.to_parquet(path, index=False, engine="pyarrow")
    assert estimate_decoded_bytes(path) > 9 << 20
    tracemalloc.start()
    with pytest.raises(FrameTooLarge, match="decodes to"):
        # 1 MB frame budget times the factor of 8 is below the 9.6 MB the file decodes to.
        read_table_bounded(path, max_cells=0, max_frame_mb=1)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 2 << 20, f"peak {peak / 2**20:.1f} MB"
    # Within budget the same file reads in batches.
    assert len(read_table_bounded(path, max_cells=0, max_frame_mb=64)) == rows


def test_estimate_decoded_bytes_by_format(tmp_path):
    from core.helpers import estimate_decoded_bytes

    csv = tmp_path / "x.csv"
    _frame(100, 3).to_csv(csv, index=False)
    assert estimate_decoded_bytes(csv) == csv.stat().st_size
    xlsx = tmp_path / "x.xlsx"
    _frame(100, 3).to_excel(xlsx, index=False, engine="openpyxl")
    assert estimate_decoded_bytes(xlsx) > xlsx.stat().st_size
