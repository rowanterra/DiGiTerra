"""Server-side limits, cleanup, and the escaping regression.

These lock in the changes made from the 2026-09-03 handoff findings: uploads are
deleted once parsed, oversized frames are rejected before they are stored,
request values that schedule work are bounded, a progress stream answers only
the session that started the run, stale session directories are swept, and the
echoed upload filename carries no markup.
"""

import io
import os
import time
import warnings
from pathlib import Path

import pandas as pd
import pytest

from python_scripts import config
from python_scripts.preprocessing.progress_tracker import get_tracker, remove_tracker
from tests.unit.test_pipelines import _train

warnings.filterwarnings("ignore")


@pytest.fixture
def client():
    from app import app

    app.config["TESTING"] = True
    return app.test_client()


def _csv_bytes(rows=20, cols=3):
    df = pd.DataFrame({f"c{i}": range(rows) for i in range(cols)})
    return io.BytesIO(df.to_csv(index=False).encode())


def test_upload_temp_file_is_removed(client):
    from core.flask_app import UPLOAD_DIR

    before = {p.name for p in Path(UPLOAD_DIR).glob("*.csv")}
    r = client.post("/upload", data={"file": (_csv_bytes(), "clean.csv")})
    assert r.status_code == 200
    after = {p.name for p in Path(UPLOAD_DIR).glob("*.csv")}
    assert after - before == set()


def test_prediction_temp_file_is_removed(client, monkeypatch):
    """The prediction route saves and parses first, so a store with no model still exercises the cleanup."""
    from core.flask_app import UPLOAD_DIR

    before = {p.name for p in Path(UPLOAD_DIR).glob("*.csv")}
    client.post("/predict", data={"predictFile": (_csv_bytes(), "new.csv")})
    after = {p.name for p in Path(UPLOAD_DIR).glob("*.csv")}
    assert after - before == set()


def test_frame_above_the_cell_cap_is_rejected(client, monkeypatch):
    import routes.upload as upload_route

    monkeypatch.setattr(upload_route, "MAX_CELLS", 50)
    r = client.post("/upload", data={"file": (_csv_bytes(rows=20, cols=3), "big.csv")})
    assert r.status_code == 400
    assert "cells" in r.get_json()["error"]
    r = client.post("/upload", data={"file": (_csv_bytes(rows=10, cols=3), "small.csv")})
    assert r.status_code == 200


def test_echoed_filename_carries_no_markup(client):
    r = client.post("/upload", data={"file": (_csv_bytes(), "<img src=x onerror=alert(1)>.csv")})
    assert r.status_code == 200
    assert "<" not in r.get_json()["filename"]


@pytest.mark.parametrize(
    "field, value",
    [("testSize", "0.99"), ("crossValidationFolds", 500), ("searchCVFolds", 50), ("searchNIter", 100000)],
)
def test_out_of_range_training_values_are_rejected(field, value):
    store = {"data": pd.read_csv("examples/diabetes.csv")}
    n = len(store["data"].columns)
    from python_scripts.app_model_training import run_model_training
    from python_scripts.preprocessing.progress_tracker import get_result

    sid = f"pytest-limits-{field}"
    from tests.unit.test_pipelines import BASE

    payload = dict(BASE, indicators=list(range(n - 1)), predictors=[n - 1], models="Linear")
    payload[field] = value
    if field == "crossValidationFolds":
        payload["crossValidationType"] = "KFold"
    run_model_training(sid, payload, "store", lambda _s: store)
    result = get_result(sid)
    remove_tracker(sid)
    assert result is not None and "error" in result
    assert field in result["error"]


def test_in_range_training_values_still_train():
    store = {}
    n = len(pd.read_csv("examples/diabetes.csv").columns)
    result, _ = _train(store, "examples/diabetes.csv", list(range(n - 1)), [n - 1], "Linear", testSize="0.30")
    assert "error" not in result


def test_progress_stream_answers_only_the_owner(client):
    tracker = get_tracker("owned-run", owner_session_id="a" * 32)
    tracker.start()
    try:
        client.set_cookie("digiterra_session", "b" * 32)
        assert client.get("/progress/owned-run").status_code == 404
        client.set_cookie("digiterra_session", "a" * 32)
        r = client.get("/progress/owned-run", buffered=False)
        assert r.status_code == 200
        r.close()
    finally:
        remove_tracker("owned-run")


def test_stale_session_directories_are_swept(monkeypatch):
    base = config.VIS_DIR.base
    old = base / ("c" * 32)
    fresh = base / ("d" * 32)
    old.mkdir(exist_ok=True)
    fresh.mkdir(exist_ok=True)
    (old / "predictions.csv").write_text("x")
    two_days_ago = time.time() - 48 * 3600
    os.utime(old, (two_days_ago, two_days_ago))
    monkeypatch.setattr(config, "STALE_SESSION_HOURS", 24.0)
    removed = config.sweep_stale_outputs(force=True)
    assert removed >= 1
    assert not old.exists() and fresh.exists()
    fresh.rmdir()
