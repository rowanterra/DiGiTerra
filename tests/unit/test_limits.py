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


def test_bounded_reader_stops_at_the_budget(tmp_path):
    """The reader must not parse the whole file before deciding it is too large."""
    import tracemalloc

    from core.helpers import FrameTooLarge, read_csv_bounded

    path = tmp_path / "wide.csv"
    pd.DataFrame({f"c{i}": range(60_000) for i in range(10)}).to_csv(path, index=False)

    tracemalloc.start()
    with pytest.raises(FrameTooLarge):
        read_csv_bounded(str(path), max_cells=10_000)
    bounded_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    tracemalloc.start()
    pd.read_csv(path)
    full_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    assert bounded_peak < full_peak / 4
    assert read_csv_bounded(str(path), max_cells=10_000_000).shape == (60_000, 10)


def test_bounded_reader_rejects_a_frame_over_the_byte_budget(tmp_path):
    from core.helpers import FrameTooLarge, read_csv_bounded

    path = tmp_path / "strings.csv"
    pd.DataFrame({"a": ["x" * 500] * 2000}).to_csv(path, index=False)
    with pytest.raises(FrameTooLarge, match="once parsed"):
        read_csv_bounded(str(path), max_cells=0, max_frame_mb=0.2)


def test_out_of_range_value_is_refused_before_a_thread_starts(client):
    """The route answers 400 without creating a training id."""
    r = client.post("/upload", data={"file": (_csv_bytes(rows=30, cols=4), "d.csv")})
    assert r.status_code == 200
    r = client.post("/preprocess", json={"indicators": [0, 1, 2], "predictors": [3], "stratify": 0})
    assert r.status_code == 200
    from tests.unit.test_pipelines import BASE

    payload = dict(BASE, indicators=[0, 1, 2], predictors=[3], models="Linear", searchNIter=100000)
    r = client.post("/process", json=payload)
    assert r.status_code == 400
    assert "searchNIter" in r.get_json()["error"]
    assert "session_id" not in r.get_json()


def test_second_concurrent_run_for_a_session_is_refused():
    from core import jobs

    jobs.acquire("session-a", "run-1")
    try:
        with pytest.raises(jobs.SlotUnavailable, match="already training"):
            jobs.acquire("session-a", "run-2")
        assert jobs.active_sessions() == {"session-a"}
    finally:
        jobs.release("session-a", "run-1")
    assert jobs.active_sessions() == set()


def test_global_training_slots_are_capped(monkeypatch):
    from core import jobs

    monkeypatch.setattr(jobs, "MAX_CONCURRENT_TRAINING", 2)
    jobs.acquire("s1", "r1")
    jobs.acquire("s2", "r2")
    try:
        with pytest.raises(jobs.SlotUnavailable, match="maximum number"):
            jobs.acquire("s3", "r3")
    finally:
        jobs.release("s1", "r1")
        jobs.release("s2", "r2")


def test_sweep_keeps_a_directory_whose_session_is_running(monkeypatch):
    base = config.VIS_DIR.base
    running = base / ("e" * 32)
    idle = base / ("f" * 32)
    for d in (running, idle):
        d.mkdir(exist_ok=True)
        two_days_ago = time.time() - 48 * 3600
        os.utime(d, (two_days_ago, two_days_ago))
    monkeypatch.setattr(config, "STALE_SESSION_HOURS", 24.0)
    monkeypatch.setattr(config, "_active_session_provider", [lambda: {"e" * 32}])
    try:
        config.sweep_stale_outputs(force=True)
        assert running.exists(), "a session with a run in flight was swept"
        assert not idle.exists()
    finally:
        if running.exists():
            running.rmdir()


def test_using_a_session_directory_refreshes_its_age(monkeypatch):
    base = config.VIS_DIR.base
    used = base / ("1" * 32)
    used.mkdir(exist_ok=True)
    two_days_ago = time.time() - 48 * 3600
    os.utime(used, (two_days_ago, two_days_ago))
    monkeypatch.setattr(config, "_last_touch", {})
    token = config.set_session_scope("1" * 32)
    try:
        _ = config.VIS_DIR / "anything.png"
        assert used.stat().st_mtime > two_days_ago + 3600
    finally:
        config.reset_session_scope(token)
        used.rmdir()
