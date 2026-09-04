"""Process-wide capacity controls: progress-stream slots, the data budget, cores per job.

These answer the second implementation review: eight open streams could hold
every gthread request thread; thirty-two rotated cookies could hold thirty-two
near-limit frames; two runs could each ask for every core.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from core import jobs, state
from python_scripts.preprocessing.progress_tracker import get_tracker, remove_tracker

warnings.filterwarnings("ignore")


@pytest.fixture
def client():
    from app import app

    app.config["TESTING"] = True
    return app.test_client()


def test_stream_slots_are_capped_and_released(client, monkeypatch):
    monkeypatch.setattr(jobs, "MAX_CONCURRENT_STREAMS", 2)
    owner = "a" * 32
    ids = ["cap-run-1", "cap-run-2", "cap-run-3"]
    for i in ids:
        get_tracker(i, owner_session_id=owner).start()
    client.set_cookie("digiterra_session", owner)
    open_streams = []
    try:
        r1 = client.get(f"/progress/{ids[0]}", buffered=False)
        r2 = client.get(f"/progress/{ids[1]}", buffered=False)
        open_streams += [r1, r2]
        assert r1.status_code == 200 and r2.status_code == 200
        # A second stream on a run that already has one is refused.
        assert client.get(f"/progress/{ids[0]}").status_code == 429
        # The process-wide cap refuses a third run's stream while two are open.
        assert client.get(f"/progress/{ids[2]}").status_code == 429
        # Unrelated requests still answer.
        assert client.get("/").status_code == 200
        # Closing a stream frees its slot, whether or not the body was ever read.
        # (Streams close in reverse order of opening because the test client
        # stacks their request contexts.)
        r2.close()
        open_streams.remove(r2)
        assert jobs.stream_count() == 1
        r3 = client.get(f"/progress/{ids[2]}", buffered=False)
        open_streams.append(r3)
        assert r3.status_code == 200
        assert jobs.stream_count() == 2
    finally:
        for r in reversed(open_streams):
            r.close()
        for i in ids:
            remove_tracker(i)
    assert jobs.stream_count() == 0


def _frame(mb: float) -> pd.DataFrame:
    n = int(mb * 1024 * 1024 / 8)
    return pd.DataFrame({"x": np.zeros(n)})


def test_total_frame_budget_evicts_idle_sessions_and_refuses_when_full(monkeypatch):
    monkeypatch.setattr(state, "MAX_TOTAL_FRAME_MB", 2.5)
    for sid in ("budget-1", "budget-2", "budget-3", "budget-4"):
        state.clear_session_storage(sid)
    state.admit_frame("budget-1", _frame(1))
    state.admit_frame("budget-2", _frame(1))
    assert state.total_data_bytes() >= 2 * 1024 * 1024
    # The third frame does not fit beside both; the oldest idle session goes.
    state.admit_frame("budget-3", _frame(1))
    assert "budget-1" not in state._STORE and "budget-2" in state._STORE and "budget-3" in state._STORE
    # A session with a running job is kept, so a frame that cannot fit is refused.
    monkeypatch.setattr(state, "_active_sessions_provider", lambda: {"budget-2", "budget-3"})
    with pytest.raises(state.MemoryBudgetExceeded):
        state.admit_frame("budget-4", _frame(1))
    # A frame larger than the whole budget is refused on its own.
    monkeypatch.setattr(state, "_active_sessions_provider", lambda: set())
    with pytest.raises(state.MemoryBudgetExceeded):
        state.admit_frame("budget-4", _frame(3))
    for sid in ("budget-1", "budget-2", "budget-3", "budget-4"):
        state.clear_session_storage(sid)


def test_upload_over_the_total_budget_returns_429(client, monkeypatch):
    import io

    monkeypatch.setattr(state, "MAX_TOTAL_FRAME_MB", 0.001)
    client.set_cookie("digiterra_session", "e" * 32)
    csv = io.BytesIO(_frame(0.01).to_csv(index=False).encode())
    r = client.post("/upload", data={"file": (csv, "big.csv")})
    assert r.status_code == 429
    assert "maximum" in r.get_json()["error"]


def test_session_cap_never_evicts_a_running_job(monkeypatch):
    monkeypatch.setattr(state, "MAX_SESSIONS", 2)
    for sid in ("cap-a", "cap-b", "cap-c"):
        state.clear_session_storage(sid)
    state.get_session_storage("cap-a")["marker"] = 1
    state.get_session_storage("cap-b")
    monkeypatch.setattr(state, "_active_sessions_provider", lambda: {"cap-a"})
    state.get_session_storage("cap-c")
    assert "cap-a" in state._STORE and "cap-b" not in state._STORE
    for sid in ("cap-a", "cap-b", "cap-c"):
        state.clear_session_storage(sid)


def test_cores_per_job_split_the_host(monkeypatch):
    monkeypatch.delenv("DIGITERRA_JOB_CORES", raising=False)
    monkeypatch.setattr(jobs, "MAX_CONCURRENT_TRAINING", 2)
    monkeypatch.setattr(jobs.os, "cpu_count", lambda: 8)
    assert jobs.job_n_jobs() == 4
    monkeypatch.setattr(jobs, "MAX_CONCURRENT_TRAINING", 16)
    assert jobs.job_n_jobs() == 1
    monkeypatch.setenv("DIGITERRA_JOB_CORES", "3")
    assert jobs.job_n_jobs() == 3


def test_upload_is_refused_from_the_estimate_before_parsing(client, monkeypatch):
    """An over-budget file is turned away on its metadata; the parser is never called."""
    import io

    import routes.upload as upload_route

    calls = []
    monkeypatch.setattr(
        upload_route, "read_table_bounded", lambda *a, **k: calls.append(1) or upload_route.read_table_bounded
    )
    monkeypatch.setattr(state, "MAX_TOTAL_FRAME_MB", 0.0005)  # about 500 bytes
    client.set_cookie("digiterra_session", "f" * 32)
    csv = io.BytesIO(_frame(0.01).to_csv(index=False).encode())
    r = client.post("/upload", data={"file": (csv, "big.csv")})
    assert r.status_code == 429
    assert calls == []


def test_model_bytes_count_toward_the_budget(monkeypatch):
    for sid in ("mb-1", "mb-2"):
        state.clear_session_storage(sid)
    monkeypatch.setattr(state, "MAX_TOTAL_FRAME_MB", 2.5)
    state.admit_frame("mb-1", _frame(1))
    state.record_model_bytes("mb-1", 1 << 20)
    assert state.total_data_bytes() >= 2 << 20
    monkeypatch.setattr(state, "_active_sessions_provider", lambda: {"mb-1"})
    with pytest.raises(state.MemoryBudgetExceeded):
        state.admit_frame("mb-2", _frame(1))
    # A new frame in mb-1 drops its recorded model size.
    state.admit_frame("mb-1", _frame(0.5))
    assert state.total_data_bytes() < 1 << 20
    for sid in ("mb-1", "mb-2"):
        state.clear_session_storage(sid)
