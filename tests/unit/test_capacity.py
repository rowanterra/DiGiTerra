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
    # The route parses through the worker now, which is the call that must not
    # happen when the estimate has already refused the file.
    monkeypatch.setattr(upload_route, "read_table_isolated", lambda *a, **k: calls.append(1))
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
    state.admit_model("mb-1", 1 << 20)
    assert state.total_data_bytes() >= 2 << 20
    monkeypatch.setattr(state, "_active_sessions_provider", lambda: {"mb-1"})
    with pytest.raises(state.MemoryBudgetExceeded):
        state.admit_frame("mb-2", _frame(1))
    # A new frame in mb-1 drops its recorded model size.
    state.admit_frame("mb-1", _frame(0.5))
    assert state.total_data_bytes() < 1 << 20
    for sid in ("mb-1", "mb-2"):
        state.clear_session_storage(sid)


def test_a_model_is_charged_beside_its_own_frame(monkeypatch):
    """The reproduction from the 2026-09-04 audit: frame plus model passed the budget together.

    _make_room excludes the whole session, which is right for a frame (it
    replaces what the session held) and wrong for a model (it is held beside the
    frame), so both were admitted and the recorded total ran past the limit.
    """
    state.clear_session_storage("pair-1")
    monkeypatch.setattr(state, "MAX_TOTAL_FRAME_MB", 1.0)  # 1 MiB
    state.admit_frame("pair-1", _frame(0.66))
    frame_charge = state.total_data_bytes()
    assert frame_charge > 0.6 * (1 << 20)
    with pytest.raises(state.MemoryBudgetExceeded):
        state.admit_model("pair-1", int(0.66 * (1 << 20)))
    assert state.total_data_bytes() == frame_charge  # the refusal charged nothing
    # A model that fits beside the frame is admitted and both are counted.
    state.admit_model("pair-1", 1000)
    assert state.total_data_bytes() == frame_charge + 1000
    state.clear_session_storage("pair-1")


def test_a_model_that_cannot_be_measured_is_discarded(monkeypatch):
    """A size that cannot be computed cannot be charged, so the model is not kept."""
    import pandas as pd

    import python_scripts.app_model_training as amt
    from python_scripts import config
    from python_scripts.app_model_training import run_model_training
    from python_scripts.preprocessing.progress_tracker import get_result, remove_tracker
    from tests.unit.test_pipelines import BASE

    def _raise(_store):
        raise RuntimeError("no size for this estimator")

    monkeypatch.setattr(amt, "serialized_size", _raise)
    sid = "unmeasurable-model"
    state.clear_session_storage(sid)
    store = state.get_session_storage(sid)
    store["data"] = pd.read_csv("examples/diabetes.csv")
    n = len(store["data"].columns)
    payload = dict(BASE, indicators=list(range(n - 1)), predictors=[n - 1], models="Linear")
    token = config.set_session_scope(sid)
    try:
        run_model_training(sid, payload, sid, lambda s: state.get_session_storage(s))
        result = get_result(sid)
    finally:
        config.reset_session_scope(token)
    remove_tracker(sid)
    assert result is not None and "error" not in result
    assert result["model_retained"] is False
    assert "could not be measured" in result["capacity_note"]
    assert state.get_session_storage(sid).get("model") is None
    state.clear_session_storage(sid)


def test_an_oversized_model_is_not_retained(monkeypatch):
    """The fit succeeds, the scores stand, the model is dropped rather than kept over budget."""
    import pandas as pd

    from python_scripts import config
    from python_scripts.app_model_training import run_model_training
    from python_scripts.preprocessing.progress_tracker import get_result, remove_tracker
    from tests.unit.test_pipelines import BASE

    sid = "oversized-model"
    state.clear_session_storage(sid)
    state.get_session_storage(sid)
    # Charged through admit_frame, as an upload is, so the frame counts against
    # the budget when the model is measured.
    state.admit_frame(sid, pd.read_csv("examples/diabetes.csv"))
    store = state.get_session_storage(sid)
    n = len(store["data"].columns)
    payload = dict(BASE, indicators=list(range(n - 1)), predictors=[n - 1], models="Linear")
    # A budget below the frame alone, so the frame plus any fitted model is over it.
    monkeypatch.setattr(state, "MAX_TOTAL_FRAME_MB", 0.01)
    token = config.set_session_scope(sid)
    try:
        run_model_training(sid, payload, sid, lambda s: state.get_session_storage(s))
        result = get_result(sid)
    finally:
        config.reset_session_scope(token)
    remove_tracker(sid)
    assert result is not None and "error" not in result
    assert result["valscore"]  # the run itself completed
    assert result["model_retained"] is False
    assert "not available" in result["capacity_note"]
    assert state.get_session_storage(sid).get("model") is None
    state.clear_session_storage(sid)
