"""The training queue: order, an atomic claim, and recovery after a restart.

Admission control refuses a run when the slots are full. That is right on a
desktop and wrong on a shared installation, where the answer should be that the
run is waiting. These tests cover what the queue promises: first in first out, one
worker per job however many race for it, and a job whose worker died going back to
the queue rather than sitting as running for ever.
"""

import json
import subprocess
import sys
import threading
import uuid
import warnings
from pathlib import Path

import pytest

from core import queue

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("DIGITERRA_QUEUE_PATH", str(tmp_path / "queue.sqlite3"))
    queue.reset_for_tests()
    yield tmp_path / "queue.sqlite3"
    queue.reset_for_tests()


def _job(session="s1", **request):
    jid = uuid.uuid4().hex
    queue.enqueue(jid, session, request or {"models": "Linear"})
    return jid


def test_jobs_come_back_in_the_order_they_arrived():
    first, second, third = _job(), _job(), _job()
    assert queue.depth() == 3
    assert queue.position(first) == 0 and queue.position(third) == 2
    assert queue.claim("w1")["id"] == first
    assert queue.claim("w1")["id"] == second
    assert queue.claim("w1")["id"] == third
    assert queue.claim("w1") is None


def test_the_request_comes_back_as_it_was_recorded():
    jid = _job(models="RF", testSize="0.25")
    got = queue.claim("w1")
    assert got["id"] == jid and got["request"]["models"] == "RF"
    assert got["request"]["testSize"] == "0.25"
    assert got["state"] == queue.RUNNING


def test_two_workers_racing_for_one_job_produce_one_winner():
    _job()
    taken = []
    barrier = threading.Barrier(8)

    def race(n):
        barrier.wait()
        taken.append(queue.claim(f"w{n}"))

    threads = [threading.Thread(target=race, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    got = [t for t in taken if t is not None]
    assert len(got) == 1, f"{len(got)} workers claimed the same job"


def test_a_job_whose_worker_stopped_returns_to_the_queue():
    jid = _job()
    queue.claim("w1")
    assert queue.get(jid)["state"] == queue.RUNNING
    # Nothing is producing its result, so after the stale window it waits again.
    assert queue.requeue_lost(stale_seconds=-1) == 1
    assert queue.get(jid)["state"] == queue.QUEUED
    assert queue.claim("w2")["id"] == jid
    # A job with a fresh heartbeat is left alone.
    queue.heartbeat(jid)
    assert queue.requeue_lost(stale_seconds=3600) == 0


def test_a_finished_job_stays_finished():
    jid = _job()
    queue.claim("w1")
    queue.finish(jid, queue.DONE)
    assert queue.get(jid)["state"] == queue.DONE
    assert queue.requeue_lost(stale_seconds=-1) == 0
    assert queue.claim("w1") is None
    assert not queue.cancel(jid)


def test_a_waiting_or_running_job_can_be_cancelled():
    waiting, running = _job(), _job()
    queue.claim("w1")  # takes `waiting`
    assert queue.cancel(running) is True
    assert queue.get(running)["state"] == queue.CANCELLED
    assert queue.cancel(waiting) is True
    assert queue.claim("w2") is None


def test_the_queue_has_a_depth():
    import core.queue as q

    original = q.MAX_QUEUE_DEPTH
    q.MAX_QUEUE_DEPTH = 3
    try:
        for _ in range(3):
            _job()
        with pytest.raises(queue.QueueFull, match="maximum of 3"):
            _job()
    finally:
        q.MAX_QUEUE_DEPTH = original


def test_the_queue_survives_the_process_that_wrote_it(db):
    """A restart is a new process reading the same file, so that is the test."""
    jid = _job(models="Ridge")
    script = f"""
import sys, json
sys.path.insert(0, {str(ROOT)!r})
import os
os.environ["DIGITERRA_QUEUE_PATH"] = {str(db)!r}
from core import queue
got = queue.claim("other-process")
print(json.dumps({{"id": got["id"], "model": got["request"]["models"], "depth": queue.depth()}}))
"""
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    seen = json.loads(out.stdout.strip().splitlines()[-1])
    assert seen == {"id": jid, "model": "Ridge", "depth": 0}
    # And this process sees what that one did.
    assert queue.get(jid)["state"] == queue.RUNNING
    assert queue.get(jid)["worker"] == "other-process"


def test_the_queue_follows_the_profile(monkeypatch):
    monkeypatch.delenv("DIGITERRA_QUEUE", raising=False)
    monkeypatch.setenv("DIGITERRA_PROFILE", "desktop")
    assert not queue.queue_enabled()
    monkeypatch.setenv("DIGITERRA_PROFILE", "server")
    assert queue.queue_enabled()
    monkeypatch.setenv("DIGITERRA_QUEUE", "0")
    assert not queue.queue_enabled()


def test_a_full_slot_queues_the_run_instead_of_refusing_it(monkeypatch, tmp_path):
    """The behaviour the queue exists for, checked through the route."""
    import pandas as pd

    from core import jobs, state
    from core.state import get_session_storage

    monkeypatch.setenv("DIGITERRA_QUEUE", "1")
    monkeypatch.setenv("DIGITERRA_PROFILE", "desktop")  # queue on, auth and CSRF off
    from app import app

    app.config["TESTING"] = True
    client = app.test_client()

    sid = uuid.uuid4().hex
    client.set_cookie("digiterra_session", sid)
    store = get_session_storage(sid)
    df = pd.read_csv("examples/diabetes.csv")
    store["data"] = df
    n = len(df.columns)
    indicators, predictors = list(range(n - 1)), [n - 1]
    store["last_preprocess_request"] = {"indicators": indicators, "predictors": predictors}
    payload = {
        "filename": "diabetes.csv",
        "indicators": indicators,
        "predictors": predictors,
        "models": "Linear",
        "hyperparameters": {},
        "nonreq": False,
        "scaler": "standard",
        "units": "",
        "sigfig": 3,
        "seedValue": 42,
        "testSize": "0.20",
        "stratifyBool": False,
        "dropMissing": "none",
        "imputeStrategy": "none",
        "dropZero": "none",
        "quantileBinDict": {"quantile": 0, "bin": 0, "binsLabel": ""},
        "useTransformer": "No",
        "transformerCols": [],
        "crossValidationType": "None",
        "crossValidationFolds": 5,
    }
    # Hold this session's slot, as a run in flight would.
    jobs.acquire(sid, "already-running")
    try:
        r = client.post("/process", json=payload)
        assert r.status_code == 202, r.get_json()
        body = r.get_json()
        assert body["status"] == "queued"
        assert body["position"] == 0
        assert "waiting" in body["message"]

        # The page can ask where it stands, and only its own session can.
        status = client.get(f"/queue/{body['session_id']}")
        assert status.status_code == 200
        assert status.get_json()["state"] == queue.QUEUED
        client.set_cookie("digiterra_session", uuid.uuid4().hex)
        assert client.get(f"/queue/{body['session_id']}").status_code == 400
    finally:
        jobs.release(sid, "already-running")
        state.clear_session_storage(sid)
