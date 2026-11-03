"""Session storage: the same session read by another process.

The memory backend is what the desktop uses and what the rest of the suite runs
against. These tests exercise the disk backend, which is what lets more than one
worker serve a deployment and lets a session outlive a restart.
"""

import json
import subprocess
import sys
import uuid
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import session_store, state

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def disk(tmp_path, monkeypatch):
    monkeypatch.setenv("DIGITERRA_SESSION_BACKEND", "disk")
    session_store.reset_backend()
    monkeypatch.setattr(session_store, "_backend", session_store.DiskBackend(tmp_path / "sessions"))
    monkeypatch.setattr(session_store, "_backend_kind", "disk")
    yield tmp_path / "sessions"
    session_store.reset_backend()


def test_a_frame_and_a_model_survive_a_round_trip(disk):
    from sklearn.linear_model import LinearRegression

    sid = uuid.uuid4().hex
    state.clear_session_storage(sid)
    store = state.get_session_storage(sid)
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
    store["data"] = df
    store["model"] = LinearRegression().fit(np.arange(6).reshape(3, 2), np.arange(3))
    store["seed"] = 42
    store["feature_order"] = ["a"]
    assert store.dirty
    state.flush(sid)
    assert not store.dirty

    # A reader that has never seen this session gets the same contents.
    loaded = session_store.backend().load(sid)
    assert loaded["seed"] == 42 and loaded["feature_order"] == ["a"]
    pd.testing.assert_frame_equal(loaded["data"], df)
    assert np.allclose(loaded["model"].coef_, store["model"].coef_)


def test_a_second_process_reads_a_session_this_one_wrote(disk):
    """The point of the disk backend, checked the only way that proves it."""
    from sklearn.linear_model import LinearRegression

    sid = uuid.uuid4().hex
    store = state.get_session_storage(sid)
    store["data"] = pd.DataFrame({"a": range(50)})
    store["model"] = LinearRegression().fit(np.arange(6).reshape(3, 2), np.arange(3))
    store["marker"] = "written by the first process"
    state.flush(sid)

    script = f"""
import sys, json
sys.path.insert(0, {str(ROOT)!r})
from core.session_store import DiskBackend
loaded = DiskBackend({str(disk)!r}).load({sid!r})
print(json.dumps({{
    "marker": loaded.get("marker"),
    "rows": int(len(loaded["data"])),
    "has_model": loaded.get("model") is not None,
}}))
"""
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    seen = json.loads(out.stdout.strip().splitlines()[-1])
    assert seen == {"marker": "written by the first process", "rows": 50, "has_model": True}


def test_a_partial_write_is_never_visible(disk):
    """Writes are renamed into place, so a reader sees one version or the other."""
    sid = uuid.uuid4().hex
    store = state.get_session_storage(sid)
    store["data"] = pd.DataFrame({"a": range(10)})
    state.flush(sid)
    first = session_store.backend().load(sid)
    store["data"] = pd.DataFrame({"a": range(20)})
    state.flush(sid)
    second = session_store.backend().load(sid)
    assert len(first["data"]) == 10 and len(second["data"]) == 20
    assert not list((disk / sid).glob("*.tmp*"))


def test_clearing_a_session_removes_its_directory(disk):
    sid = uuid.uuid4().hex
    store = state.get_session_storage(sid)
    store["marker"] = "here"
    state.flush(sid)
    assert (disk / sid).is_dir()
    state.clear_session_storage(sid)
    assert not (disk / sid).exists()
    assert session_store.backend().load(sid) is None


def test_an_unusable_session_id_is_refused_rather_than_used_as_a_path(disk):
    with pytest.raises(ValueError, match="Unusable session id"):
        session_store.backend().load("../../etc/passwd")


def test_the_backend_follows_the_profile(monkeypatch):
    session_store.reset_backend()
    monkeypatch.delenv("DIGITERRA_SESSION_BACKEND", raising=False)
    monkeypatch.setenv("DIGITERRA_PROFILE", "desktop")
    assert session_store.backend_name() == "memory"
    monkeypatch.setenv("DIGITERRA_PROFILE", "server")
    assert session_store.backend_name() == "disk"
    monkeypatch.setenv("DIGITERRA_SESSION_BACKEND", "memory")
    assert session_store.backend_name() == "memory"
    session_store.reset_backend()
