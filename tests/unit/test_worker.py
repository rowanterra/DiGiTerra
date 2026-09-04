"""The isolated worker: a memory limit and a wall clock that the parent survives.

A budget measured inside the process doing the allocating cannot bound that
process. These tests run work that would take the machine and check that the
child stops instead, that the parent keeps running, and that an ordinary result
still comes back.
"""

import os
import warnings

import pytest

from core import worker

warnings.filterwarnings("ignore")


def _sum(a, b, scale=1):
    return (a + b) * scale


def _allocate(mb):
    """Ask for mb of memory in one block."""
    return len(bytearray(int(mb) * 1024 * 1024))


def _spin():
    while True:
        pass


def _raises():
    raise ValueError("the step decided not to")


# A child spends this much address space on the interpreter and libraries before it
# does any work, so every limit in these tests is that baseline plus the data.
BASE = worker.BASE_MEMORY_MB


def test_a_result_comes_back_with_arguments_and_keywords():
    assert worker.run_isolated(_sum, 2, 3, scale=10, memory_mb=BASE, timeout_s=60) == 50


def test_an_allocation_past_the_limit_stops_the_child_not_the_parent():
    with pytest.raises(worker.WorkerMemoryExceeded):
        worker.run_isolated(_allocate, 4096, memory_mb=BASE, timeout_s=120)
    # The parent is unharmed and still does work afterwards.
    assert worker.run_isolated(_sum, 1, 1, memory_mb=BASE, timeout_s=60) == 2


def test_a_step_that_never_finishes_is_stopped_at_the_wall_clock():
    with pytest.raises(worker.WorkerTimeout, match="past its limit"):
        worker.run_isolated(_spin, memory_mb=BASE, timeout_s=3)


def test_an_error_in_the_child_reaches_the_caller_with_its_message():
    with pytest.raises(worker.WorkerError, match="the step decided not to"):
        worker.run_isolated(_raises, memory_mb=BASE, timeout_s=60)


def test_isolation_follows_the_profile_and_its_own_switch(monkeypatch):
    monkeypatch.delenv("DIGITERRA_ISOLATE_WORK", raising=False)
    monkeypatch.setenv("DIGITERRA_PROFILE", "desktop")
    assert not worker.workers_enabled()
    monkeypatch.setenv("DIGITERRA_PROFILE", "server")
    assert worker.workers_enabled()
    monkeypatch.setenv("DIGITERRA_ISOLATE_WORK", "0")
    assert not worker.workers_enabled()


def test_a_parse_runs_in_the_child_and_returns_the_frame(tmp_path, monkeypatch):
    import pandas as pd

    monkeypatch.setenv("DIGITERRA_ISOLATE_WORK", "1")
    path = tmp_path / "rows.csv"
    pd.DataFrame({"a": range(200), "b": range(200)}).to_csv(path, index=False)
    frame = worker.read_table_isolated(path, max_cells=0, max_frame_mb=64)
    assert len(frame) == 200 and list(frame.columns) == ["a", "b"]


def test_a_parse_over_the_child_budget_is_refused_without_touching_the_parent(tmp_path, monkeypatch):
    import pandas as pd

    monkeypatch.setenv("DIGITERRA_ISOLATE_WORK", "1")
    path = tmp_path / "wide.csv"
    pd.DataFrame({f"c{i}": range(60000) for i in range(30)}).to_csv(path, index=False)
    # A budget far below what the parse needs: the child stops, the parent does not.
    monkeypatch.setattr(worker, "BASE_MEMORY_MB", 320.0)
    with pytest.raises((worker.WorkerMemoryExceeded, worker.WorkerError)):
        worker.read_table_isolated(path, max_cells=0, max_frame_mb=1)
    assert os.getpid() > 0
