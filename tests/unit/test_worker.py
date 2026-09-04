"""The isolated worker: a memory limit and a wall clock that the parent survives.

A budget measured inside the process doing the allocating cannot bound that
process. These tests run work that would take the machine and check that the
child stops instead, that the parent keeps running, and that an ordinary result
still comes back.

RLIMIT_AS is not honored on every platform: on Darwin it is a second name for
RLIMIT_RSS, which the kernel does not implement, so setrlimit refuses every
value. These tests therefore assert the guarantee the running platform actually
offers rather than skipping. Where the ceiling can be set, a step past it is
stopped. Where it cannot, a step that asks for a ceiling is refused before a
child starts, which is the correct outcome there and is checked as one.
"""

import os
import sys
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


def _take_ceiling_in_child(mb):
    """Try to take the real ceiling from inside a child, as a step would."""
    return worker._apply_limits(mb)[0]


# A child spends this much address space on the interpreter and libraries before it
# does any work, so every limit in these tests is that baseline plus the data.
BASE = worker.BASE_MEMORY_MB

# Whether this platform will hold a process to an address-space ceiling.
LIMIT_SUPPORTED, LIMIT_REASON = worker.memory_limit_supported()

# Where a ceiling cannot be set, run_isolated refuses any step that asks for one.
# The tests below that are about results, errors and the wall clock ask for no
# ceiling there, so on either platform they exercise a guarantee that platform
# really makes. The ceiling itself is tested separately, on both.
LIMIT_MB = BASE if LIMIT_SUPPORTED else 0


def test_a_result_comes_back_with_arguments_and_keywords():
    assert worker.run_isolated(_sum, 2, 3, scale=10, memory_mb=LIMIT_MB, timeout_s=60) == 50


def test_an_allocation_past_the_limit_stops_the_child_not_the_parent():
    """Where a ceiling holds, the child stops at it; where none can be set, nothing runs."""
    if LIMIT_SUPPORTED:
        with pytest.raises(worker.WorkerMemoryExceeded):
            worker.run_isolated(_allocate, 4096, memory_mb=BASE, timeout_s=120)
    else:
        with pytest.raises(worker.WorkerLimitUnavailable) as caught:
            worker.run_isolated(_allocate, 4096, memory_mb=BASE, timeout_s=120)
        message = str(caught.value)
        assert sys.platform in message, message
        assert "RLIMIT_AS" in message, message
    # The parent is unharmed and still does work afterwards.
    assert worker.run_isolated(_sum, 1, 1, memory_mb=LIMIT_MB, timeout_s=60) == 2


def test_a_step_is_refused_before_a_child_starts_when_no_ceiling_can_be_set(monkeypatch):
    """Fail closed, and do it in the parent: no process, no work, and a reason.

    Simulated rather than platform-dependent, so the refusal path is covered
    wherever the suite runs, including the platforms where the ceiling works.
    """
    monkeypatch.setattr(worker, "memory_limit_supported", lambda: (False, "this platform refuses setrlimit(RLIMIT_AS)"))
    started = []
    monkeypatch.setattr(worker.mp, "get_context", lambda *a, **k: started.append(1))
    with pytest.raises(worker.WorkerLimitUnavailable) as caught:
        worker.run_isolated(_sum, 1, 1, memory_mb=BASE, timeout_s=60)
    assert not started, "a child was started for a step that could not be bounded"
    assert "RLIMIT_AS" in str(caught.value)


def test_the_parent_answer_matches_whether_a_child_can_take_the_ceiling():
    """What the parent claims about the platform, the allocating process must find.

    The parent refuses on a platform reading rather than a syscall, so this is the
    check that keeps the reading true: a child asks for the same budget a step
    would, and reports whether the kernel gave it.
    """
    assert worker.run_isolated(_take_ceiling_in_child, BASE, memory_mb=0, timeout_s=60) is LIMIT_SUPPORTED


def test_a_step_that_never_finishes_is_stopped_at_the_wall_clock():
    with pytest.raises(worker.WorkerTimeout, match="past its limit"):
        worker.run_isolated(_spin, memory_mb=LIMIT_MB, timeout_s=3)


def test_an_error_in_the_child_reaches_the_caller_with_its_message():
    with pytest.raises(worker.WorkerError, match="the step decided not to"):
        worker.run_isolated(_raises, memory_mb=LIMIT_MB, timeout_s=60)


def test_isolation_follows_the_profile_and_its_own_switch(monkeypatch):
    monkeypatch.delenv("DIGITERRA_ISOLATE_WORK", raising=False)
    monkeypatch.setenv("DIGITERRA_PROFILE", "desktop")
    assert not worker.workers_enabled()
    monkeypatch.setenv("DIGITERRA_PROFILE", "server")
    assert worker.workers_enabled()
    monkeypatch.setenv("DIGITERRA_ISOLATE_WORK", "0")
    assert not worker.workers_enabled()


def _rows(tmp_path):
    import pandas as pd

    path = tmp_path / "rows.csv"
    pd.DataFrame({"a": range(200), "b": range(200)}).to_csv(path, index=False)
    return path


def test_a_parse_with_isolation_off_returns_the_frame_in_this_process(tmp_path, monkeypatch):
    monkeypatch.setenv("DIGITERRA_ISOLATE_WORK", "0")
    frame = worker.read_table_isolated(_rows(tmp_path), max_cells=0, max_frame_mb=64)
    assert len(frame) == 200 and list(frame.columns) == ["a", "b"]


def test_a_parse_with_isolation_on_runs_in_a_bounded_child_or_does_not_run(tmp_path, monkeypatch):
    """Under the server profile a parse is bounded, or it is refused. Never neither."""
    monkeypatch.setenv("DIGITERRA_ISOLATE_WORK", "1")
    path = _rows(tmp_path)
    if LIMIT_SUPPORTED:
        frame = worker.read_table_isolated(path, max_cells=0, max_frame_mb=64)
        assert len(frame) == 200 and list(frame.columns) == ["a", "b"]
    else:
        with pytest.raises(worker.WorkerLimitUnavailable):
            worker.read_table_isolated(path, max_cells=0, max_frame_mb=64)


def test_a_parse_over_the_child_budget_is_refused_without_touching_the_parent(tmp_path, monkeypatch):
    import pandas as pd

    monkeypatch.setenv("DIGITERRA_ISOLATE_WORK", "1")
    path = tmp_path / "wide.csv"
    pd.DataFrame({f"c{i}": range(60000) for i in range(30)}).to_csv(path, index=False)
    # A budget far below what the parse needs: the child stops, the parent does not.
    # Where no ceiling can be set the same call is refused outright, which is the
    # same promise kept a different way: the parse does not run unbounded.
    monkeypatch.setattr(worker, "BASE_MEMORY_MB", 320.0)
    with pytest.raises(worker.WorkerError):
        worker.read_table_isolated(path, max_cells=0, max_frame_mb=1)
    assert os.getpid() > 0
