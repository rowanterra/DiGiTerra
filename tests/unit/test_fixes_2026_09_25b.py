"""Fixes of 2026-09-25 from the Mac session with penguins.csv.

1. Choosing the same column for the pairplot's X and Y failed with "Pairplot update
   failed: The server could not complete that request": the frame had a duplicated
   column name, which seaborn rejects. The message went to #errorDiv, which sits in
   the Modeling form, so it appeared there after the user had left Data Exploration.
2. The MLP classifier reported "Result not available after training completed". The
   progress reached 100 before the run wrote its record, and the stream waited about
   2 seconds for the result; that step took 1.35 seconds for the MLP in the container.
"""

import json
import time

import pandas as pd

import routes.main as main_routes
from python_scripts.app_exploration import handle_pairplot


def test_the_same_column_twice_draws_its_histogram(tmp_path):
    store = {"data": pd.read_csv("examples/penguins.csv")}
    body = {"x": "bill_depth_mm", "y": "bill_depth_mm", "colsIgnore": "all"}
    result, status = handle_pairplot(store, body, tmp_path, lambda u: u)
    assert status == 200, result
    assert (tmp_path / "pairplot_bill_depth_mm_bill_depth_mm.png").stat().st_size > 0


class _Tracker:
    def get_progress(self):
        return {"overall_progress": 100}


def test_the_stream_waits_for_a_result_written_after_the_progress_reads_100(monkeypatch):
    ready_at = time.time() + 3.0  # longer than the old 2-second wait
    monkeypatch.setattr(main_routes, "get_result", lambda _sid: {"ok": True} if time.time() >= ready_at else None)
    monkeypatch.setattr(main_routes, "remove_tracker", lambda _sid: None)
    events = [json.loads(e[len("data: ") :]) for e in main_routes._progress_events(_Tracker(), "s", time.time())]
    assert events[-1] == {"type": "result", "data": {"ok": True}}
    assert not any("error" in e for e in events)


def test_the_stream_still_reports_a_result_that_never_arrives(monkeypatch):
    monkeypatch.setattr(main_routes, "PROGRESS_RESULT_WAIT_SECONDS", 0.6)
    monkeypatch.setattr(main_routes, "get_result", lambda _sid: None)
    monkeypatch.setattr(main_routes, "remove_tracker", lambda _sid: None)
    events = [json.loads(e[len("data: ") :]) for e in main_routes._progress_events(_Tracker(), "s", time.time())]
    assert events[-1] == {"error": "Result not available after training completed"}
