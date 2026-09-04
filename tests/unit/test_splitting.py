"""Checks on the time-ordered and grouped train/test splits in splitting.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from python_scripts.preprocessing.splitting import (
    SplitPlan,
    get_split_plan,
    reset_split_plan,
    set_split_plan,
    split_indices,
)


def _frame(n=50, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "date": pd.date_range("2020-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
            "site": rng.choice(["S1", "S2", "S3", "S4", "S5"], size=n),
            "y": rng.normal(size=n),
        }
    )
    # A filtered, shuffled index, as the pipelines produce after dropna.
    frame = frame.sample(frac=0.9, random_state=seed)
    return frame


def test_random_matches_sklearn_for_the_same_seed():
    frame = _frame()
    X = frame[["a", "b"]]
    y = frame["y"]
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    train_idx, test_idx = split_indices(X, 0.2, 42, plan=SplitPlan())
    assert list(train_idx) == list(X_train.index)
    assert list(test_idx) == list(X_test.index)


def test_random_with_stratify_matches_sklearn():
    frame = _frame(n=80)
    X = frame[["a", "b"]]
    strat = (frame["a"] > 0).astype(int)
    X_train, X_test = train_test_split(X, test_size=0.25, stratify=strat, random_state=7)
    train_idx, test_idx = split_indices(X, 0.25, 7, strat=strat, plan=None)
    assert list(train_idx) == list(X_train.index)
    assert list(test_idx) == list(X_test.index)


def test_time_split_puts_the_latest_rows_in_the_test_set():
    frame = _frame()
    X = frame[["a", "b"]]
    plan = SplitPlan(method="time", column="date")
    train_idx, test_idx = split_indices(X, 0.2, 1, strat=pd.Series(0, index=X.index), plan=plan, source=frame)
    dates = pd.to_datetime(frame["date"])
    assert dates.loc[train_idx].max() < dates.loc[test_idx].min()
    assert len(test_idx) == int(np.ceil(0.2 * len(X)))
    assert len(train_idx) + len(test_idx) == len(X)
    assert "Time-ordered holdout on column 'date'" in plan.note
    assert "Stratification does not apply" in plan.note


def test_time_split_falls_back_to_numeric_order():
    frame = _frame()
    frame["step"] = np.arange(len(frame))[::-1]
    plan = SplitPlan(method="time", column="step")
    train_idx, test_idx = split_indices(frame[["a"]], 0.3, 1, plan=plan, source=frame)
    assert frame.loc[train_idx, "step"].max() < frame.loc[test_idx, "step"].min()


def test_time_split_rejects_a_column_that_is_neither_date_nor_number():
    frame = _frame()
    with pytest.raises(ValueError, match="neither as a date nor as a number"):
        split_indices(frame[["a"]], 0.2, 1, plan=SplitPlan(method="time", column="site"), source=frame)


def test_group_split_shares_no_group_between_halves():
    frame = _frame(n=120)
    plan = SplitPlan(method="group", column="site")
    train_idx, test_idx = split_indices(frame[["a", "b"]], 0.2, 3, plan=plan, source=frame)
    train_groups = set(frame.loc[train_idx, "site"])
    test_groups = set(frame.loc[test_idx, "site"])
    assert train_groups.isdisjoint(test_groups)
    assert train_groups | test_groups == set(frame["site"])
    assert len(train_idx) + len(test_idx) == len(frame)
    assert "Grouped holdout on column 'site'" in plan.note


def test_split_column_that_is_a_feature_is_named_in_the_note():
    frame = _frame()
    frame["step"] = np.arange(len(frame))
    X = frame[["a", "step"]]
    plan = SplitPlan(method="time", column="step")
    split_indices(X, 0.2, 1, plan=plan, feature_columns=X.columns)
    assert "also an indicator" in plan.note


def test_plan_validation():
    with pytest.raises(ValueError):
        SplitPlan(method="time")
    with pytest.raises(ValueError):
        SplitPlan(method="bogus", column="x")
    assert SplitPlan(method=None).is_random()


def test_contextvar_round_trip():
    assert get_split_plan() is None
    token = set_split_plan(SplitPlan(method="group", column="site"))
    assert get_split_plan().method == "group"
    reset_split_plan(token)
    assert get_split_plan() is None


def test_pipeline_time_split_through_run_model_training(tmp_path):
    from tests.unit.test_pipelines import _train

    rng = np.random.default_rng(0)
    n = 60
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=n, freq="7D").strftime("%Y-%m-%d"),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    frame["target"] = 2 * frame["x1"] - frame["x2"] + rng.normal(scale=0.1, size=n)
    csv = tmp_path / "dated.csv"
    frame.to_csv(csv, index=False)
    store = {}
    result, _ = _train(
        store, str(csv), [1, 2], [3], "Linear", modelingMode="advanced", splitMethod="time", splitColumn="date"
    )
    assert "split_note" in result
    assert "Time-ordered holdout on column 'date'" in result["split_note"]
