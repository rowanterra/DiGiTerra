"""Train/test splits that respect time order or row groups.

A random split leaks when the rows are ordered in time or when several rows
come from the same site, well, or sample. A time-ordered holdout keeps the
latest rows for the test set. A grouped holdout keeps every row of a group in
one half. The random method is the sklearn train_test_split call the pipelines
made before this module existed, so the default behavior is unchanged.

The plan travels to the pipeline through a context variable, for the same
reason as the fill plan in split_fill.py: the train_* wrappers pass a fixed
argument list.
"""

from __future__ import annotations

import math
import warnings
from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

_ACTIVE: ContextVar = ContextVar("digiterra_split_plan", default=None)

SPLIT_METHODS = ("random", "time", "group")


@dataclass
class SplitPlan:
    """Which split method the pipeline uses, and the column the method reads.

    note is filled by split_indices and is meant for the result dict, so the
    user sees which rows formed the test set.
    """

    method: str = "random"
    column: str | None = None
    note: str = ""

    def __post_init__(self):
        self.method = (self.method or "random").strip().lower()
        if self.method not in SPLIT_METHODS:
            raise ValueError(f"Unknown split method {self.method!r}. Expected one of {', '.join(SPLIT_METHODS)}.")
        if self.method != "random" and not self.column:
            raise ValueError(f"The {self.method} split method requires a column.")

    def is_random(self) -> bool:
        return self.method == "random"


def set_split_plan(plan: SplitPlan | None):
    """Make plan the active plan for the current thread of execution."""
    return _ACTIVE.set(plan)


def reset_split_plan(token) -> None:
    try:
        _ACTIVE.reset(token)
    except (ValueError, RuntimeError):
        _ACTIVE.set(None)


def get_split_plan() -> SplitPlan | None:
    return _ACTIVE.get()


def _test_count(n: int, test_size) -> int:
    """Rows in the test set: ceil(test_size * n) for a fraction, the value itself for a count."""
    test_size = float(test_size)
    if test_size >= 1:
        count = int(test_size)
    else:
        count = math.ceil(test_size * n)
    if count < 1 or count >= n:
        raise ValueError(f"test_size {test_size} leaves no rows for one of the two halves (n={n}).")
    return count


def _order_keys(values: pd.Series, column: str) -> np.ndarray:
    """Sortable keys for a time split: datetimes when the column parses as dates, numbers otherwise."""
    present = values.notna()
    if not present.any():
        raise ValueError(f"Split column '{column}' has no values.")
    with warnings.catch_warnings():
        # Mixed formats are parsed one element at a time; the format warning is not useful here.
        warnings.simplefilter("ignore", UserWarning)
        as_time = pd.to_datetime(values, errors="coerce")
    if not (as_time.isna() & present).any():
        return as_time.to_numpy()
    as_num = pd.to_numeric(values, errors="coerce")
    if not (as_num.isna() & present).any():
        return as_num.to_numpy(dtype=float)
    sample = values[present & as_time.isna() & as_num.isna()].iloc[0]
    raise ValueError(
        f"Split column '{column}' has values such as '{sample}' that parse neither as a date nor as a number, "
        "so the rows cannot be ordered in time."
    )


def split_indices(
    ddf: pd.DataFrame,
    test_size,
    seed,
    strat=None,
    plan: SplitPlan | None = None,
    source: pd.DataFrame | None = None,
    feature_columns=None,
):
    """Return (train_index, test_index) over ddf.index for the plan's method.

    ddf is the frame the pipeline models (rows already filtered). source is the
    frame the split column is read from when the column is not in ddf; the
    frame is aligned on ddf.index. feature_columns is used only to word the
    note when the split column is also a feature. The note is written to
    plan.note. strat is used by the random method alone.
    """
    if plan is None:
        plan = SplitPlan()
    n = len(ddf)
    index = ddf.index

    if plan.is_random():
        positions = np.arange(n)
        train_pos, test_pos = train_test_split(positions, test_size=test_size, stratify=strat, random_state=seed)
        plan.note = ""
        return index[train_pos], index[test_pos]

    column = plan.column
    if column in ddf.columns:
        values = ddf[column]
    elif source is not None and column in source.columns:
        values = source[column].reindex(index)
    else:
        raise ValueError(f"Split column '{column}' is not in the data.")
    if values.isna().any():
        raise ValueError(f"Split column '{column}' has missing values in {int(values.isna().sum())} rows.")

    n_test = _test_count(n, test_size)
    feature_note = ""
    if feature_columns is not None and column in list(feature_columns):
        feature_note = f" The column '{column}' is also an indicator, so the model reads the split column as a feature."
    strat_note = " Stratification does not apply to this method and was not used." if strat is not None else ""

    if plan.method == "time":
        keys = _order_keys(values, column)
        order = np.argsort(keys, kind="mergesort")
        train_pos, test_pos = order[:-n_test], order[-n_test:]
        pct = round(100 * n_test / n)
        plan.note = (
            f"Time-ordered holdout on column '{column}': the latest {pct}% of rows "
            f"({n_test} of {n}) are the test set.{strat_note}{feature_note}"
        )
        return index[train_pos], index[test_pos]

    splitter = GroupShuffleSplit(n_splits=1, test_size=float(test_size), random_state=seed)
    groups = values.astype(str).to_numpy()
    train_pos, test_pos = next(splitter.split(np.zeros(n), groups=groups))
    n_groups = len(np.unique(groups))
    n_test_groups = len(np.unique(groups[test_pos]))
    plan.note = (
        f"Grouped holdout on column '{column}': {n_test_groups} of {n_groups} groups "
        f"({len(test_pos)} of {n} rows) are the test set, and no group is in both halves.{strat_note}{feature_note}"
    )
    return index[train_pos], index[test_pos]


__all__ = [
    "SPLIT_METHODS",
    "SplitPlan",
    "get_split_plan",
    "reset_split_plan",
    "set_split_plan",
    "split_indices",
]
