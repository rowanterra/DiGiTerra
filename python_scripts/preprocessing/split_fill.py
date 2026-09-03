"""Statistical fills fitted on the training split only.

Mean and median imputation, and mean or median replacement of zeros, are
statistics of the data. Computing them over the whole file before the
train/test split lets test rows inform the training values, which inflates
test metrics. preprocess_data therefore records which columns need a
statistical fill and defers the fill; each pipeline fits the plan on its
training rows after the split, applies the same values to the test rows, and
the fitted values are stored so inference reuses them.

The plan travels to the pipeline through a context variable rather than a
parameter because the 67 train_* wrappers pass a fixed argument list.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field

import pandas as pd

_ACTIVE: ContextVar = ContextVar("digiterra_fill_plan", default=None)


@dataclass
class FillPlan:
    """Which columns get which statistical fill, and the values once fitted."""

    nan_strategy: str | None = None  # 'mean' or 'median'
    nan_columns: list[str] = field(default_factory=list)
    zero_strategy: str | None = None  # 'mean' or 'median' of the non-zero values
    zero_columns: list[str] = field(default_factory=list)
    nan_fills: dict[str, float] = field(default_factory=dict)
    zero_fills: dict[str, float] = field(default_factory=dict)
    fitted: bool = False

    @property
    def columns(self) -> set[str]:
        return set(self.nan_columns) | set(self.zero_columns)

    def is_empty(self) -> bool:
        return not self.nan_columns and not self.zero_columns

    def fit(self, train_rows: pd.DataFrame) -> FillPlan:
        """Compute every fill value from train_rows and nothing else."""
        self.nan_fills.clear()
        self.zero_fills.clear()
        for col in self.nan_columns:
            if col not in train_rows.columns:
                continue
            s = pd.to_numeric(train_rows[col], errors="coerce")
            v = s.mean() if self.nan_strategy == "mean" else s.median()
            if pd.notna(v):
                self.nan_fills[col] = float(v)
        for col in self.zero_columns:
            if col not in train_rows.columns:
                continue
            s = pd.to_numeric(train_rows[col], errors="coerce")
            non_zero = s[(s != 0) & s.notna()]
            v = non_zero.mean() if self.zero_strategy == "mean" else non_zero.median()
            self.zero_fills[col] = float(v) if pd.notna(v) else 0.0
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted values. A column absent from the frame is skipped."""
        if not self.fitted:
            raise RuntimeError("FillPlan.transform called before fit")
        out = frame.copy()
        for col, v in self.zero_fills.items():
            if col in out.columns:
                mask = out[col] == 0
                if mask.any():
                    out.loc[mask, col] = v
        for col, v in self.nan_fills.items():
            if col in out.columns:
                out[col] = out[col].fillna(v)
        return out

    def fit_transform_by_index(self, frame: pd.DataFrame, train_index) -> pd.DataFrame:
        """Fit on the rows in train_index, then fill every row of frame."""
        self.fit(frame.loc[train_index])
        return self.transform(frame)

    def dropna_subset(self, use_cols: list[str]) -> list[str]:
        """Columns whose NaN still drops the row: everything the plan will not fill."""
        return [c for c in use_cols if c not in set(self.nan_columns)]


def set_fill_plan(plan: FillPlan | None):
    """Make plan the active plan for the current thread of execution."""
    return _ACTIVE.set(plan)


def reset_fill_plan(token) -> None:
    try:
        _ACTIVE.reset(token)
    except ValueError:
        _ACTIVE.set(None)


def get_fill_plan() -> FillPlan | None:
    plan = _ACTIVE.get()
    return None if plan is None or plan.is_empty() else plan


def _to_records(plan: FillPlan | None) -> dict:
    """Serializable form for session storage and the response."""
    if plan is None:
        return {"nan_fills": {}, "zero_fills": {}}
    return {
        "nan_fills": {k: float(v) for k, v in plan.nan_fills.items()},
        "zero_fills": {k: float(v) for k, v in plan.zero_fills.items()},
        "nan_strategy": plan.nan_strategy,
        "zero_strategy": plan.zero_strategy,
        "fitted_on": "training split",
    }


def apply_fill_plan_to_split(ddf: pd.DataFrame, train_index) -> tuple[pd.DataFrame, FillPlan | None]:
    """Fit the active plan on the training rows of ddf and fill every row.

    Returns the filled frame and the plan, or the frame unchanged and None when
    no statistical fill was requested.
    """
    plan = get_fill_plan()
    if plan is None:
        return ddf, None
    filled = plan.fit_transform_by_index(ddf, train_index)
    return filled, plan


__all__ = [
    "FillPlan",
    "apply_fill_plan_to_split",
    "get_fill_plan",
    "reset_fill_plan",
    "set_fill_plan",
]
