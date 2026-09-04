"""How far a new row sits from the rows a model was trained on.

A model returns a prediction for any row whose columns parse, including rows nothing
like the training data. Nothing in the result says which of those a prediction is.
This fits a detector on the training design matrix, records the range of each
feature, and keeps the threshold that separates the training rows from the rest, so
a prediction can be reported with how far its row sits from what the model saw.

The pipelines set this while the training matrix is in scope and the caller reads it
afterwards, which is the pattern splitting.py already uses for the split plan. It
avoids widening three return tuples and the code that unpacks them by position.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

# A row below this percentile of the training rows' own scores is reported as unlike
# them. One in a hundred training rows falls here, which is the rate a reader should
# expect on data that matches.
NOVELTY_PERCENTILE = 1.0


@dataclass
class Applicability:
    """The detector, the threshold it is read against, and the training ranges."""

    detector: IsolationForest
    threshold: float
    feature_order: list[str]
    feature_min: dict[str, float] = field(default_factory=dict)
    feature_max: dict[str, float] = field(default_factory=dict)
    n_train: int = 0
    percentile: float = NOVELTY_PERCENTILE


_ACTIVE: ContextVar = ContextVar("digiterra_applicability", default=None)


def set_applicability(value: Applicability | None) -> None:
    _ACTIVE.set(value)


def get_applicability() -> Applicability | None:
    return _ACTIVE.get()


def fit_applicability(X_train, seed) -> Applicability | None:
    """Fit the detector on the matrix the model sees and record the ranges.

    Returns None rather than raising when the matrix cannot carry a detector, since
    a run must not fail because this could not be measured.
    """
    try:
        frame = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
        if frame.empty or frame.shape[0] < 2:
            return None
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        if numeric.isna().all().all():
            return None
        numeric = numeric.fillna(numeric.mean(numeric_only=True)).fillna(0.0)
        detector = IsolationForest(random_state=int(seed) if seed is not None else None, contamination="auto")
        detector.fit(numeric)
        scores = detector.score_samples(numeric)
        threshold = float(np.percentile(scores, NOVELTY_PERCENTILE))
        order = [str(c) for c in numeric.columns]
        return Applicability(
            detector=detector,
            threshold=threshold,
            feature_order=order,
            feature_min={c: float(numeric[c].min()) for c in numeric.columns},
            feature_max={c: float(numeric[c].max()) for c in numeric.columns},
            n_train=int(numeric.shape[0]),
        )
    except Exception as e:
        logger.warning("Applicability not measured: %s", e)
        return None


def record_applicability(X_train, seed) -> None:
    """Fit and publish, for the caller that assembles the session store."""
    set_applicability(fit_applicability(X_train, seed))


def score_rows(app: Applicability, X_new) -> dict:
    """Score new rows against the training rows.

    Returns the per-row novelty score, whether each row falls below the threshold,
    and the columns whose value sits outside the training range for that row.
    """
    frame = X_new if isinstance(X_new, pd.DataFrame) else pd.DataFrame(X_new, columns=app.feature_order)
    numeric = frame.reindex(columns=app.feature_order).apply(pd.to_numeric, errors="coerce")
    filled = numeric.fillna(0.0)
    scores = app.detector.score_samples(filled)
    novel = scores < app.threshold

    out_of_range = []
    for _, row in numeric.iterrows():
        names = []
        for column in app.feature_order:
            value = row.get(column)
            if value is None or pd.isna(value):
                continue
            low, high = app.feature_min.get(column), app.feature_max.get(column)
            if low is None or high is None:
                continue
            if value < low or value > high:
                names.append(column)
        out_of_range.append(";".join(names))
    return {
        "novelty_score": [float(s) for s in scores],
        "novel_row": [bool(n) for n in novel],
        "out_of_range_columns": out_of_range,
    }
