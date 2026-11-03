"""How much to trust one prediction, rather than the model as a whole.

A regression result reports one number per row and says nothing about how far that
number might be from the truth. Split conformal gives an interval: the absolute
residuals on the validation rows are measured, their (1 - alpha) quantile is kept,
and every later prediction carries that half-width. The interval covers the truth
for at least 1 - alpha of new rows, provided those rows are exchangeable with the
validation rows. That proviso is the whole of the method's weakness, and the
novelty score in applicability.py is what checks it.

A classifier that reports probabilities needs none of this: the probability is the
per-row statement, and it is written out beside the prediction. A classifier without
predict_proba gets nothing, and the model card says why.

The quantile is measured in target units, after the y_scaler inverse, because that
is what a reader of the interval is looking at.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# One interval in ten is allowed to miss. Anything tighter needs more validation
# rows than a run of this size usually has.
ALPHA = 0.1


@dataclass
class Conformal:
    """The half-width of the interval, one per target, and where it came from."""

    half_width: dict[str, float] = field(default_factory=dict)
    alpha: float = ALPHA
    n_validation: int = 0
    target_names: list[str] = field(default_factory=list)


_ACTIVE: ContextVar = ContextVar("digiterra_conformal", default=None)


def set_conformal(value: Conformal | None) -> None:
    _ACTIVE.set(value)


def get_conformal() -> Conformal | None:
    return _ACTIVE.get()


def fit_conformal(y_true, y_pred, alpha: float = ALPHA) -> Conformal | None:
    """The (1 - alpha) quantile of the absolute residuals, per target.

    Both arguments are in target units. Returns None rather than raising, since a
    run must not fail because an interval could not be measured.
    """
    try:
        truth = y_true if isinstance(y_true, pd.DataFrame) else pd.DataFrame(y_true)
        pred = y_pred if isinstance(y_pred, pd.DataFrame) else pd.DataFrame(y_pred)
        if truth.empty or truth.shape != pred.shape:
            return None
        names = [str(c) for c in truth.columns]
        widths = {}
        for i, name in enumerate(names):
            residual = np.abs(np.asarray(truth.iloc[:, i], dtype=float) - np.asarray(pred.iloc[:, i], dtype=float))
            residual = residual[np.isfinite(residual)]
            if residual.size == 0:
                continue
            # The finite-sample correction: the rank that holds the guarantee for n
            # validation rows rather than the plain quantile of the sample.
            n = residual.size
            level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
            widths[name] = float(np.quantile(residual, level))
        if not widths:
            return None
        return Conformal(half_width=widths, alpha=float(alpha), n_validation=int(truth.shape[0]), target_names=names)
    except Exception as e:
        logger.warning("Conformal interval not measured: %s", e)
        return None


def record_conformal(y_true, y_pred, alpha: float = ALPHA) -> None:
    set_conformal(fit_conformal(y_true, y_pred, alpha))
