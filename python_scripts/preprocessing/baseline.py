"""Baseline predictors scored on the same validation split as the model.

A regression baseline predicts the training mean of each target for every
validation row. A classification baseline predicts the most common training
class. A model that does not beat these has learned nothing beyond the target
distribution, and the report says so next to the score.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error, r2_score


def _as_2d(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def regression_baseline(y_train, y_test, model_r2: float | None) -> dict:
    """Score a predict-the-training-mean baseline on the validation rows."""
    ytr = _as_2d(y_train)
    yte = _as_2d(y_test)
    means = np.nanmean(ytr, axis=0)
    pred = np.tile(means, (yte.shape[0], 1))
    r2 = float(r2_score(yte, pred)) if yte.shape[0] > 1 else float("nan")
    rmse = float(np.sqrt(mean_squared_error(yte, pred)))
    beats = None
    if model_r2 is not None and np.isfinite(model_r2) and np.isfinite(r2):
        beats = bool(model_r2 > r2)
    return {
        "kind": "regression",
        "description": "Predicting the training mean",
        "metric": "R²",
        "baseline_value": round(r2, 4) if np.isfinite(r2) else None,
        "model_value": round(float(model_r2), 4) if model_r2 is not None and np.isfinite(model_r2) else None,
        "baseline_rmse": round(rmse, 6),
        "training_mean": [float(m) for m in means],
        "model_beats_baseline": beats,
    }


def classification_baseline(y_train, y_test, model_accuracy: float | None) -> dict:
    """Score a predict-the-most-common-class baseline on the validation rows."""
    ytr = [str(v) for v in np.ravel(np.asarray(y_train))]
    yte = np.array([str(v) for v in np.ravel(np.asarray(y_test))])
    majority, count = Counter(ytr).most_common(1)[0]
    pred = np.full(yte.shape, majority)
    acc = float(accuracy_score(yte, pred))
    bal = float(balanced_accuracy_score(yte, pred)) if len(set(yte)) > 1 else acc
    beats = None
    if model_accuracy is not None and np.isfinite(model_accuracy):
        beats = bool(model_accuracy > acc)
    return {
        "kind": "classification",
        "description": f"Predicting the most common training class ({majority}, {count} of {len(ytr)} training rows)",
        "metric": "accuracy",
        "baseline_value": round(acc, 4),
        "model_value": round(float(model_accuracy), 4) if model_accuracy is not None else None,
        "baseline_balanced_accuracy": round(bal, 4),
        "majority_class": majority,
        "model_beats_baseline": beats,
    }


def baseline_for_frame(y_train: pd.DataFrame | pd.Series, y_test, kind: str, model_value: float | None) -> dict:
    if kind == "classification":
        return classification_baseline(y_train, y_test, model_value)
    return regression_baseline(y_train, y_test, model_value)
