"""Estimator parameters typed on the guided page, applied by their scikit-learn names.

The trainers read the classic page's Advanced form, whose field names differ from
scikit-learn's for most models (RidgeFitIntersept for fit_intercept, for example).
The guided page reads the parameter names from the estimator itself, so its values
are applied to the built estimator with set_params instead of going through the
trainer's own names. The route sets the values for one run in a context variable;
the regression, classification and clustering pipelines apply them before the fit.

A name the estimator does not have stops the run with a message that names it. A
string value is read as None, a bool, an int, a float or a Python literal (a tuple
for hidden_layer_sizes, for example) when it parses as one, and is kept as a string
otherwise, because a text field sends every value as a string.
"""

from __future__ import annotations

import ast
from contextvars import ContextVar, Token
from typing import Any

from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

_PARAMS: ContextVar[dict | None] = ContextVar("digiterra_estimator_params", default=None)


def set_estimator_params(params: dict | None) -> Token:
    """Set the parameters for the run in this thread; returns the token for reset."""
    return _PARAMS.set(dict(params) if params else None)


def reset_estimator_params(token: Token) -> None:
    _PARAMS.reset(token)


def current_estimator_params() -> dict:
    return dict(_PARAMS.get() or {})


def coerce_value(value: Any) -> Any:
    """Read a form value as the Python value scikit-learn expects."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    low = text.lower()
    if low in ("none", "null"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def clean_params(params: dict | None) -> dict:
    """Drop blank values and read the rest with coerce_value."""
    out = {}
    for key, value in (params or {}).items():
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        out[str(key)] = coerce_value(value)
    return out


def apply_estimator_params(model):
    """Set the run's parameters on the estimator, or on the estimator a multi-output wrapper holds."""
    _capture_check(model)
    params = clean_params(_PARAMS.get())
    if not params:
        return model
    wrapped = isinstance(model, (MultiOutputRegressor, MultiOutputClassifier))
    inner = model.estimator if wrapped else model
    prefix = "estimator__" if wrapped else ""
    valid = model.get_params(deep=True)
    unknown = sorted(k for k in params if prefix + k not in valid)
    if unknown:
        raise ValueError(
            f"{type(inner).__name__} has no parameter named {', '.join(unknown)}. "
            "Clear the field or choose a parameter from the table, then train again."
        )
    model.set_params(**{prefix + k: v for k, v in params.items()})
    return model


# ---------------------------------------------------------------------------
# The estimator a trainer builds from an empty form. The guided page's Default
# column shows these values, because a blank field leaves the trainer's value in
# place and several trainers set values other than scikit-learn's (a gradient
# boosting regressor with absolute_error loss, a random forest regressor with
# max_features "sqrt", an MLP with max_iter 10000). Every pipeline calls
# apply_estimator_params before it reads the data, so a capture there returns the
# built estimator without fitting anything.

RUN_SEED = 918273645  # stands for the run seed; the page shows "run seed"

_CAPTURE: ContextVar[bool] = ContextVar("digiterra_capture_estimator", default=False)


class EstimatorCaptured(Exception):
    """Raised by apply_estimator_params during a capture; carries the built estimator."""

    def __init__(self, model):
        super().__init__("estimator captured")
        self.model = model


def _capture_check(model) -> None:
    if _CAPTURE.get():
        raise EstimatorCaptured(model)


def built_estimator(name: str):
    """The estimator the trainer for `name` builds from an empty form, not fitted.

    The trainer runs on an eight-row frame with the seed RUN_SEED; the pipeline stops
    at its first line. Returns the inner estimator of a multi-output wrapper.
    """
    import pandas as pd

    from python_scripts.model_registry import MODEL_REGISTRY, get_model_kwargs

    train_fn, task = MODEL_REGISTRY[name]
    X = pd.DataFrame({"a": [1.0, 2, 3, 4, 5, 6, 7, 8], "b": [2.0, 1, 4, 3, 6, 5, 8, 7]})
    y = pd.DataFrame({"t": [0, 1, 0, 1, 0, 1, 0, 1]})
    bins = {"quantile": 0, "bin": 0, "binsLabel": ""}
    kwargs = get_model_kwargs(name, {}, False, RUN_SEED)
    common = dict(
        units="",
        X_scaler_type="none",
        seed=RUN_SEED,
        quantileBinDict=bins,
        sigfig=3,
        useTransformer="No",
        categorical_cols=[],
    )
    token = _CAPTURE.set(True)
    try:
        if task == "cluster":
            train_fn(train_data=X, X=X, test_size=0.2, k_min=2, k_max=8, **common, **kwargs)
        else:
            base = dict(
                target_variables=["t"],
                train_data=pd.concat([X, y], axis=1),
                use_stratified_split=False,
                X=X,
                y=y,
                stratifyColumn="",
                testSize=0.2,
                **common,
            )
            if task == "regression":
                train_fn(name, y_scaler_type="none", **base, **kwargs)
            else:
                train_fn(**base, **kwargs)
    except EstimatorCaptured as captured:
        model = captured.model
    else:
        raise RuntimeError(f"The trainer for {name} did not reach its pipeline.")
    finally:
        _CAPTURE.reset(token)
    if isinstance(model, (MultiOutputRegressor, MultiOutputClassifier)):
        return model.estimator
    return model
