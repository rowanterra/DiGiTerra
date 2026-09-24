"""The guided page and the two small routes it needs.

/guided serves one page with a Guided and a Full mode. /column-types tells
the page which uploaded columns are numeric and how many distinct values
each holds, so it can propose a target and a task. /model-defaults/<name>
returns the estimator's own parameter names and defaults for the models
whose trainers accept scikit-learn's names, so the page's hyperparameter
table cannot drift from the library.
"""

from __future__ import annotations

import inspect

import pandas as pd
from flask import Blueprint, jsonify, render_template

from core.constants import HTTP_BAD_REQUEST, HTTP_NOT_FOUND
from core.helpers import with_prefix
from core.state import get_session_storage
from python_scripts import config
from python_scripts.model_registry import _PASSTHROUGH_CLASSIFIERS, MODEL_REGISTRY

bp = Blueprint("guided", __name__)

# Registry name -> scikit-learn class, for the classifiers whose trainers read
# scikit-learn's own parameter names (see model_registry._PASSTHROUGH_CLASSIFIERS).
_SKLEARN_CLASS = {
    "AdaBoost_classifier": ("sklearn.ensemble", "AdaBoostClassifier"),
    "Bagging_classifier": ("sklearn.ensemble", "BaggingClassifier"),
    "BernoulliNB_classifier": ("sklearn.naive_bayes", "BernoulliNB"),
    "CategoricalNB_classifier": ("sklearn.naive_bayes", "CategoricalNB"),
    "ComplementNB_classifier": ("sklearn.naive_bayes", "ComplementNB"),
    "DecisionTree_classifier": ("sklearn.tree", "DecisionTreeClassifier"),
    "ExtraTrees_classifier": ("sklearn.ensemble", "ExtraTreesClassifier"),
    "GaussianNB_classifier": ("sklearn.naive_bayes", "GaussianNB"),
    "GradientBoosting_classifier": ("sklearn.ensemble", "GradientBoostingClassifier"),
    "HistGradientBoosting_classifier": ("sklearn.ensemble", "HistGradientBoostingClassifier"),
    "KNeighbors_classifier": ("sklearn.neighbors", "KNeighborsClassifier"),
    "LDA_classifier": ("sklearn.discriminant_analysis", "LinearDiscriminantAnalysis"),
    "LinearSVC_classifier": ("sklearn.svm", "LinearSVC"),
    "MultinomialNB_classifier": ("sklearn.naive_bayes", "MultinomialNB"),
    "NuSVC_classifier": ("sklearn.svm", "NuSVC"),
    "PassiveAggressive_classifier": ("sklearn.linear_model", "PassiveAggressiveClassifier"),
    "QDA_classifier": ("sklearn.discriminant_analysis", "QuadraticDiscriminantAnalysis"),
    "Ridge_classifier": ("sklearn.linear_model", "RidgeClassifier"),
    "SGD_classifier": ("sklearn.linear_model", "SGDClassifier"),
}


@bp.route("/guided")
def guided():
    """Render the guided page."""
    return render_template(
        "guided.html",
        api_root=config.URL_PREFIX,
        static_root=with_prefix("/static"),
        full_page=with_prefix("/"),
    )


@bp.route("/models")
def models():
    """Every trainable model, grouped by task, in registry order."""
    out = {"regression": [], "classification": [], "cluster": []}
    for name, (train_fn, task) in MODEL_REGISTRY.items():
        if train_fn is None:
            continue
        out[task].append({"name": name, "editable": name in _PASSTHROUGH_CLASSIFIERS or name == "Linear"})
    return jsonify(out)


@bp.route("/column-types")
def column_types():
    """Type, distinct count and missing count of every uploaded column."""
    store = get_session_storage()
    df = store.get("data")
    if df is None:
        return jsonify({"error": "Upload a file first."}), HTTP_BAD_REQUEST
    cols = []
    n = len(df)
    for i, c in enumerate(df.columns):
        s = df[c]
        numeric = bool(pd.api.types.is_numeric_dtype(s))
        distinct = int(s.nunique(dropna=True))
        cols.append(
            {
                "index": i,
                "name": str(c),
                "numeric": numeric,
                "distinct": distinct,
                "missing": int(s.isna().sum()),
                # A numeric column with few distinct values reads as a class label;
                # the page proposes classification for it and regression otherwise.
                "looks_categorical": (not numeric) or (distinct <= max(2, min(10, n // 20))),
            }
        )
    return jsonify({"rows": n, "columns": cols})


@bp.route("/model-defaults/<name>")
def model_defaults(name: str):
    """Parameter names, defaults and kinds for one model, from the estimator itself."""
    if name not in MODEL_REGISTRY or MODEL_REGISTRY[name][0] is None:
        return jsonify({"error": f"Unknown model {name!r}."}), HTTP_NOT_FOUND
    if name not in _SKLEARN_CLASS:
        return jsonify(
            {
                "name": name,
                "editable": False,
                "note": "This model's trainer uses its own parameter names. Its settings are edited on the full page.",
                "params": [],
            }
        )
    module, cls = _SKLEARN_CLASS[name]
    est_cls = getattr(__import__(module, fromlist=[cls]), cls)
    params = []
    for p in inspect.signature(est_cls.__init__).parameters.values():
        if p.name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        d = p.default
        kind = "text"
        if isinstance(d, bool):
            kind = "bool"
        elif isinstance(d, int):
            kind = "int"
        elif isinstance(d, float):
            kind = "float"
        elif d is None:
            kind = "text"
        params.append({"name": p.name, "default": None if d is inspect.Parameter.empty else d, "kind": kind})
    return jsonify({"name": name, "editable": True, "estimator": cls, "params": params})
