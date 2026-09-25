"""The guided page and the two small routes it needs.

/guided serves one page with a Guided and a Full mode. /column-types tells
the page which uploaded columns are numeric and how many distinct values
each holds, so it can propose a target and a task. /model-defaults/<name>
returns the estimator's own parameter names and defaults for every trainable
model, so the page's hyperparameter table cannot drift from the library.
"""

from __future__ import annotations

import functools
import inspect

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, render_template

from core.constants import HTTP_BAD_REQUEST, HTTP_NOT_FOUND
from core.helpers import with_prefix
from core.state import get_session_storage
from python_scripts import config
from python_scripts.estimator_params import RUN_SEED, built_estimator
from python_scripts.model_registry import _PASSTHROUGH_CLASSIFIERS, MODEL_REGISTRY

bp = Blueprint("guided", __name__)

# Registry name -> scikit-learn class of the estimator the trainer builds. The
# classifiers in model_registry._PASSTHROUGH_CLASSIFIERS read these names through the
# trainer; every other model has the typed values set on the built estimator by
# python_scripts.estimator_params. Multi-output wrappers take the inner class.
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
    "Linear": ("sklearn.linear_model", "LinearRegression"),
    "BayesianRidge": ("sklearn.linear_model", "BayesianRidge"),
    "ARDRegression": ("sklearn.linear_model", "ARDRegression"),
    "Ridge": ("sklearn.linear_model", "Ridge"),
    "Lasso": ("sklearn.linear_model", "Lasso"),
    "ElasticNet": ("sklearn.linear_model", "ElasticNet"),
    "SVM": ("sklearn.svm", "SVR"),
    "RF": ("sklearn.ensemble", "RandomForestRegressor"),
    "ExtraTrees": ("sklearn.ensemble", "ExtraTreesRegressor"),
    "MLP": ("sklearn.neural_network", "MLPRegressor"),
    "K-Nearest": ("sklearn.neighbors", "KNeighborsRegressor"),
    "gradient_boosting": ("sklearn.ensemble", "GradientBoostingRegressor"),
    "AdaBoost": ("sklearn.ensemble", "AdaBoostRegressor"),
    "Bagging": ("sklearn.ensemble", "BaggingRegressor"),
    "DecisionTree": ("sklearn.tree", "DecisionTreeRegressor"),
    "ElasticNetCV": ("sklearn.linear_model", "ElasticNetCV"),
    "HistGradientBoosting": ("sklearn.ensemble", "HistGradientBoostingRegressor"),
    "Huber": ("sklearn.linear_model", "HuberRegressor"),
    "LARS": ("sklearn.linear_model", "Lars"),
    "LARSCV": ("sklearn.linear_model", "LarsCV"),
    "LassoCV": ("sklearn.linear_model", "LassoCV"),
    "LassoLars": ("sklearn.linear_model", "LassoLars"),
    "LinearSVR": ("sklearn.svm", "LinearSVR"),
    "NuSVR": ("sklearn.svm", "NuSVR"),
    "OMP": ("sklearn.linear_model", "OrthogonalMatchingPursuit"),
    "PassiveAggressive": ("sklearn.linear_model", "PassiveAggressiveRegressor"),
    "Quantile": ("sklearn.linear_model", "QuantileRegressor"),
    "RadiusNeighbors": ("sklearn.neighbors", "RadiusNeighborsRegressor"),
    "RANSAC": ("sklearn.linear_model", "RANSACRegressor"),
    "RidgeCV": ("sklearn.linear_model", "RidgeCV"),
    "SGD": ("sklearn.linear_model", "SGDRegressor"),
    "TheilSen": ("sklearn.linear_model", "TheilSenRegressor"),
    "Logistic_classifier": ("sklearn.linear_model", "LogisticRegression"),
    "MLP_classifier": ("sklearn.neural_network", "MLPClassifier"),
    "RF_classifier": ("sklearn.ensemble", "RandomForestClassifier"),
    "SVC_classifier": ("sklearn.svm", "SVC"),
    "kmeans": ("sklearn.cluster", "KMeans"),
    "gmm": ("sklearn.mixture", "GaussianMixture"),
    "agglo": ("sklearn.cluster", "AgglomerativeClustering"),
    "dbscan": ("sklearn.cluster", "DBSCAN"),
    "birch": ("sklearn.cluster", "Birch"),
    "spectral": ("sklearn.cluster", "SpectralClustering"),
    "affinity_propagation": ("sklearn.cluster", "AffinityPropagation"),
    "bisecting_kmeans": ("sklearn.cluster", "BisectingKMeans"),
    "hdbscan": ("sklearn.cluster", "HDBSCAN"),
    "meanshift": ("sklearn.cluster", "MeanShift"),
    "minibatch_kmeans": ("sklearn.cluster", "MiniBatchKMeans"),
    "optics": ("sklearn.cluster", "OPTICS"),
    "Polynomial": ("python_scripts.models.regression_models.train_polynomial", "PolynomialRegression"),
    "Perceptron": ("sklearn.linear_model", "Perceptron"),
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
        out[task].append({"name": name, "editable": name in _SKLEARN_CLASS})
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


@functools.cache
def _built_params(name: str) -> dict:
    """get_params of the estimator the trainer builds from an empty form."""
    return built_estimator(name).get_params(deep=False)


def _display(param: str, value):
    """A JSON value for the Default column; the run seed is named, objects are written as repr."""
    if param == "random_state" and value == RUN_SEED:
        return "run seed"
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return repr(value)


@bp.route("/model-defaults/<name>")
def model_defaults(name: str):
    """Parameter names, kinds and the values a blank field leaves, for one model."""
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
    # The Default column shows the value a blank field leaves in place, which is the
    # value the trainer sets, not always scikit-learn's (see built_estimator).
    built = _built_params(name)
    params = []
    for p in inspect.signature(est_cls.__init__).parameters.values():
        if p.name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        library = None if p.default is inspect.Parameter.empty else p.default
        value = built.get(p.name, library)
        # The kind follows the value a blank field leaves; the input type must accept it.
        probe = value if value is not None else library
        kind = "text"
        if isinstance(probe, bool):
            kind = "bool"
        elif isinstance(probe, (int, np.integer)):
            kind = "int"
        elif isinstance(probe, (float, np.floating)):
            kind = "float"
        params.append({"name": p.name, "default": _display(p.name, value), "kind": kind})
    return jsonify(
        {
            "name": name,
            "editable": True,
            "estimator": cls,
            "params": params,
            # The Advanced-form path reads these names for the pass-through classifiers;
            # every model also takes them by scikit-learn name as estimatorParams.
            "passthrough": name in _PASSTHROUGH_CLASSIFIERS,
        }
    )
