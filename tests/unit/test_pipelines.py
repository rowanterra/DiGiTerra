"""Fast end-to-end checks on the training pipelines through run_model_training.

Each test drives the same function the /process route calls, with a dict in
place of session storage, on the example CSVs. Models are the cheapest in each
family (Linear, LogisticRegression, KMeans) so the file stays quick.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from python_scripts.app_model_training import run_model_training
from python_scripts.helpers import _build_cv_summary
from python_scripts.preprocessing.progress_tracker import get_result, get_tracker, remove_tracker

warnings.filterwarnings("ignore")

BASE = dict(
    hyperparameters={}, nonreq=False, scaler="standard", units="", sigfig=3,
    seedValue=42, testSize="0.20", stratifyBool=False, dropMissing="none",
    imputeStrategy="none", dropZero="none",
    quantileBinDict={"quantile": 0, "bin": 0, "binsLabel": ""},
    useTransformer="No", transformerCols=[], filename="test.csv",
    crossValidationType="None", modelingMode="simple",
)

_counter = [0]


def _train(store, csv, indicators, predictors, model, **extra):
    """Run one training pass and return (result dict, tracker stages)."""
    if "data" not in store:
        store["data"] = pd.read_csv(csv)
    _counter[0] += 1
    sid = f"pytest-pipelines-{_counter[0]}"
    payload = dict(BASE, indicators=indicators, predictors=predictors, models=model)
    payload.update(extra)
    run_model_training(sid, payload, "store", lambda _s: store)
    result = get_result(sid)
    stages = {k: dict(v) for k, v in get_tracker(sid).stages.items()}
    remove_tracker(sid)
    assert result is not None
    assert "error" not in result, result.get("error")
    return result, stages


def test_cv_summary_error_metrics_are_positive():
    results = {
        "test_RMSE": np.array([-3.0, -2.0]),
        "test_MAE": np.array([-1.5, -1.0]),
        "test_R2": np.array([0.8, 0.9]),
    }
    scoring = {"RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error", "R2": "r2"}
    summary, _ = _build_cv_summary(results, scoring)
    means = dict(zip(summary["Metric"], summary["Mean"]))
    assert means["RMSE"] > 0 and means["MAE"] > 0
    assert means["RMSE"] == pytest.approx(2.5)


def test_regression_cv_summary_rmse_positive():
    store = {}
    result, stages = _train(store, "examples/synthetic_regression.csv", list(range(10)), [10], "Linear",
                            crossValidationType="KFold", crossValidationFolds=3)
    summary = result["cross_validation_summary"]
    assert summary, "cross_validation_summary missing"
    rmse = [row for row in summary if str(row.get("Metric", "")).upper() == "RMSE"]
    assert rmse and rmse[0]["Mean"] > 0
    assert stages["cross_validation"]["status"] == "completed"


def test_feature_selection_names_match_model_and_predict_works():
    store = {}
    result, _ = _train(store, "examples/synthetic_regression.csv", list(range(10)), [10], "Linear",
                       featureSelectionMethod="SelectKBest", featureSelectionK=4, modelingMode="advanced")
    model = store["model"]
    names = getattr(model, "_digiterra_model_features", None) or store["feature_order"]
    assert model.n_features_in_ == 4
    assert len(names) == model.n_features_in_
    info = result["feature_selection_info"]
    assert info and len(info.get("selected_features", names)) == 4

    # Predict on new rows the way the inference route does: scale first, then subset.
    X_new = store["data"].iloc[:5, :10]
    scaler = store["X_scaler"]
    scaled = pd.DataFrame(scaler.transform(X_new[list(scaler.feature_names_in_)]),
                          columns=scaler.feature_names_in_)
    preds = model.predict(scaled[names].to_numpy())
    assert preds.shape == (5,) or preds.shape == (5, 1)
    assert np.all(np.isfinite(preds))


def test_classification_with_integer_target_has_string_report_keys():
    iris = pd.read_csv("examples/iris.csv")
    iris[iris.columns[-1]] = pd.factorize(iris[iris.columns[-1]])[0]
    store = {"data": iris}
    result, _ = _train(store, None, [0, 1, 2, 3], [4], "Logistic_classifier")
    assert float(result["accuracy"]) > 0.5
    assert store["model_type"] == "classification"

    # The raw report, straight from the pipeline: class keys must be strings on both sides.
    from sklearn.linear_model import LogisticRegression

    from python_scripts.helpers import unpack_classification_result
    from python_scripts.preprocessing.run_classification_pipeline import run_classification
    target = iris.columns[-1]
    out = run_classification(LogisticRegression(max_iter=200), "Logistic_classifier", iris, [target], False,
                             iris.iloc[:, :4], iris[[target]], "", "", "standard", 42, 3,
                             dict(BASE["quantileBinDict"]), "No", [], 0.2)
    report = unpack_classification_result(out)[0]
    class_keys = [k for k in report if k not in ("accuracy", "macro avg", "weighted avg")]
    assert class_keys and all(isinstance(k, str) for k in class_keys)
    assert set(class_keys) == {"0", "1", "2"}


def test_clustering_best_k_matches_labels_and_agglo_centers_finite():
    store = {}
    result, _ = _train(store, "examples/iris_clustering.csv", [0, 1, 2, 3], [], "kmeans",
                       hyperparameters={"n_clusters": 3})
    assert result["best_k"] == len(np.unique(store["model"].labels_))

    from sklearn.cluster import AgglomerativeClustering

    from python_scripts.preprocessing.run_clustering_pipeline import run_clustering
    df = pd.read_csv("examples/iris_clustering.csv")
    out = run_clustering(AgglomerativeClustering(n_clusters=3), "agglo", df, df, "", "standard",
                         42, 3, dict(BASE["quantileBinDict"]), "No", [], 0.2, 2, 8)
    best_k, centers = out[8], out[9]
    assert best_k == len(np.unique(out[4].labels_))
    assert centers.shape[0] == best_k
    assert np.all(np.isfinite(centers))


def test_clustering_passes_outlier_handling_and_skips_supervised_stages():
    store = {}
    result, stages = _train(store, "examples/iris_clustering.csv", [0, 1, 2, 3], [], "kmeans",
                            hyperparameters={"n_clusters": 3}, outlierMethod="IQR", outlierAction="remove",
                            featureSelectionMethod="SelectKBest", featureSelectionK=2,
                            crossValidationType="KFold")
    info = result["outlier_info"]
    assert info and info["n_outliers"] >= 0 and info["method"] == "IQR"
    assert stages["outlier_handling"]["status"] == "completed"
    assert stages["feature_selection"]["status"] == "skipped"
    assert stages["feature_selection"]["message"] == "Not applicable to clustering"
    assert stages["cross_validation"]["status"] == "skipped"
    skipped = {s["option"] for s in result["skipped_options"]}
    assert {"feature_selection_method", "cross_validation"} <= skipped


def test_stale_y_scaler_cleared_after_cluster_run():
    store = {}
    _train(store, "examples/diabetes.csv", list(range(10)), [10], "Linear")
    assert store["y_scaler"] is not None
    store["data"] = pd.read_csv("examples/iris_clustering.csv")
    _train(store, None, [0, 1, 2, 3], [], "kmeans", hyperparameters={"n_clusters": 3})
    assert store["y_scaler"] is None
    assert store["model_type"] == "cluster"
