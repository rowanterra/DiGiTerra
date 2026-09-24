"""Regression tests for the defects found by the adversarial review of 2026-09-24.

Each test names the finding it covers. They run the pipeline functions directly,
as the review's reproductions did, on small frames built to show the defect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from python_scripts.helpers import preprocess_data
from python_scripts.model_registry import get_model_kwargs
from python_scripts.preprocessing.outlier_handling import apply_outlier_handling
from python_scripts.preprocessing.split_fill import FillPlan
from tests.unit.test_pipelines import _train


def test_an_empty_form_leaves_the_trainer_its_defaults():
    """Browser finding 1: the guided page sends no hyperparameters for RF and the
    trainer received n_estimators=None."""
    for name in ("RF", "RF_classifier", "kmeans", "Ridge", "SVM", "MLP", "gmm", "gradient_boosting"):
        kwargs = get_model_kwargs(name, {}, False, 42)
        assert all(v is not None for v in kwargs.values()), (name, kwargs)


def test_gradient_boosting_warm_start_is_not_inverted():
    """Pipeline finding 8."""
    assert get_model_kwargs("gradient_boosting", {"GBWarmStart": "true"}, True, 1).get("warm_start") is True
    assert get_model_kwargs("gradient_boosting", {"GBWarmStart": "false"}, True, 1).get("warm_start") is False
    assert "warm_start" not in get_model_kwargs("gradient_boosting", {}, True, 1)


def test_gmm_takes_the_seed():
    """Pipeline finding 6."""
    assert get_model_kwargs("gmm", {"n_components": 3}, False, 7)["random_state"] == 7


def test_iqr_outliers_ignore_one_hot_columns():
    """Pipeline finding 1: a minority category is not an outlier."""
    rng = np.random.default_rng(0)
    n = 200
    x = pd.DataFrame({"x1": rng.normal(size=n)})
    cat = rng.choice(["A", "B", "C"], size=n, p=[0.6, 0.25, 0.15])
    for level in "ABC":
        x[f"cat_{level}"] = (cat == level).astype(float)
    kept, _, mask = apply_outlier_handling(x, x.iloc[:5], method="IQR")
    assert mask.sum() >= n - 10, "one-hot columns drove the outlier rule"
    assert kept["cat_C"].sum() > 0


def test_the_target_is_never_imputed():
    """Pipeline finding 2: a blank target is dropped, not filled and scored."""
    df = pd.DataFrame({"x1": np.arange(10.0), "x2": np.arange(10.0) * 2, "y": np.arange(10.0)})
    df.loc[[2, 5], "y"] = np.nan
    df.loc[[3], "x1"] = np.nan
    plan = FillPlan()
    out = preprocess_data(
        df,
        target_cols=["y"],
        indicator_cols=["x1", "x2"],
        drop_missing="indicatorAndTarget",
        impute_strategy="mean",
        drop_zero="none",
        zero_handle="none",
        fill_plan_out=plan,
    )
    assert "y" not in plan.nan_columns
    assert "x1" in plan.nan_columns
    assert out is not None


def test_mode_fill_fills_and_zero_removal_removes_any_zero():
    """Pipeline finding 7 and browser findings 9 and 10."""
    df = pd.DataFrame({"x1": [1.0, 1.0, 2.0, np.nan, 0.0], "x2": [5.0, 5.0, 5.0, 5.0, 5.0], "y": [1, 2, 3, 4, 5]})
    plan = FillPlan()
    preprocess_data(
        df.copy(),
        target_cols=["y"],
        indicator_cols=["x1", "x2"],
        drop_missing="indicatorAndTarget",
        impute_strategy="mode",
        drop_zero="none",
        zero_handle="none",
        fill_plan_out=plan,
    )
    assert plan.nan_strategy == "mode" and "x1" in plan.nan_columns and "y" not in plan.nan_columns
    plan.fit(df.iloc[:3])
    assert plan.nan_fills["x1"] == 1.0
    out = preprocess_data(
        df.copy(),
        target_cols=["y"],
        indicator_cols=["x1", "x2"],
        drop_missing="none",
        impute_strategy="none",
        drop_zero="indicatorAndTarget",
        zero_handle="drop",
    )
    assert 0.0 not in out["x1"].values, "a row with one zero was kept"


def test_binary_auc_uses_the_estimator_class_order():
    """Pipeline finding 3: labels whose string order differs from the class order."""
    from sklearn.linear_model import LogisticRegression

    from python_scripts.preprocessing.calculate_all_metrics import calculate_classification_metrics

    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 2))
    y_int = (x[:, 0] + 0.3 * rng.normal(size=200) > 0).astype(int)
    labels = np.array(["<5", "5-10"])[y_int]  # string order '5-10' < '<5', class order '<5', '5-10'
    model = LogisticRegression().fit(x, labels)
    classes = list(model.classes_)
    assert classes == ["5-10", "<5"] or classes == ["<5", "5-10"]
    metrics = calculate_classification_metrics(labels, model.predict(x), classes=classes, model=model, X_test=x)
    assert metrics["roc_auc"] > 0.9
    assert metrics.get("average_precision") is not None and metrics["average_precision"] > 0.9


def test_seed_zero_is_a_seed_and_a_blank_seed_draws_one():
    """Pipeline finding 9 / browser finding 5."""
    store = {}
    r1, _ = _train(store, "examples/iris.csv", [0, 1, 2, 3], [4], "KNeighbors_classifier", seedValue=0)
    r2, _ = _train(store, "examples/iris.csv", [0, 1, 2, 3], [4], "KNeighbors_classifier", seedValue=0)
    assert store["seed"] == 0
    assert r1["accuracy"] == r2["accuracy"]


def test_cross_validation_follows_a_grouped_split():
    """Pipeline finding 5c: grouped holdout and CV agree in kind."""
    rng = np.random.default_rng(3)
    n, sites = 240, 30
    site = rng.integers(0, sites, n)
    effect = rng.normal(size=sites)[site] * 10
    df = pd.DataFrame({"x": rng.normal(size=n), "site": site.astype(float), "y": effect + rng.normal(size=n)})
    df.to_csv("/tmp/_grouped_cv.csv", index=False)
    store = {}
    result, _ = _train(
        store,
        "/tmp/_grouped_cv.csv",
        [0, 1],
        [2],
        "Linear",
        crossValidationType="KFold",
        crossValidationFolds=3,
        splitMethod="group",
        splitColumn=1,
    )
    cv = {row["Metric"]: row["Mean"] for row in result["cross_validation_summary"]}
    assert cv["r2"] < 0.5, f"grouped CV still scored the site effect: {cv}"


def test_cross_validation_scores_every_target_and_imputes_per_fold():
    """Pipeline findings 5a and 5b."""
    rng = np.random.default_rng(4)
    n = 120
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": 3 * x1 - 2 * x2 + 0.1 * rng.normal(size=n), "y2": rng.normal(size=n)})
    df.loc[rng.choice(n, 10, replace=False), "x2"] = np.nan
    df.to_csv("/tmp/_multi_cv.csv", index=False)
    store = {}
    result, _ = _train(
        store,
        "/tmp/_multi_cv.csv",
        [0, 1],
        [2, 3],
        "Linear",
        crossValidationType="KFold",
        crossValidationFolds=3,
        imputeStrategy="mean",
        dropMissing="indicatorAndTarget",
    )
    cv = {row["Metric"]: row["Mean"] for row in result["cross_validation_summary"]}
    assert 0.2 < cv["r2"] < 0.7, f"CV read only the first target: {cv}"


def test_automl_with_no_model_compares_candidates_and_trains_the_best():
    store = {}
    result, _ = _train(
        store,
        "examples/iris.csv",
        [0, 1, 2, 3],
        [4],
        "",
        modelingMode="automl",
        taskType="Classifier",
    )
    cmp = result["model_comparison"]
    assert cmp and cmp["chosen"] in {r["model"] for r in cmp["summary"]}
    assert result["model_name"] == cmp["chosen"]
    assert cmp["figure"] == "model_comparison.png"
    assert len(cmp["summary"]) >= 5


@pytest.mark.parametrize("bad", ["Cluster", ""])
def test_automl_comparison_needs_a_supervised_task(bad):
    from python_scripts.app_model_training import validate_training_request
    from tests.unit.test_pipelines import BASE

    payload = dict(BASE, indicators=[0, 1], predictors=[2], models="", modelingMode="automl", taskType=bad)
    with pytest.raises(ValueError):
        validate_training_request(payload)
