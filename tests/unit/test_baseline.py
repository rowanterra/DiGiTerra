"""Baseline comparison: training mean for regression, majority class for classification."""

import numpy as np
import pandas as pd

from python_scripts.preprocessing.baseline import classification_baseline, regression_baseline
from tests.unit.test_pipelines import _train


def test_mean_baseline_scores_near_zero_r2_and_flags_a_weaker_model():
    rng = np.random.default_rng(0)
    y_train = pd.Series(rng.normal(10, 2, size=200))
    y_test = pd.Series(rng.normal(10, 2, size=50))
    info = regression_baseline(y_train, y_test, model_r2=-0.3)
    assert info["metric"] == "R²"
    assert abs(info["baseline_value"]) < 0.2
    assert info["model_beats_baseline"] is False
    assert regression_baseline(y_train, y_test, model_r2=0.8)["model_beats_baseline"] is True


def test_mean_baseline_handles_two_targets():
    y_train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
    y_test = pd.DataFrame({"a": [2.0, 3.0], "b": [20.0, 30.0]})
    info = regression_baseline(y_train, y_test, None)
    assert info["training_mean"] == [2.5, 25.0]
    assert info["model_beats_baseline"] is None


def test_majority_baseline_reports_the_class_share():
    y_train = ["a"] * 70 + ["b"] * 30
    y_test = ["a"] * 7 + ["b"] * 3
    info = classification_baseline(y_train, y_test, model_accuracy=0.7)
    assert info["majority_class"] == "a"
    assert info["baseline_value"] == 0.7
    assert info["baseline_balanced_accuracy"] == 0.5
    assert info["model_beats_baseline"] is False


def test_training_result_includes_the_baseline_for_both_problem_types():
    n = len(pd.read_csv("examples/diabetes.csv").columns)
    result, _ = _train({}, "examples/diabetes.csv", list(range(n - 1)), [n - 1], "Linear")
    base = result["baseline"]
    assert base["kind"] == "regression" and base["model_beats_baseline"] is True
    assert result["data_checks"]["blocking"] == []

    result, _ = _train({}, "examples/iris.csv", [0, 1, 2, 3], [4], "Logistic_classifier")
    base = result["baseline"]
    assert base["kind"] == "classification" and base["model_beats_baseline"] is True
