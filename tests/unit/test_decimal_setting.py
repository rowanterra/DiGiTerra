"""The decimal-places setting reaches the numbers the page and manifest report.

Found by the E1 differential test (2026-09-24): with the setting at 10 the
regression metrics still came back rounded to three decimals and padded
("53.8080000000" for 53.80842980371550), and the descriptive statistics were
rounded to four decimals, which printed the centered diabetes columns as -0.0.
"""

import numpy as np
import pandas as pd

from python_scripts.app_exploration import DESCRIPTIVE_DECIMALS
from python_scripts.preprocessing.utilities import regression_report


def test_regression_report_honours_the_decimal_setting():
    y = pd.DataFrame({"t": [1.0, 2.0, 3.0, 4.0]})
    pred = pd.DataFrame({"t": [1.1, 2.3, 2.7, 4.4]})
    ten = regression_report(y, pred, target_names=["t"], sigfig=10)
    three = regression_report(y, pred, target_names=["t"], sigfig=3)
    rmse_ten = float(ten["RMSE"].iloc[-1])
    rmse_three = float(three["RMSE"].iloc[-1])
    exact = float(np.sqrt(np.mean((y["t"] - pred["t"]) ** 2)))
    assert rmse_three == round(exact, 3)
    assert abs(rmse_ten - exact) < 1e-9
    assert rmse_ten != rmse_three


def test_pipeline_metrics_carry_the_request_decimals():
    from tests.unit.test_pipelines import _train

    store = {}
    result, _ = _train(store, "examples/diabetes.csv", list(range(10)), [10], "Linear", sigfig=8, scaler="none")
    rmse = result["valrmse"]
    assert len(rmse.split(".")[1]) == 8
    # a value rounded to three decimals and padded would end in five zeros
    assert not rmse.endswith("00000"), rmse


def test_descriptive_statistics_are_not_rounded_to_four_and_carry_no_negative_zero(tmp_path):
    from python_scripts.app_exploration import handle_corr

    n = 50
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    df = pd.DataFrame({"a": x - x.mean(), "b": rng.normal(size=n) * 1e-3, "y": rng.normal(size=n)})
    store = {"data": df, "filename": "t.csv"}
    resp, status = handle_corr(
        store,
        {
            "colsIgnore": [0, 1, 2],
            "dropMissing": "none",
            "imputeStrategy": "none",
            "dropZero": "none",
            "zeroHandle": "none",
        },
        tmp_path,
        lambda s: s,
    )
    assert status == 200, resp
    rows = {r["column"]: r for r in resp["descriptive_stats"]}
    assert DESCRIPTIVE_DECIMALS >= 6
    assert str(rows["a"]["mean"]) != "-0.0"
    assert abs(rows["b"]["std"] - df["b"].std()) < 10 ** (-DESCRIPTIVE_DECIMALS)
    assert rows["b"]["std"] != round(df["b"].std(), 4)


def test_knn_classifier_form_values_reach_the_estimator():
    """E1 finding F3: n_neighbors typed on the Advanced form was dropped by get_model_kwargs."""
    from python_scripts.model_registry import get_model_kwargs
    from tests.unit.test_pipelines import _train

    assert get_model_kwargs("KNeighbors_classifier", {"n_neighbors": 3, "weights": "distance", "p": ""}, True, 0) == {
        "n_neighbors": 3,
        "weights": "distance",
    }
    store = {}
    _train(
        store,
        "examples/iris.csv",
        [0, 1, 2, 3],
        [4],
        "KNeighbors_classifier",
        hyperparameters={"n_neighbors": 3},
        modelingMode="advanced",
    )
    model = store["model"]
    est = getattr(model, "_digiterra_model", model)
    assert est.get_params()["n_neighbors"] == 3


def test_cleared_advanced_fields_do_not_reach_the_estimator_as_none():
    """E1 finding F5: a cleared Advanced field arrives as null and must fall back to the default."""
    from python_scripts.model_registry import get_model_kwargs
    from tests.unit.test_pipelines import _train

    kw = get_model_kwargs("Logistic_classifier", {"Class_CLogistic": 0.5, "Class_LogisticMaxIterations": 500}, True, 0)
    assert None not in kw.values()
    assert kw["Class_CLogistic"] == 0.5 and kw["Class_LogisticMaxIterations"] == 500
    # the sweep harness reads the private form, which keeps the None keys
    from python_scripts.model_registry import _model_kwargs

    assert None in _model_kwargs("Ridge", {}, False, 0).values()
    assert None not in get_model_kwargs("Ridge", {}, False, 0).values()
    store = {}
    _train(
        store,
        "examples/iris.csv",
        [0, 1, 2, 3],
        [4],
        "Logistic_classifier",
        hyperparameters={"Class_CLogistic": 0.5, "Class_LogisticMaxIterations": 500, "Class_LogisticTol": None},
        nonreq=True,
        modelingMode="advanced",
    )
    est = getattr(store["model"], "_digiterra_model", store["model"])
    assert est.get_params()["C"] == 0.5 and est.get_params()["max_iter"] == 500
