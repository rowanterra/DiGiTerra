"""Data checks that run before training: leakage, degenerate columns, small data."""

import numpy as np
import pandas as pd
import pytest

from python_scripts.preprocessing.guardrails import run_guardrails
from python_scripts.preprocessing.progress_tracker import get_result, remove_tracker
from tests.unit.test_pipelines import BASE


def _codes(report):
    return {f.code for f in report.findings}


def test_clean_regression_frame_has_no_blocking_findings():
    df = pd.read_csv("examples/diabetes.csv")
    cols = list(df.columns)
    report = run_guardrails(df, cols[:-1], [cols[-1]], "regression")
    assert report.blocking == []


def test_copy_of_the_target_is_blocked():
    df = pd.read_csv("examples/diabetes.csv")
    df["leak"] = df.iloc[:, -1] * 2 + 1
    cols = [c for c in df.columns if c != df.columns[-2]]
    report = run_guardrails(df, cols, [df.columns[-2]], "regression")
    assert "target_leakage" in {f.code for f in report.blocking}
    assert "leak" in report.block_message()


def test_target_listed_as_a_feature_is_blocked():
    df = pd.read_csv("examples/diabetes.csv")
    cols = list(df.columns)
    report = run_guardrails(df, cols, [cols[-1]], "regression")
    assert "target_as_feature" in {f.code for f in report.blocking}


def test_categorical_column_that_maps_one_to_one_onto_the_class_is_blocked():
    df = pd.read_csv("examples/iris.csv")
    target = df.columns[-1]
    df["code"] = pd.factorize(df[target])[0].astype(str)
    features = [c for c in df.columns if c != target]
    report = run_guardrails(df, features, [target], "classification")
    assert "target_leakage" in {f.code for f in report.blocking}


def test_constant_identifier_and_missing_columns_warn():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "const": 1.0,
            "row_id": np.arange(n),
            "sparse": np.where(rng.random(n) < 0.7, np.nan, 1.0),
            "x": rng.normal(size=n),
        }
    )
    df["y"] = df["x"] * 3 + rng.normal(size=n)
    report = run_guardrails(df, ["const", "row_id", "sparse", "x"], ["y"], "regression")
    codes = _codes(report)
    assert {"constant_columns", "identifier_columns", "heavy_missingness"} <= codes
    assert report.blocking == []


def test_tiny_frame_and_rare_class_warn():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(size=20), "y": ["a"] * 18 + ["b", "b"]})
    report = run_guardrails(df, ["x"], ["y"], "classification", test_size=0.2, cv_folds=5)
    codes = _codes(report)
    assert "few_rows" in codes
    assert "small_test_set" in codes
    assert "class_imbalance" in codes or "class_smaller_than_folds" in codes


def test_clustering_skips_target_checks():
    df = pd.read_csv("examples/iris.csv")
    report = run_guardrails(df, list(df.columns[:-1]), [], "cluster")
    assert "target_leakage" not in _codes(report)


@pytest.mark.parametrize("allow_unsafe, expect_error", [(None, True), ("true", False)])
def test_training_request_honors_the_block_and_the_override(allow_unsafe, expect_error, monkeypatch):
    from core import constants
    from python_scripts.app_model_training import run_model_training

    monkeypatch.setattr(constants, "ALLOW_UNSAFE_OVERRIDE", True)

    df = pd.read_csv("examples/diabetes.csv")
    df["leak"] = df.iloc[:, -1]
    store = {"data": df}
    n = len(df.columns)
    target = n - 2
    payload = dict(BASE, indicators=[i for i in range(n) if i != target], predictors=[target], models="Linear")
    if allow_unsafe is not None:
        payload["allowUnsafe"] = allow_unsafe
        payload["allowUnsafeReason"] = "leak column kept on purpose for a sensitivity check"
    sid = f"pytest-guardrails-{allow_unsafe}"
    run_model_training(sid, payload, "store", lambda _s: store)
    result = get_result(sid)
    remove_tracker(sid)
    assert result is not None
    if expect_error:
        assert "error" in result
        assert "leak" in result["error"]
    else:
        assert "error" not in result
        checks = result["data_checks"]
        assert checks["allow_unsafe"] is True and checks["forced"] is True
        assert checks["decision_use"] == "not valid for decision use"
        assert checks["override_reason"].startswith("leak column")
        assert any(f["code"] == "target_leakage" for f in checks["blocking"])


def test_override_is_a_server_side_switch_and_needs_a_reason(monkeypatch):
    from core import constants
    from python_scripts.app_model_training import validate_training_request

    base = dict(BASE, indicators=[0], predictors=[1], models="Linear", allowUnsafe="true")
    monkeypatch.setattr(constants, "ALLOW_UNSAFE_OVERRIDE", False)
    with pytest.raises(ValueError, match="not available on this deployment"):
        validate_training_request(dict(base, allowUnsafeReason="a perfectly good reason"))
    monkeypatch.setattr(constants, "ALLOW_UNSAFE_OVERRIDE", True)
    with pytest.raises(ValueError, match="allowUnsafeReason"):
        validate_training_request(dict(base, allowUnsafeReason="short"))
    validate_training_request(dict(base, allowUnsafeReason="a perfectly good reason"))
    # Without the flag the reason is irrelevant and the request is ordinary.
    validate_training_request(dict(BASE, indicators=[0], predictors=[1], models="Linear"))


def test_forced_run_is_labeled_in_the_model_card():
    from python_scripts.reporting.bundle import BUNDLE_FORMAT, render_model_card

    manifest = {
        "format": BUNDLE_FORMAT,
        "model": {"name": "Linear", "features": ["a"], "targets": ["b"]},
        "data": {"filename": "x.csv"},
        "metrics": {},
        "environment": {"packages": {}},
        "data_checks": {
            "forced": True,
            "override_reason": "known proxy kept for a sensitivity check",
            "blocking": [{"code": "target_leakage", "message": "'a' is identical to the target 'b'."}],
            "warnings": [],
        },
    }
    card = render_model_card(manifest)
    assert "not valid for decision use" in card
    assert "known proxy kept" in card


def test_the_block_message_offers_the_override_only_where_it_exists(monkeypatch):
    """A served page must not name a control the operator has not turned on."""
    from core import constants
    from python_scripts.preprocessing.guardrails import GuardrailReport

    report = GuardrailReport()
    report.add("target_as_feature", "block", "'leak' is identical to the target 'y'.", ["leak"])

    monkeypatch.setattr(constants, "ALLOW_UNSAFE_OVERRIDE", True)
    with_override = report.block_message()
    assert "Data Checks" in with_override, with_override
    assert "Ask the operator" not in with_override, with_override

    monkeypatch.setattr(constants, "ALLOW_UNSAFE_OVERRIDE", False)
    without = report.block_message()
    assert "Data Checks" not in without, without
    assert "Ask the operator" in without, without

    # Either way the finding comes before anything said about the override.
    for message in (with_override, without):
        assert message.startswith("DiGiTerra stopped the run."), message
        finding = message.index("'leak' is identical to the target 'y'.")
        tail = max(message.find("Data Checks"), message.find("Ask the operator"))
        assert tail > finding, message
