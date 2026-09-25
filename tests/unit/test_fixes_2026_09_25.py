"""Fixes of 2026-09-25, found while writing the guide of that date.

1. The guided page sent useTransformer "No", so text predictors were dropped
   without a message (penguins: island and sex). The page now sends the text
   predictors to the transformer; the browser test in
   tests/integration/test_browser_guided.py reads the fitted features back.
2. scripts/score_bundle.py pointed the application support directory at a
   temporary one, so the installation's key file was not found and every bundle
   was refused unless DIGITERRA_BUNDLE_KEY was set. The test suite never saw it
   because conftest sets DIGITERRA_APP_SUPPORT_DIR for every subprocess. The
   script also left out the novelty and range columns that the web app writes.
3. The guided hyperparameter table was editable for 20 of 67 models, and for
   Linear the typed values never reached the estimator. Every model now takes
   values by scikit-learn name as estimatorParams, set on the built estimator.
"""

import os
import shutil
import subprocess
import sys
import uuid
import warnings
from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

from python_scripts.estimator_params import (
    apply_estimator_params,
    coerce_value,
    reset_estimator_params,
    set_estimator_params,
)
from python_scripts.model_registry import MODEL_REGISTRY
from tests.unit.test_bundle import DIABETES, _train_in_session

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def client():
    from app import app

    app.config["TESTING"] = True
    return app.test_client()


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("3", 3),
        ("0.5", 0.5),
        ("None", None),
        ("true", True),
        ("False", False),
        ("(20, 10)", (20, 10)),
        ("sqrt", "sqrt"),
        (7, 7),
        (False, False),
    ],
)
def test_form_values_are_read_as_python_values(raw, expected):
    assert coerce_value(raw) == expected


def test_parameters_are_set_on_the_built_estimator():
    token = set_estimator_params({"alpha": "10", "fit_intercept": False, "tol": ""})
    try:
        model = apply_estimator_params(Ridge())
    finally:
        reset_estimator_params(token)
    assert model.alpha == 10 and model.fit_intercept is False and model.tol == Ridge().tol


def test_a_multi_output_wrapper_passes_the_values_to_the_inner_estimator():
    token = set_estimator_params({"C": 2.5})
    try:
        model = apply_estimator_params(MultiOutputRegressor(SVR()))
    finally:
        reset_estimator_params(token)
    assert model.estimator.C == 2.5


def test_a_name_the_estimator_does_not_have_stops_the_run():
    token = set_estimator_params({"RidgeFitIntersept": True})
    try:
        with pytest.raises(ValueError, match="Ridge has no parameter named RidgeFitIntersept"):
            apply_estimator_params(Ridge())
    finally:
        reset_estimator_params(token)


def test_every_trainable_model_has_an_editable_table(client):
    listed = client.get("/models").get_json()
    names = [m["name"] for group in listed.values() for m in group]
    trainable = [n for n, (fn, _) in MODEL_REGISTRY.items() if fn is not None]
    assert sorted(names) == sorted(trainable)
    assert all(m["editable"] for group in listed.values() for m in group)
    for name in trainable:
        d = client.get(f"/model-defaults/{name}").get_json()
        assert d["editable"] and d["params"], name
    assert client.get("/model-defaults/Ridge").get_json()["passthrough"] is False
    assert client.get("/model-defaults/KNeighbors_classifier").get_json()["passthrough"] is True


def test_estimator_params_reach_a_model_whose_trainer_uses_its_own_names():
    result = _train_in_session(uuid.uuid4().hex, {"estimatorParams": {"fit_intercept": False}})
    assert result["model_params"]["fit_intercept"] is False


def test_score_bundle_finds_the_installation_key_and_writes_the_web_columns(client, tmp_path):
    sid = uuid.uuid4().hex
    _train_in_session(sid)
    client.set_cookie("digiterra_session", sid)
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(client.get("/export-bundle").data)
    rows = tmp_path / "rows.csv"
    pd.read_csv(DIABETES).iloc[:8, :-1].to_csv(rows, index=False)

    # An environment like a user's: no support-directory or key variables, and the
    # installation's key file in the default support directory under a fresh home.
    env = {k: v for k, v in os.environ.items() if not k.startswith("DIGITERRA_")}
    home = tmp_path / "home"
    env.update(HOME=str(home), XDG_DATA_HOME=str(home / "share"), APPDATA=str(home / "appdata"))
    default_dir = subprocess.run(
        [sys.executable, "-c", "from python_scripts import config; print(config.APP_SUPPORT_DIR)"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        check=True,
    ).stdout.strip()
    Path(default_dir).mkdir(parents=True, exist_ok=True)
    from python_scripts import config

    shutil.copy(Path(config.APP_SUPPORT_DIR) / "bundle_signing.key", Path(default_dir) / "bundle_signing.key")

    out = tmp_path / "pred.csv"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "score_bundle.py"), str(bundle), str(rows), "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    pred = pd.read_csv(out)
    for col in ("novelty_score", "novel_row", "out_of_range_columns", "lower_90", "upper_90"):
        assert any(c.startswith(col) for c in pred.columns), col


def test_the_default_column_shows_what_a_blank_field_leaves(client):
    """Several trainers set values other than scikit-learn's; the table shows the trainer's."""
    rows = {p["name"]: p for p in client.get("/model-defaults/gradient_boosting").get_json()["params"]}
    assert rows["loss"]["default"] == "absolute_error"
    assert rows["random_state"]["default"] == "run seed"
    rows = {p["name"]: p for p in client.get("/model-defaults/RF").get_json()["params"]}
    assert rows["max_features"]["default"] == "sqrt" and rows["max_features"]["kind"] == "text"
    rows = {p["name"]: p for p in client.get("/model-defaults/MLP").get_json()["params"]}
    assert rows["max_iter"]["default"] == 10000


def test_every_trainer_builds_its_estimator_without_fitting():
    from python_scripts.estimator_params import built_estimator

    for name, (fn, _) in MODEL_REGISTRY.items():
        if fn is None:
            continue
        model = built_estimator(name)
        assert not any(k.endswith("_") and not k.startswith("_") for k in vars(model)), f"{name} was fitted"
