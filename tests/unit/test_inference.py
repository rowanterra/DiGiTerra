"""Inference on new rows after a run in the same session, for each problem type.

The regression path failed before this test existed: the scaler's
feature_names_in_ array was used in a boolean `or`, which raises for more than
one column, so every scaled regression model returned 500 from /predict.
"""

import io
import uuid
import warnings

import pandas as pd
import pytest

from core.state import get_session_storage
from python_scripts import config
from python_scripts.preprocessing.progress_tracker import get_result, remove_tracker
from tests.unit.test_pipelines import BASE

warnings.filterwarnings("ignore")


@pytest.fixture
def client():
    from app import app

    app.config["TESTING"] = True
    return app.test_client()


def _run(sid, csv, indicators, predictors, model, **extra):
    from python_scripts.app_model_training import run_model_training

    store = get_session_storage(sid)
    store.clear()
    store["data"] = pd.read_csv(csv)
    payload = dict(BASE, indicators=indicators, predictors=predictors, models=model)
    payload.update(extra)
    token = config.set_session_scope(sid)
    try:
        run_model_training(sid, payload, sid, lambda s: get_session_storage(s))
        result = get_result(sid)
    finally:
        config.reset_session_scope(token)
    remove_tracker(sid)
    assert result is not None and "error" not in result, result
    return store


def _post_rows(client, frame):
    buf = io.BytesIO(frame.to_csv(index=False).encode())
    return client.post("/predict", data={"predictFile": (buf, "new.csv")})


@pytest.mark.parametrize(
    "csv, indicators, predictors, model, scaler",
    [
        ("examples/diabetes.csv", list(range(10)), [10], "Linear", "standard"),
        ("examples/diabetes.csv", list(range(10)), [10], "Linear", "none"),
        ("examples/diabetes.csv", list(range(10)), [10], "RF", "minmax"),
        ("examples/iris.csv", [0, 1, 2, 3], [4], "Logistic_classifier", "standard"),
        ("examples/iris_clustering.csv", [0, 1, 2, 3], [], "kmeans", "standard"),
    ],
)
def test_predict_answers_after_training(client, csv, indicators, predictors, model, scaler):
    sid = uuid.uuid4().hex
    extra = {"scaler": scaler}
    if model == "kmeans":
        extra["hyperparameters"] = {"n_clusters": 3}
    if model == "RF":
        extra["hyperparameters"] = {"n_estimators": 20}
    _run(sid, csv, indicators, predictors, model, **extra)
    client.set_cookie("digiterra_session", sid)
    frame = pd.read_csv(csv).iloc[:6, indicators]
    r = _post_rows(client, frame)
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body["summary"] and body["summary"][0]["n"] == 6


def test_predict_names_the_missing_feature(client):
    sid = uuid.uuid4().hex
    _run(sid, "examples/diabetes.csv", list(range(10)), [10], "Linear")
    client.set_cookie("digiterra_session", sid)
    frame = pd.read_csv("examples/diabetes.csv").iloc[:6, :9]
    r = _post_rows(client, frame)
    assert r.status_code == 400
    message = r.get_json()["error"]
    # The refusal says what happened, then what to do about it, and names the column.
    assert "has no column for these features" in message, message
    assert "Add the columns" in message, message
    missing = pd.read_csv("examples/diabetes.csv").columns[9]
    assert missing in message, message
