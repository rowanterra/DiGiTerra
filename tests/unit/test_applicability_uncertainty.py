"""What a prediction says about itself: how far its row is, and how much to trust it.

Both numbers are recorded at training time and read at prediction time, so the
checks that matter are that the recorded values describe the training rows, that a
row unlike them is reported as such, and that a bundle carries both across a session
boundary unchanged.

The harness drives the routes rather than the functions, because two earlier checks
passed against the wrong artifact: predictions.csv is written to a per-session
directory, so it is read back through /download, and /preprocess refuses a stratify
column that is the target unless outputType says the run is a classification.
"""

import io
import json
import uuid
import warnings
import zipfile

import pandas as pd
import pytest

from core.flask_app import app

warnings.filterwarnings("ignore")

RUN_SECONDS = 40


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _train(client, csv, indicators, predictors, model, output_type, stratify, units=""):
    """Upload, preprocess and train through the routes. Returns the session id."""
    sid = uuid.uuid4().hex
    client.set_cookie("digiterra_session", sid)
    with open(csv, "rb") as fh:
        assert (
            client.post(
                "/upload", data={"file": (fh, csv.split("/")[-1])}, content_type="multipart/form-data"
            ).status_code
            == 200
        )
    assert (
        client.post(
            "/preprocess",
            json={
                "indicators": indicators,
                "predictors": predictors,
                "stratify": stratify,
                "outputType": output_type,
                "quantileBins": "None",
            },
        ).status_code
        == 200
    )
    client.post(
        "/process",
        json=dict(
            hyperparameters={},
            nonreq=False,
            scaler="standard",
            units=units,
            sigfig=3,
            seedValue=33,
            testSize="0.20",
            stratifyBool=False,
            dropMissing="none",
            imputeStrategy="none",
            dropZero="none",
            quantileBinDict={"quantile": 0, "bin": 0, "binsLabel": ""},
            useTransformer="No",
            transformerCols=[],
            filename=csv.split("/")[-1],
            crossValidationType="None",
            modelingMode="simple",
            indicators=indicators,
            predictors=predictors,
            models=model,
        ),
    )
    _wait_for_model(client)
    return sid


def _wait_for_model(client):
    import time

    for _ in range(RUN_SECONDS * 2):
        if client.get("/export-bundle").status_code == 200:
            return
        time.sleep(0.5)
    raise AssertionError("the run did not finish")


def _predict(client, frame):
    r = client.post(
        "/predict",
        data={"predictFile": (io.BytesIO(frame.to_csv(index=False).encode()), "new.csv")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200, r.get_json()
    # Read it back through the route, which resolves the per-session directory.
    return pd.read_csv(io.BytesIO(client.get("/download/predictions.csv").data)), r.get_json()


# --- Classification -------------------------------------------------------


def test_penguins_carries_probabilities_and_a_novelty_score(client):
    frame = pd.read_csv("examples/penguins.csv").dropna()
    _train(client, "examples/penguins.csv", [2, 3, 4, 5], [0], "Logistic_classifier", "Classifier", 2)
    predictions, payload = _predict(client, frame)

    for column in ("novelty_score", "novel_row", "out_of_range_columns", "max_probability"):
        assert column in predictions.columns, sorted(predictions.columns)
    assert pd.api.types.is_float_dtype(predictions["novelty_score"])
    assert pd.api.types.is_bool_dtype(predictions["novel_row"])
    assert pd.api.types.is_float_dtype(predictions["max_probability"])
    assert [c for c in predictions.columns if c.startswith("probability_")], "no per-class probabilities"

    # The training rows re-presented: the threshold is the first percentile, so a
    # couple of percent is the most it should flag.
    flagged = float(predictions["novel_row"].mean())
    assert flagged <= 0.02, f"{flagged:.1%} of the training rows were called novel"
    assert payload["uncertainty_summary"]["kind"] == "probability"


def test_a_row_outside_the_training_range_is_reported(client):
    _train(client, "examples/penguins.csv", [2, 3, 4, 5], [0], "Logistic_classifier", "Classifier", 2)
    rows = pd.read_csv("examples/penguins_new.csv")
    rows.loc[0, "bill_length_mm"] = 90.0  # no penguin in the file is near this
    predictions, _payload = _predict(client, rows)

    assert bool(predictions.loc[0, "novel_row"]), "a bill of 90 mm was not called novel"
    listed = str(predictions.loc[0, "out_of_range_columns"])
    assert "bill_length_mm" in listed, listed
    # The rows that were left alone are not swept up with it.
    assert not bool(predictions.loc[1, "novel_row"])


# --- Regression -----------------------------------------------------------


def test_diabetes_intervals_cover_the_validation_targets(client):
    frame = pd.read_csv("examples/diabetes.csv")
    _train(client, "examples/diabetes.csv", list(range(10)), [10], "Linear", "Numeric", 0, units="mg/dL")
    predictions, payload = _predict(client, frame)

    for column in ("lower_90", "upper_90", "novelty_score"):
        assert column in predictions.columns, sorted(predictions.columns)
        assert pd.api.types.is_float_dtype(predictions[column]) or column == "novel_row"

    truth = frame.iloc[:, 10].to_numpy()
    inside = ((truth >= predictions["lower_90"]) & (truth <= predictions["upper_90"])).mean()
    assert inside >= 0.85, f"the 90 percent interval covered {inside:.1%}"

    assert payload["uncertainty_summary"]["kind"] == "interval"
    assert payload["uncertainty_summary"]["half_width"] > 0


# --- The bundle carries both ----------------------------------------------


def test_a_bundle_reproduces_the_scores_and_the_interval_in_a_fresh_session(client, monkeypatch):
    monkeypatch.setenv("DIGITERRA_ALLOW_BUNDLE_IMPORT", "1")
    frame = pd.read_csv("examples/diabetes.csv")
    _train(client, "examples/diabetes.csv", list(range(10)), [10], "Linear", "Numeric", 0)
    first, _ = _predict(client, frame.iloc[:20, :10])
    bundle = client.get("/export-bundle").data

    manifest = json.loads(zipfile.ZipFile(io.BytesIO(bundle)).read("manifest.json"))
    assert manifest["applicability"]["detector"] == "IsolationForest"
    assert manifest["uncertainty"]["method"] == "split conformal"

    # A session that has trained nothing, which then imports the bundle.
    fresh = uuid.uuid4().hex
    client.set_cookie("digiterra_session", fresh)
    r = client.post(
        "/import-bundle",
        data={"bundle": (io.BytesIO(bundle), "bundle.zip")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200, r.get_json()

    second, _ = _predict(client, frame.iloc[:20, :10])
    pd.testing.assert_series_equal(first["novelty_score"], second["novelty_score"], check_names=False, rtol=1e-9)
    width_before = (first["upper_90"] - first["lower_90"]).iloc[0]
    width_after = (second["upper_90"] - second["lower_90"]).iloc[0]
    assert width_before == pytest.approx(width_after, rel=1e-9)
