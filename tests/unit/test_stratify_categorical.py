"""Stratify on a categorical target, which is what the tutorial tells a reader to do.

The split held the class proportions only when the column was first cut into bins
or quantiles. A categorical column needs neither: the values are already the groups,
which is what train_test_split's own stratify argument takes. The client refused the
request with "Select bins or quantiles", the route called it a target leak, and the
pipeline fell through to an unstratified split.
"""

import uuid
import warnings

import pandas as pd
import pytest

from core.flask_app import app
from python_scripts.app_model_training import run_model_training
from python_scripts.preprocessing.progress_tracker import get_result, remove_tracker

warnings.filterwarnings("ignore")

PENGUINS = "examples/penguins.csv"


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_the_route_accepts_a_categorical_target_as_the_stratify_column(client):
    """Naming the target is correct for classification, not a leak."""
    client.set_cookie("digiterra_session", uuid.uuid4().hex)
    with open(PENGUINS, "rb") as fh:
        assert (
            client.post("/upload", data={"file": (fh, "penguins.csv")}, content_type="multipart/form-data").status_code
            == 200
        )
    r = client.post(
        "/preprocess",
        json={
            "indicators": [1, 2, 3, 4, 5, 6],
            "predictors": [0],
            "stratify": 0,
            "outputType": "Classifier",
            "quantileBins": "None",
        },
    )
    assert r.status_code == 200, r.get_json()


def test_the_route_still_refuses_the_target_for_a_regression_split(client):
    """The guard belongs to regression, where the target is cut into bins."""
    client.set_cookie("digiterra_session", uuid.uuid4().hex)
    with open(PENGUINS, "rb") as fh:
        client.post("/upload", data={"file": (fh, "penguins.csv")}, content_type="multipart/form-data")
    r = client.post(
        "/preprocess",
        json={
            "indicators": [1, 2, 3, 4],
            "predictors": [5],
            "stratify": 5,
            "outputType": "Numeric",
            "quantileBins": "quantiles",
        },
    )
    assert r.status_code == 400
    assert "leaks the target" in r.get_json()["error"]


def test_a_categorical_stratify_column_runs_end_to_end():
    """The whole path accepts it: route, training guard and pipeline."""
    frame = pd.read_csv(PENGUINS).dropna()
    store = {"data": frame}
    sid = f"pytest-strat-{uuid.uuid4().hex[:8]}"
    payload = dict(
        hyperparameters={},
        nonreq=False,
        scaler="standard",
        units="",
        sigfig=3,
        seedValue=42,
        testSize="0.20",
        stratifyBool=True,
        stratifyColumn=0,  # column A, species
        dropMissing="none",
        imputeStrategy="none",
        dropZero="none",
        # Neither bins nor quantiles: the categorical case this fixes.
        quantileBinDict={"quantile": 0, "bin": 0, "binsLabel": ""},
        useTransformer="No",
        transformerCols=[],
        filename="penguins.csv",
        crossValidationType="None",
        modelingMode="simple",
        # The four numeric measurements. One-hot encoding is not what this checks.
        indicators=[2, 3, 4, 5],
        predictors=[0],
        models="Logistic_classifier",
        outputType="Classifier",
    )
    run_model_training(sid, payload, "store", lambda _s: store)
    result = get_result(sid)
    remove_tracker(sid)
    assert result is not None and "error" not in result, result

    assert result.get("support"), "the run produced no test rows"


def test_the_pipeline_passes_the_column_values_as_the_stratify_series(monkeypatch):
    """The proof that it stratifies, at the point the split is made.

    The result records no per-class support, so the run alone cannot show whether the
    proportions were held. This captures what split_indices receives: with a
    categorical column and no bins it must be the column's own values, not None.
    """
    from python_scripts.preprocessing import run_classification_pipeline as pipeline

    seen = {}
    real = pipeline.split_indices

    def capture(X, test_size, seed, strat=None, **kwargs):
        seen["strat"] = strat
        return real(X, test_size, seed, strat=strat, **kwargs)

    monkeypatch.setattr(pipeline, "split_indices", capture)

    frame = pd.read_csv(PENGUINS).dropna()
    sid = f"pytest-strat-{uuid.uuid4().hex[:8]}"
    payload = dict(
        hyperparameters={},
        nonreq=False,
        scaler="standard",
        units="",
        sigfig=3,
        seedValue=42,
        testSize="0.20",
        stratifyBool=True,
        stratifyColumn=0,
        dropMissing="none",
        imputeStrategy="none",
        dropZero="none",
        quantileBinDict={"quantile": 0, "bin": 0, "binsLabel": ""},
        useTransformer="No",
        transformerCols=[],
        filename="penguins.csv",
        crossValidationType="None",
        modelingMode="simple",
        indicators=[2, 3, 4, 5],
        predictors=[0],
        models="Logistic_classifier",
        outputType="Classifier",
    )
    run_model_training(sid, payload, "store", lambda _s: {"data": frame.copy()})
    remove_tracker(sid)

    strat = seen.get("strat")
    assert strat is not None, "the split was not stratified"
    assert set(pd.Series(strat).unique()) == {"Adelie", "Gentoo", "Chinstrap"}, (
        f"the stratify series is not the species column: {pd.Series(strat).unique()}"
    )
