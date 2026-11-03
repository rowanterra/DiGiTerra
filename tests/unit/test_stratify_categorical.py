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
    )
    run_model_training(sid, payload, "store", lambda _s: store)
    result = get_result(sid)
    remove_tracker(sid)
    assert result is not None and "error" not in result, result

    assert result.get("support"), "the run produced no test rows"


def test_the_split_receives_the_target_as_the_stratify_series(monkeypatch):
    """The proof that it stratifies, at the call that does it.

    The result records no per-class support, so a finished run cannot show whether
    the proportions were held. This captures the stratify argument sklearn actually
    receives: with penguins column A and no bins it must be the species values.
    """
    from python_scripts.preprocessing import splitting

    seen = {}
    real = splitting.train_test_split

    def capture(*args, **kwargs):
        seen["stratify"] = kwargs.get("stratify")
        return real(*args, **kwargs)

    monkeypatch.setattr(splitting, "train_test_split", capture)

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
        stratifyColumn=0,  # column A, species
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
    )
    run_model_training(sid, payload, "store", lambda _s: {"data": frame.copy()})
    remove_tracker(sid)

    # The page's payload has no outputType key, so this one does not either: the
    # task has to be read from the registry entry for the chosen model.
    strat = seen.get("stratify")
    assert strat is not None, "train_test_split was called without a stratify series"
    captured = pd.Series(strat).reset_index(drop=True)
    expected = frame["species"].reset_index(drop=True)
    assert captured.equals(expected), "the stratify series is not the target column"


def test_automl_classification_may_stratify_on_its_target(monkeypatch):
    """The guard reads the task from the registry entry for the chosen model.

    AutoML sets modelingMode but still names a registry model, so the exemption has
    to hold there too. A model the registry does not know would read as regression
    and be refused, which is the path this covers.
    """
    from python_scripts.preprocessing import splitting

    seen = {}
    real = splitting.train_test_split

    def capture(*args, **kwargs):
        seen["stratify"] = kwargs.get("stratify")
        return real(*args, **kwargs)

    monkeypatch.setattr(splitting, "train_test_split", capture)

    frame = pd.read_csv(PENGUINS).dropna()
    sid = f"pytest-automl-{uuid.uuid4().hex[:8]}"
    run_model_training(
        sid,
        dict(
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
            modelingMode="automl",
            indicators=[2, 3, 4, 5],
            predictors=[0],
            models="Logistic_classifier",
        ),
        "store",
        lambda _s: {"data": frame.copy()},
    )
    from python_scripts.preprocessing.progress_tracker import get_result

    result = get_result(sid)
    remove_tracker(sid)
    assert result is not None and "error" not in result, result
    assert seen.get("stratify") is not None, "the AutoML run was not stratified"
