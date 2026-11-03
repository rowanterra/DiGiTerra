"""The result page offers "ROC Curve (micro)" for a classification run.

RocCurveDisplay.from_estimator takes a binary classifier only. On a three-class
target it raised, the warning was swallowed, and roc_curve.png was never written, so
the page requested a file that did not exist and drew its alt text. Only the
per-class file was produced.
"""

import uuid
import warnings
from pathlib import Path

import pandas as pd
import pytest

warnings.filterwarnings("ignore")


@pytest.fixture
def vis_dir():
    """The directory the plots really go to.

    config.VIS_DIR binds at import, so setting DIGITERRA_OUTPUT_DIR from a test does
    not move it. Reading the object and clearing the two files first is what makes
    this test observe the run rather than whatever an earlier run left behind.
    """
    from python_scripts.config import VIS_DIR

    out = Path(str(VIS_DIR))
    out.mkdir(parents=True, exist_ok=True)
    for name in ("roc_curve.png", "roc_curve_per_class.png"):
        (out / name).unlink(missing_ok=True)
    return out


def _run(frame, predictors, indicators, model="Logistic_classifier"):
    from python_scripts.app_model_training import run_model_training
    from python_scripts.preprocessing.progress_tracker import get_result, remove_tracker

    sid = f"pytest-roc-{uuid.uuid4().hex[:8]}"
    payload = dict(
        hyperparameters={},
        nonreq=False,
        scaler="standard",
        units="",
        sigfig=3,
        seedValue=42,
        testSize="0.20",
        stratifyBool=False,
        dropMissing="none",
        imputeStrategy="none",
        dropZero="none",
        quantileBinDict={"quantile": 0, "bin": 0, "binsLabel": ""},
        useTransformer="No",
        transformerCols=[],
        filename="t.csv",
        crossValidationType="None",
        modelingMode="simple",
        indicators=indicators,
        predictors=predictors,
        models=model,
        outputType="Classifier",
    )
    run_model_training(sid, payload, "store", lambda _s: {"data": frame.copy()})
    result = get_result(sid)
    remove_tracker(sid)
    assert result is not None and "error" not in result, result
    return result


def _written(directory):
    return {p.name for p in Path(directory).rglob("*.png")}


def test_a_three_class_run_writes_the_micro_average_roc(vis_dir):
    """The file the page asks for by default must exist for more than two classes."""
    frame = pd.read_csv("examples/penguins.csv").dropna()
    assert frame["species"].nunique() == 3
    _run(frame, predictors=[0], indicators=[2, 3, 4, 5])
    names = _written(vis_dir)
    assert "roc_curve.png" in names, f"the micro-average ROC was not written: {sorted(names)}"
    assert "roc_curve_per_class.png" in names, f"the per-class ROC was not written: {sorted(names)}"


def test_a_two_class_run_still_writes_the_roc(vis_dir):
    """The binary path used the Display class and must keep working."""
    frame = pd.read_csv("examples/penguins.csv").dropna()
    frame = frame[frame["species"].isin(["Adelie", "Gentoo"])]
    assert frame["species"].nunique() == 2
    _run(frame, predictors=[0], indicators=[2, 3, 4, 5])
    names = _written(vis_dir)
    assert "roc_curve.png" in names, f"the binary ROC was not written: {sorted(names)}"
