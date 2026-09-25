"""Round-trip every registered estimator through the artifact format.

For each entry in MODEL_REGISTRY the script trains on the matching example file
through the same function the /process route calls, writes the fitted objects with
python_scripts.reporting.artifact, reads them back, and compares predictions with
the objects that were never written. A model whose predictions differ, or whose
artifact names a type outside the allowlist, is reported.

Usage:
    MPLBACKEND=Agg .venv/bin/python scripts/check_artifact_roundtrip.py [--only NAME ...] [--json out.json]
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import traceback
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_sup = Path(tempfile.mkdtemp(prefix="dgt_artifact_"))
os.environ.setdefault("DIGITERRA_APP_SUPPORT_DIR", str(_sup))
os.environ.setdefault("DIGITERRA_OUTPUT_DIR", str(_sup / "vis"))
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from python_scripts import config  # noqa: E402
from python_scripts.app_model_training import run_model_training  # noqa: E402
from python_scripts.model_registry import _PASSTHROUGH_CLASSIFIERS, MODEL_REGISTRY  # noqa: E402
from python_scripts.preprocessing.progress_tracker import get_result, remove_tracker  # noqa: E402
from python_scripts.reporting import artifact  # noqa: E402
from python_scripts.reporting.bundle import fitted_objects, platform_record  # noqa: E402

EXAMPLES = ROOT / "examples"
DATASETS = {
    "regression": ("diabetes.csv", None),
    "classification": ("iris.csv", None),
    "cluster": ("iris_clustering.csv", None),
}
# CategoricalNB takes integer-coded categories, not the continuous measurements of
# the iris file, so it is trained on a discretized copy rather than reported as a
# failure of the format.
DISCRETE_CLASSIFIERS = ("CategoricalNB_classifier",)
# The defaults an empty form would leave as None, matching train_every_model.py.
FILLS = {
    "n_estimators": 20,
    "n_clusters": 3,
    "n_components": 3,
    "alpha": 1.0,
    "C": 1.0,
    "max_iter": 200,
    "n_neighbors": 5,
    "max_depth": 5,
    "learning_rate": 0.1,
    "eps": 0.5,
    "min_samples": 5,
    "min_cluster_size": 5,
    "radius": 1.0,
    "threshold": 0.5,
    "l1_ratio": 0.5,
    "epsilon": 1.35,
    "quantile": 0.5,
    "nu": 0.5,
    "activation": "relu",
    "hidden_layer_sizes": (100,),
    "solver": "adam",
}

BASE = {
    # The Excel export reads the file name from the request. A missing one is a
    # warning in the application now rather than a lost run, but the harness
    # should send what a page sends.
    "filename": "example.csv",
    "scaler": "standard",
    "hyperparameters": {},
    "nonreq": False,
    "units": "",
    "sigfig": 3,
    "seedValue": 42,
    "testSize": "0.20",
    "stratifyBool": False,
    "dropMissing": "none",
    "imputeStrategy": "none",
    "dropZero": "none",
    "zeroHandle": "none",
    "quantileBinDict": {"quantile": 0, "bin": 0, "binsLabel": ""},
    "useTransformer": "No",
    "transformerCols": [],
    "crossValidationType": "None",
    "crossValidationFolds": 5,
}


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_record(values) -> list:
    """Predictions as JSON at full precision, so a reader can compare them by hand."""
    out = []
    for v in np.ravel(np.asarray(values, dtype=object))[:8]:
        if isinstance(v, (np.floating, float)):
            out.append(float(v))
        elif isinstance(v, (np.integer, int)):
            out.append(int(v))
        else:
            out.append(str(v))
    return out


def _outputs(objects, X):
    """What the model produces: predictions, or the labels of a fit that has none.

    Several clustering models have no predict, which the application already knows
    (app_prediction refuses inference for them). Comparing their labels is the
    equivalent check that the fit came back the same.
    """
    model = objects["model"]
    if not hasattr(model, "predict"):
        labels = getattr(model, "labels_", None)
        if labels is None:
            raise AttributeError(f"{type(model).__name__} has neither predict nor labels_")
        return np.asarray(labels, dtype=object)
    return _predict(objects, X)


def _predict(objects, X):
    model = objects["model"]
    scaler = objects.get("X_scaler")
    Xs = scaler.transform(X) if scaler is not None else X
    order = getattr(model, "_digiterra_model_features", None)
    if order is not None and hasattr(Xs, "columns"):
        Xs = Xs[order]
    return np.asarray(model.predict(Xs), dtype=object)


def check(name: str, task: str) -> dict:
    csv, _ = DATASETS[task]
    df = pd.read_csv(EXAMPLES / csv)
    if name in DISCRETE_CLASSIFIERS:
        target = df.columns[-1]
        for col in df.columns[:-1]:
            df[col] = pd.qcut(df[col], 4, labels=False, duplicates="drop")
        df[target] = df[target]
    n = len(df.columns)
    indicators = list(range(n - 1)) if task != "cluster" else list(range(n))
    predictors = [] if task == "cluster" else [n - 1]
    payload = dict(BASE, indicators=indicators, predictors=predictors, models=name)
    # FILLS is a blanket dict that relied on get_model_kwargs dropping unknown
    # keys. The classifiers that now pass form values through would receive
    # solver="adam" and learning_rate=0.1, which their estimators reject, and a
    # real form never sends another model's fields; they get an empty form.
    payload["hyperparameters"] = {} if name in _PASSTHROUGH_CLASSIFIERS else dict(FILLS)
    store = {"data": df}
    sid = f"artifact-{name}"
    token = config.set_session_scope(sid)
    try:
        run_model_training(sid, payload, sid, lambda _s: store)
        result = get_result(sid)
    finally:
        config.reset_session_scope(token)
    remove_tracker(sid)
    if result is None or "error" in result:
        return {"model": name, "task": task, "status": "train_failed", "detail": (result or {}).get("error")}
    row = {
        "model": name,
        "task": task,
        "estimator": type(store.get("model")).__name__,
        "dataset": csv,
        "dataset_sha256": _sha256(EXAMPLES / csv),
        "rows": int(len(df)),
        "seed": BASE["seedValue"],
        "test_size": BASE["testSize"],
    }
    objects = fitted_objects(store)
    try:
        data = artifact.dumps(objects)
    except artifact.ArtifactError as e:
        status = "unsupported" if row["estimator"] in artifact.UNSUPPORTED_ESTIMATORS else "write_failed"
        return {**row, "status": status, "detail": str(e)}
    refused = artifact.untrusted_types(data)
    if refused:
        return {**row, "status": "untrusted_types", "detail": refused}
    back = artifact.loads(data)
    X = df.iloc[:8, indicators]
    try:
        before, after = _outputs(objects, X), _outputs(back, X)
    except Exception as e:
        return {**row, "status": "predict_failed", "detail": f"{type(e).__name__}: {e}"}
    return {
        **row,
        "status": "ok" if np.array_equal(before, after) else "predictions_differ",
        "bytes": len(data),
        "artifact_sha256": hashlib.sha256(data).hexdigest(),
        "compared": "predict" if hasattr(objects["model"], "predict") else "labels_",
        "output_before": _as_record(before),
        "output_after": _as_record(after),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--json")
    args = ap.parse_args()

    rows = []
    for name, (trainer, task) in sorted(MODEL_REGISTRY.items()):
        if trainer is None or task not in DATASETS:
            continue
        if args.only and name not in args.only:
            continue
        try:
            rows.append(check(name, task))
        except Exception as e:
            rows.append(
                {
                    "model": name,
                    "task": task,
                    "status": "error",
                    "detail": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc()[-600:],
                }
            )
        print(f"{rows[-1]['status']:18s} {name}", flush=True)

    ok = [r for r in rows if r["status"] == "ok"]
    # A model that will not train on the example data says nothing about the
    # format, so it is reported separately and does not fail the run. Anything
    # else did reach the format and did not survive it.
    untrainable = [r for r in rows if r["status"] == "train_failed"]
    unsupported = [r for r in rows if r["status"] == "unsupported"]
    bad = [r for r in rows if r["status"] not in ("ok", "train_failed", "unsupported")]
    print(f"\n{len(ok)} of {len(rows) - len(untrainable)} trained models round-tripped with identical predictions")
    for r in untrainable:
        print(f"  not trained on the example data: {r['model']}: {str(r.get('detail'))[:160]}")
    for r in unsupported:
        print(f"  cannot be written without a pickle, and says so: {r['model']}")
    for r in bad:
        print(f"  FORMAT: {r['model']}: {r['status']} {str(r.get('detail'))[:200]}")
    if args.json:
        import datetime as _dt
        from importlib import metadata

        def _version(pkg):
            try:
                return metadata.version(pkg)
            except metadata.PackageNotFoundError:
                return None

        record = {
            "written_at": _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat(),
            "how_to_read_this": (
                "output_before is what the model in memory predicted for the first eight rows of the dataset; "
                "output_after is what the same model predicted after being written to the artifact format and "
                "read back. They are compared exactly, not within a tolerance. Rerun one row with "
                "'MPLBACKEND=Agg .venv/bin/python scripts/check_artifact_roundtrip.py --only <model>'."
            ),
            "environment": {
                **{p: _version(p) for p in ("skops", "scikit-learn", "numpy", "pandas", "scipy", "joblib")},
                "python": platform.python_version(),
                **platform_record(),
            },
            "summary": {
                "trained": len(rows) - len(untrainable),
                "round_tripped": len(ok),
                "unsupported": [r["model"] for r in unsupported],
                "not_trained": [r["model"] for r in untrainable],
            },
            "rows": rows,
        }
        Path(args.json).write_text(json.dumps(record, indent=2, default=str))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
