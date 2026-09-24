"""Train every registered model once, through the same path the app uses.

For each entry in MODEL_REGISTRY the script calls run_model_training with a
dict-backed session store on the matching example CSV, using the default
hyperparameters that get_model_kwargs produces from an empty form. Required
parameters the form would leave as None are filled from DEFAULT_FILLS and
reported, since the UI would send None for them.

Usage:
    MPLBACKEND=Agg .venv/bin/python scripts/train_every_model.py [--only NAME ...] [--json out.json]
"""

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_sup = Path(tempfile.mkdtemp(prefix="dgt_sweep_"))
os.environ.setdefault("DIGITERRA_APP_SUPPORT_DIR", str(_sup))
os.environ.setdefault("DIGITERRA_OUTPUT_DIR", str(_sup / "vis"))
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from python_scripts.app_model_training import run_model_training  # noqa: E402
from python_scripts.model_registry import MODEL_REGISTRY, _model_kwargs  # noqa: E402
from python_scripts.preprocessing.progress_tracker import get_result, remove_tracker  # noqa: E402

EXAMPLES = ROOT / "examples"

# Defaults for parameters the form leaves empty. Keyed by the kwarg name that
# get_model_kwargs emits; the value is what a user would reasonably pick.
DEFAULT_FILLS = {
    "n_estimators": 50,
    "n_clusters": 3,
    "n_components": 3,
    "alpha": 1.0,
    "C": 1.0,
    "l1_ratio": 0.5,
    "max_iter": 200,
    "n_neighbors": 5,
    "max_depth": 5,
    "learning_rate": 0.1,
    "eps": 0.5,
    "min_samples": 5,
    "min_cluster_size": 5,
    "epsilon": 1.35,
    "quantile": 0.5,
    "nu": 0.5,
    "radius": 1.0,
    "threshold": 0.5,
    "branching_factor": 50,
    "damping": 0.5,
    "bandwidth": None,
    "n_nonzero_coefs": None,
    "degree": 2,
    "hidden_layer_sizes": "(64,32)",
    "batch_size": 100,
    "n_init": 10,
    "activation": "relu",
    "solver": "adam",
    "eta0": 1.0,
    "tol": 1e-4,
}

BASE = dict(
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
    filename="sweep.csv",
    crossValidationType="None",
    modelingMode="simple",
)


def dataset_for(problem_type, name):
    if problem_type == "cluster":
        return EXAMPLES / "iris_clustering.csv"
    if problem_type == "classification":
        # Naive Bayes variants that require non-negative counts get wine, which is all positive.
        if name in {"MultinomialNB_classifier", "ComplementNB_classifier", "CategoricalNB_classifier"}:
            return EXAMPLES / "wine.csv"
        return EXAMPLES / "iris.csv"
    return EXAMPLES / "diabetes.csv"


def hyperparameters_for(name):
    """Return (hyperparameters dict, list of filled keys) so get_model_kwargs sees no None."""
    empty = _model_kwargs(name, {}, False, 42)  # keeps the None keys an empty form leaves
    filled = {}
    for key, value in empty.items():
        if value is None and key in DEFAULT_FILLS and DEFAULT_FILLS[key] is not None:
            filled[key] = DEFAULT_FILLS[key]
    return filled, sorted(filled)


def run_one(name, problem_type, wait=300):
    csv = dataset_for(problem_type, name)
    df = pd.read_csv(csv)
    ncols = len(df.columns)
    if problem_type == "cluster":
        indicators, predictors = list(range(ncols)), []
    else:
        indicators, predictors = list(range(ncols - 1)), [ncols - 1]
    hyper, filled = hyperparameters_for(name)
    store = {"data": df}
    payload = dict(BASE, indicators=indicators, predictors=predictors, models=name, hyperparameters=hyper)
    session_id = f"sweep-{name}"
    t0 = time.time()
    try:
        run_model_training(session_id, payload, "store", lambda _sid: store)
        res = get_result(session_id)
    except Exception as exc:  # run_model_training catches internally; this is belt and braces
        res = {"error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-800:]}"}
    finally:
        try:
            remove_tracker(session_id)
        except Exception:
            pass
    secs = time.time() - t0
    status = "fail" if (res is None or "error" in res) else "pass"
    return {
        "model": name,
        "task": problem_type,
        "status": status,
        "seconds": round(secs, 2),
        "dataset": csv.name,
        "filled_defaults": filled,
        "error": (res or {}).get("error") if status == "fail" else None,
    }


def _run_parallel(args, names):
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    def child(name):
        if MODEL_REGISTRY[name][0] is None:
            return {
                "model": name,
                "task": MODEL_REGISTRY[name][1],
                "status": "skip",
                "seconds": 0.0,
                "dataset": None,
                "filled_defaults": [],
                "error": "no train function registered",
            }
        env = dict(os.environ)
        sub = Path(tempfile.mkdtemp(prefix=f"dgt_sweep_{name}_"))
        env["DIGITERRA_APP_SUPPORT_DIR"] = str(sub)
        env["DIGITERRA_OUTPUT_DIR"] = str(sub / "vis")
        proc = subprocess.run(
            [sys.executable, __file__, "--_child", name], capture_output=True, text=True, env=env, cwd=ROOT, timeout=900
        )
        last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        try:
            return json.loads(last)
        except Exception:
            return {
                "model": name,
                "task": MODEL_REGISTRY[name][1],
                "status": "fail",
                "seconds": 0.0,
                "dataset": None,
                "filled_defaults": [],
                "error": (proc.stderr or proc.stdout)[-600:],
            }

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for row in ex.map(child, names):
            rows.append(row)
            extra = f"  filled={row['filled_defaults']}" if row["filled_defaults"] else ""
            err = f"  {str(row['error'])[:160]}" if row["error"] else ""
            print(
                f"{row['model']:32s} {row['task']:14s} {row['status']:5s} {row['seconds']:7.2f}s{extra}{err}",
                flush=True,
            )
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, default=str))
    n_fail = sum(r["status"] == "fail" for r in rows)
    print(f"\n{len(rows)} models, {n_fail} failed")
    return 1 if n_fail else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--workers", type=int, default=1, help="run models in parallel subprocesses")
    ap.add_argument("--_child", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args._child:
        name = args._child
        print(json.dumps(run_one(name, MODEL_REGISTRY[name][1]), default=str))
        return 0
    rows = []
    if args.workers > 1:
        return _run_parallel(args, [n for n in MODEL_REGISTRY if not args.only or n in args.only])
    for name, (train_fn, problem_type) in MODEL_REGISTRY.items():
        if args.only and name not in args.only:
            continue
        if train_fn is None:
            rows.append(
                {
                    "model": name,
                    "task": problem_type,
                    "status": "skip",
                    "seconds": 0.0,
                    "dataset": None,
                    "filled_defaults": [],
                    "error": "no train function registered",
                }
            )
            print(f"{name:32s} {problem_type:14s} skip   (no train function)", flush=True)
            continue
        row = run_one(name, problem_type)
        rows.append(row)
        extra = f"  filled={row['filled_defaults']}" if row["filled_defaults"] else ""
        err = f"  {row['error'][:160]!s}" if row["error"] else ""
        print(f"{name:32s} {problem_type:14s} {row['status']:5s} {row['seconds']:7.2f}s{extra}{err}", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, default=str))
    n_fail = sum(r["status"] == "fail" for r in rows)
    print(f"\n{len(rows)} models, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
