"""Tree ensembles under five depth settings, and fitted models carried between platforms.

Part 1 fits RF (20 trees), ExtraTrees (100) and Bagging (10) on the training
matrix the pipeline built (the probe's <prefix>_RF.npz; bitwise equal on Linux and
the Mac), with seed 42, under five settings: scikit-learn defaults, min_samples_leaf
2 and 5, max_depth 10 and 5. For each fit it records a hash of every tree and the
predictions on 2,000 query rows. The query rows take each column independently
from that column's training values with numpy's Generator(PCG64) seed 20260925,
which gives the same rows on every platform.

Part 2 writes the three default models with DiGiTerra's artifact format
(--write DIR) or reads artifacts written elsewhere and predicts the query rows
(--read DIR). Reading shows whether a fitted model predicts the same values on
another platform without retraining.

Run from the repository root:
    python <path>/platform_experiment_2026-09-25.py DATA.npz OUT_PREFIX [--write DIR | --read DIR]
Writes OUT_PREFIX.json and OUT_PREFIX_preds.npz.
"""

import argparse
import hashlib
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd()))

MODELS = {
    "RF": (RandomForestRegressor, {"n_estimators": 20}),
    "ExtraTrees": (ExtraTreesRegressor, {"n_estimators": 100}),
    "Bagging": (BaggingRegressor, {"n_estimators": 10}),
}
SETTINGS = {
    "default": {},
    "min_samples_leaf=2": {"min_samples_leaf": 2},
    "min_samples_leaf=5": {"min_samples_leaf": 5},
    "max_depth=10": {"max_depth": 10},
    "max_depth=5": {"max_depth": 5},
}


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def tree_hash(est) -> str:
    t = est.tree_
    return sha(np.concatenate([t.feature.astype(float), t.threshold, t.children_left.astype(float),
                               t.children_right.astype(float), t.value.ravel()]))


def build(model: str, setting: dict):
    cls, base = MODELS[model]
    if model == "Bagging":
        from sklearn.tree import DecisionTreeRegressor

        return cls(estimator=DecisionTreeRegressor(**setting), random_state=42, **base)
    return cls(random_state=42, **base, **setting)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--write")
    ap.add_argument("--read")
    args = ap.parse_args()

    d = np.load(args.data)
    X, y = d["X"], d["y"].ravel()
    rng = np.random.Generator(np.random.PCG64(20260925))
    Q = np.column_stack([X[rng.integers(0, len(X), 2000), f] for f in range(X.shape[1])])
    out = {"platform": platform.platform(), "machine": platform.machine(), "sklearn": sklearn.__version__,
           "sklearn_file": sklearn.__file__, "X": sha(X), "Q": sha(Q), "fits": {}}
    preds = {}
    fitted = {}
    for model in MODELS:
        for name, setting in SETTINGS.items():
            est = build(model, setting).fit(X, y)
            key = f"{model} {name}"
            p = est.predict(Q)
            preds[key] = p
            out["fits"][key] = {"trees": [tree_hash(e) for e in est.estimators_], "pred_Q": sha(p)}
            if name == "default":
                fitted[model] = est
    if args.write or args.read:
        from python_scripts.reporting import artifact

        folder = Path(args.write or args.read)
        folder.mkdir(parents=True, exist_ok=True)
        out["artifacts"] = {}
        for model, est in fitted.items():
            path = folder / f"{model}_default.skops"
            if args.write:
                path.write_bytes(artifact.dumps({"model": est}))
            loaded = artifact.loads(path.read_bytes())["model"]
            p = loaded.predict(Q)
            preds[f"artifact {model}"] = p
            out["artifacts"][model] = {"file_sha": sha(np.frombuffer(path.read_bytes(), np.uint8)), "pred_Q": sha(p)}
    Path(args.out + ".json").write_text(json.dumps(out, indent=1))
    np.savez(args.out + "_preds.npz", **preds)
    print(out["machine"], out["sklearn"], out["X"], out["Q"], len(out["fits"]), "fits")


if __name__ == "__main__":
    main()
