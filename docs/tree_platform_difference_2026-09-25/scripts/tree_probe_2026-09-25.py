"""Per-tree fingerprints of the RF, ExtraTrees and Bagging regressors, for a cross-platform comparison.

Trains each model through run_model_training with the same request as
scripts/check_artifact_roundtrip.py (diabetes example, seed 42, test size 0.20)
and records, for every tree: the tree's integer seed, a SHA-256 of its bootstrap
row indices, and a SHA-256 of its node arrays (feature, threshold, children,
value), and the full node arrays to <out>_<model>_trees.npz. The training matrix the estimator saw is saved to <out>_<model>.npz so the
two platforms' inputs can be compared value by value.

Run from the repository root:
    MPLBACKEND=Agg <python> <path>/tree_probe_2026-09-25.py <out_prefix>
"""

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import check_artifact_roundtrip as rt  # noqa: E402
from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor  # noqa: E402
from sklearn.ensemble._forest import _generate_sample_indices, _get_n_samples_bootstrap  # noqa: E402

SEEN = {}


def _wrap(cls):
    original = cls.fit

    def fit(self, X, y, *a, **k):
        SEEN[cls.__name__] = (np.array(X, dtype=float, copy=True), np.array(y, dtype=float, copy=True))
        return original(self, X, y, *a, **k)

    cls.fit = fit


for c in (RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor):
    _wrap(c)


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def tree_hash(t) -> str:
    t = t.tree_
    return sha(np.concatenate([t.feature.astype(float), t.threshold, t.children_left.astype(float),
                               t.children_right.astype(float), t.value.ravel()]))


def probe(name: str, prefix: str) -> dict:
    captured = {}
    orig_dumps = rt.artifact.dumps

    def dumps(objects):
        captured["model"] = objects["model"]
        return orig_dumps(objects)

    rt.artifact.dumps = dumps
    try:
        rt.check(name, "regression")
    finally:
        rt.artifact.dumps = orig_dumps
    model = captured["model"]
    X, y = SEEN[type(model).__name__]
    np.savez(f"{prefix}_{name}.npz", X=X, y=y)
    n = X.shape[0]
    trees = []
    for i, est in enumerate(model.estimators_):
        row = {"i": i, "tree": tree_hash(est), "nodes": int(est.tree_.node_count)}
        if isinstance(model, BaggingRegressor):
            row["rows"] = sha(np.sort(model.estimators_samples_[i]))
            row["feats"] = sha(model.estimators_features_[i])
        else:
            row["seed"] = int(est.random_state)
            if model.bootstrap:
                nb = _get_n_samples_bootstrap(n, model.max_samples, None)
                row["rows"] = sha(_generate_sample_indices(est.random_state, n, nb, None))
        trees.append(row)
    np.savez(f"{prefix}_{name}_trees.npz", **{f"t{i}_{k}": getattr(e.tree_, k) for i, e in enumerate(model.estimators_)
                                              for k in ("feature", "threshold", "children_left", "children_right",
                                                        "n_node_samples", "impurity", "value")})
    return {"model": name, "estimator": type(model).__name__, "X": sha(X), "y": sha(y),
            "shape": list(X.shape), "params": {k: repr(v) for k, v in model.get_params(deep=False).items()},
            "trees": trees}


if __name__ == "__main__":
    prefix = sys.argv[1]
    out = [probe(m, prefix) for m in ("RF", "ExtraTrees", "Bagging")]
    Path(f"{prefix}.json").write_text(json.dumps(out, indent=1))
    for r in out:
        print(r["model"], r["X"], r["y"], len(r["trees"]))
