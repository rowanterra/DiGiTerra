"""Compare the estimator each trainer builds from an empty form with scikit-learn's defaults.

For every registered model the script builds the estimator through
python_scripts.estimator_params.built_estimator (the pipeline stops before any fit)
and lists each parameter whose value differs from the value of a default instance of
the scikit-learn class that /model-defaults names. A random_state taken from the run
seed is reported as "run seed". Values are compared by repr, so 1 and 1.0 count as
different and are reported with both forms.

Usage (from the repository root):
    MPLBACKEND=Agg .venv/bin/python scripts/compare_trainer_defaults.py [--json out.json]
"""

import argparse
import importlib
import json
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
warnings.filterwarnings("ignore")

from python_scripts.estimator_params import RUN_SEED, built_estimator  # noqa: E402
from python_scripts.model_registry import MODEL_REGISTRY  # noqa: E402
from routes.guided import _SKLEARN_CLASS  # noqa: E402


def compare(name: str) -> dict:
    module, cls = _SKLEARN_CLASS[name]
    library = getattr(importlib.import_module(module), cls)().get_params(deep=False)
    built = built_estimator(name).get_params(deep=False)
    diff = {}
    for key, value in built.items():
        if key not in library or repr(value) == repr(library[key]):
            continue
        shown = "run seed" if key == "random_state" and value == RUN_SEED else repr(value)
        diff[key] = {"scikit_learn": repr(library[key]), "trainer": shown}
    return {"model": name, "estimator": cls, "differs": diff}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json")
    args = ap.parse_args()
    rows = [compare(n) for n, (fn, _) in MODEL_REGISTRY.items() if fn is not None]
    changed = [r for r in rows if r["differs"]]
    beyond_seed = [r for r in changed if set(r["differs"]) - {"random_state"}]
    for r in changed:
        print(
            f"{r['model']:32s} "
            + ", ".join(f"{k}={v['trainer']} (sklearn {v['scikit_learn']})" for k, v in r["differs"].items())
        )
    print(
        f"\n{len(changed)} of {len(rows)} trainers set at least one value other than scikit-learn's; "
        f"{len(beyond_seed)} of {len(rows)} set a value other than random_state"
    )
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "models": len(rows),
                    "differ": len(changed),
                    "differ_beyond_random_state": len(beyond_seed),
                    "rows": rows,
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
