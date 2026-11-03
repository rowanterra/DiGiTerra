"""Compare candidate models by cross-validation before one is chosen.

The AutoML page offered "Let AutoML choose", and the page then quietly sent a
random forest. This module is what makes the choice: every candidate is scored
by the same folds, repeated over several seeds, and the candidate with the best
mean score is trained. The fold scores are drawn as one box per candidate, in
the style of the seed sweep of the computational prospecting paper (Terra et
al., 2026, Supplementary Table), so a winner separated from the field by less
than the fold-to-fold spread is shown as such rather than announced alone.

The candidates are trained with an empty form, so each takes its scikit-learn
defaults through the same trainer the run uses; the winner is then trained by
the ordinary route, with whatever the form carries.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# The shortlist scored when no model is chosen. Every entry is in MODEL_REGISTRY
# and trains from an empty form.
CANDIDATES = {
    "regression": [
        "Linear",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "K-Nearest",
        "SVM",
        "RF",
        "ExtraTrees",
        "gradient_boosting",
    ],
    "classification": [
        "Logistic_classifier",
        "KNeighbors_classifier",
        "SVC_classifier",
        "RF_classifier",
        "ExtraTrees_classifier",
        "GradientBoosting_classifier",
        "GaussianNB_classifier",
    ],
}
SCORING = {"regression": "r2", "classification": "accuracy"}
DEFAULT_SEEDS = 3
DEFAULT_FOLDS = 5
LARGE_FRAME_ROWS = 5000  # above this, one seed keeps the comparison to minutes
COMPARISON_PNG = "model_comparison.png"
COMPARISON_CSV = "model_comparison.csv"


# Registry name to the scikit-learn class the trainer of that name builds, with
# the defaults an empty form gives it. Only the candidates above are listed.
_CLASSES = {
    "Linear": ("sklearn.linear_model", "LinearRegression", {}),
    "Ridge": ("sklearn.linear_model", "Ridge", {}),
    "Lasso": ("sklearn.linear_model", "Lasso", {}),
    "ElasticNet": ("sklearn.linear_model", "ElasticNet", {}),
    "K-Nearest": ("sklearn.neighbors", "KNeighborsRegressor", {}),
    "SVM": ("sklearn.svm", "SVR", {}),
    "RF": ("sklearn.ensemble", "RandomForestRegressor", {"n_estimators": 100, "random_state": "seed"}),
    "ExtraTrees": ("sklearn.ensemble", "ExtraTreesRegressor", {"n_estimators": 100, "random_state": "seed"}),
    "gradient_boosting": ("sklearn.ensemble", "GradientBoostingRegressor", {"random_state": "seed"}),
    "Logistic_classifier": ("sklearn.linear_model", "LogisticRegression", {"max_iter": 1000, "random_state": "seed"}),
    "KNeighbors_classifier": ("sklearn.neighbors", "KNeighborsClassifier", {}),
    "SVC_classifier": ("sklearn.svm", "SVC", {"random_state": "seed"}),
    "RF_classifier": ("sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 100, "random_state": "seed"}),
    "ExtraTrees_classifier": (
        "sklearn.ensemble",
        "ExtraTreesClassifier",
        {"n_estimators": 100, "random_state": "seed"},
    ),
    "GradientBoosting_classifier": ("sklearn.ensemble", "GradientBoostingClassifier", {"random_state": "seed"}),
    "GaussianNB_classifier": ("sklearn.naive_bayes", "GaussianNB", {}),
}


def _estimator_for(name: str, seed: int, task: str):
    """A fresh estimator for a registry name, with the defaults an empty form gives it."""
    module, cls, kwargs = _CLASSES[name]
    est_cls = getattr(__import__(module, fromlist=[cls]), cls)
    return est_cls(**{k: (seed if v == "seed" else v) for k, v in kwargs.items()})


def compare_candidates(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    scaler,
    candidates: list[str] | None = None,
    seeds: int = DEFAULT_SEEDS,
    folds: int = DEFAULT_FOLDS,
    seed: int = 42,
    nan_strategy: str | None = None,
    out_dir: Path | None = None,
    progress=None,
) -> dict:
    """Score every candidate on the same folds; return the pick and the record.

    X is the raw predictor frame after one-hot encoding is not needed (numeric
    columns only; text columns are dropped here, as the comparison is a ranking,
    not the run). Returns {"chosen": name, "scores": {name: [fold scores]},
    "summary": [{model, mean, std, n}], "png": path, "csv": path, "note": str}.
    """
    names = list(candidates or CANDIDATES.get(task, []))
    numeric = X.select_dtypes(include=np.number)
    if numeric.shape[1] == 0 or len(names) == 0:
        return {"chosen": None, "scores": {}, "summary": [], "note": "No numeric predictors to compare on."}
    if len(numeric) > LARGE_FRAME_ROWS:
        seeds = 1
    y_values = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y
    keep = y_values.notna()
    numeric, y_values = numeric[keep], y_values[keep]
    if not nan_strategy:
        keep = numeric.notna().all(axis=1)
        numeric, y_values = numeric[keep], y_values[keep]

    scores: dict[str, list[float]] = {}
    done = 0
    total = len(names) * seeds
    for name in names:
        try:
            est = _estimator_for(name, seed, task)
        except Exception as e:
            logger.info("Candidate %s skipped: %s", name, e)
            continue
        steps = []
        if nan_strategy:
            steps.append(
                ("impute", SimpleImputer(strategy="most_frequent" if nan_strategy == "mode" else nan_strategy))
            )
        if scaler is not None:
            steps.append(("scaler", clone(scaler)))
        steps.append(("model", est))
        pipe = Pipeline(steps)
        per_fold: list[float] = []
        for s in range(seeds):
            rs = int(seed) + s
            cv = (
                StratifiedKFold(n_splits=folds, shuffle=True, random_state=rs)
                if task == "classification"
                else KFold(n_splits=folds, shuffle=True, random_state=rs)
            )
            try:
                res = cross_val_score(pipe, numeric, y_values, cv=cv, scoring=SCORING[task], error_score=np.nan)
                per_fold.extend(float(v) for v in res)
            except Exception as e:
                logger.info("Candidate %s failed on seed %s: %s", name, rs, e)
            done += 1
            if progress is not None:
                progress(int(100 * done / total), f"Comparing {name} ({done} of {total} rounds)")
        finite = [v for v in per_fold if np.isfinite(v)]
        if finite:
            scores[name] = finite

    if not scores:
        return {"chosen": None, "scores": {}, "summary": [], "note": "No candidate could be scored."}
    summary = sorted(
        (
            {"model": n, "mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}
            for n, v in scores.items()
        ),
        key=lambda r: r["mean"],
        reverse=True,
    )
    chosen = summary[0]["model"]
    runner_up = summary[1] if len(summary) > 1 else None
    margin = (summary[0]["mean"] - runner_up["mean"]) if runner_up else None
    spread = summary[0]["std"]
    if runner_up is None:
        note = f"{chosen} was the only candidate scored."
    elif margin is not None and margin < spread:
        note = (
            f"{chosen} scored best (mean {SCORING[task]} {summary[0]['mean']:.3f}), but its lead over "
            f"{runner_up['model']} ({runner_up['mean']:.3f}) is smaller than its fold-to-fold spread ({spread:.3f}); "
            "the two are not separable on this data."
        )
    else:
        note = (
            f"{chosen} scored best (mean {SCORING[task]} {summary[0]['mean']:.3f}); the runner-up "
            f"{runner_up['model']} scored {runner_up['mean']:.3f}."
        )
    record = {"chosen": chosen, "scores": scores, "summary": summary, "note": note, "seeds": seeds, "folds": folds}
    if out_dir is not None:
        try:
            record["png"] = str(plot_comparison(scores, summary, task, seeds, folds, Path(out_dir) / COMPARISON_PNG))
            rows = [{"model": n, "fold": i + 1, SCORING[task]: v} for n, vs in scores.items() for i, v in enumerate(vs)]
            pd.DataFrame(rows).to_csv(Path(out_dir) / COMPARISON_CSV, index=False)
            record["csv"] = str(Path(out_dir) / COMPARISON_CSV)
        except Exception as e:
            logger.warning("Model comparison figure not written: %s", e)
    return record


def plot_comparison(scores: dict, summary: list[dict], task: str, seeds: int, folds: int, out_path: Path) -> Path:
    """One box per candidate of its fold scores, best at the top, journal style."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [r["model"] for r in summary][::-1]  # best at the top of a horizontal plot
    data = [scores[n] for n in order]
    fig, ax = plt.subplots(figsize=(6.5, max(2.4, 0.42 * len(order) + 1.2)))
    box = ax.boxplot(
        data,
        orientation="horizontal",
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "black", "linewidth": 0.9},
        capprops={"color": "black", "linewidth": 0.9},
        boxprops={"linewidth": 0.9, "edgecolor": "black", "facecolor": "white"},
    )
    box["boxes"][-1].set_facecolor("#9aa7a3")  # the chosen model, filled
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        ax.scatter(
            vals,
            i + rng.uniform(-0.15, 0.15, len(vals)),
            s=9,
            facecolors="none",
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels([n.replace("_classifier", "").replace("_", " ") for n in order], fontsize=8)
    ax.set_xlabel(
        f"Cross-validated {SCORING[task]} ({folds} folds, {seeds} seed{'s' if seeds != 1 else ''})", fontsize=9
    )
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Candidate models on the same folds; the filled box is the model chosen", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    return out_path
