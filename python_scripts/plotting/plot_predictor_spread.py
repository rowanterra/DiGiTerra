"""The spread of each predictor in the training rows against the rows being predicted.

A model returns a prediction for any row whose columns parse. This figure is the
check a reader makes before trusting one: does the new data lie inside the range
the model was fitted on? One row per numeric predictor, in the file's units: the
training rows as a filled box, the new rows as a hatched box, and each new row
that the novelty detector marked as unlike the training rows as a point. The
layout follows the data-spread panel of the computational prospecting paper
(Terra et al., 2026, Figure 1c): boxes and whiskers only, solid against hatched,
readable in grayscale.

write_training_spread is called by the pipelines with the raw training predictors
and writes training_predictor_spread.json next to the other run records;
plot_predictor_spread reads that file at prediction time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SPREAD_FILE = "training_predictor_spread.json"
MAX_ROWS_DRAWN = 24


def write_training_spread(X_train_raw: pd.DataFrame, out_dir: Path) -> None:
    """Record the five-number summary of every numeric training predictor."""
    try:
        rows = []
        for col in X_train_raw.columns:
            s = pd.to_numeric(X_train_raw[col], errors="coerce").dropna()
            if s.empty or s.nunique() < 2:
                continue
            q = s.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).tolist()
            rows.append({"column": str(col), "n": int(s.count()), "quantiles": [float(v) for v in q]})
        with open(Path(out_dir) / SPREAD_FILE, "w") as f:
            json.dump({"predictors": rows}, f, indent=2)
    except Exception as e:  # a run must not fail because this record could not be written
        logger.debug("Training predictor spread not written: %s", e)


def _box_stats(label: str, q: list[float]) -> dict:
    return {"label": label, "whislo": q[0], "q1": q[1], "med": q[2], "q3": q[3], "whishi": q[4], "fliers": []}


def plot_predictor_spread(
    spread_path: Path,
    new_rows: pd.DataFrame,
    novel_mask: pd.Series | None,
    out_path: Path,
) -> Path | None:
    """Draw the training boxes against the new rows. Returns the path, or None."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        record = json.loads(Path(spread_path).read_text())
    except Exception as e:
        logger.debug("Training predictor spread not read: %s", e)
        return None
    predictors = [p for p in record.get("predictors", []) if p["column"] in new_rows.columns][:MAX_ROWS_DRAWN]
    if not predictors:
        return None

    n = len(predictors)
    fig, axes = plt.subplots(n, 1, figsize=(8.0, max(2.6, 1.05 * n + 0.8)), squeeze=False)
    axes = axes[:, 0]
    for ax, p in zip(axes, predictors, strict=True):
        col = p["column"]
        new = pd.to_numeric(new_rows[col], errors="coerce").dropna()
        train_stats = _box_stats("training", p["quantiles"])
        stats = [train_stats]
        if len(new) >= 2:
            q = new.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).tolist()
            stats.append(_box_stats("new rows", q))
        positions = [1.0, 0.0][: len(stats)]
        boxes = ax.bxp(
            stats,
            positions=positions,
            orientation="horizontal",
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
            whiskerprops={"color": "black", "linewidth": 0.9},
            capprops={"color": "black", "linewidth": 0.9},
            boxprops={"linewidth": 0.9, "edgecolor": "black"},
        )
        boxes["boxes"][0].set_facecolor("#9aa7a3")
        if len(boxes["boxes"]) > 1:
            boxes["boxes"][1].set_facecolor("white")
            boxes["boxes"][1].set_hatch("////")
        # Every new row is drawn as a point; a row the detector marked as unlike the
        # training rows is drawn larger and filled, so it can be found by eye.
        if len(new):
            mask = None
            if novel_mask is not None:
                mask = pd.Series(novel_mask).reindex(new.index).fillna(False).astype(bool)
            y_jitter = np.random.default_rng(0).uniform(-0.18, 0.18, size=len(new))
            ordinary = new if mask is None else new[~mask]
            ax.scatter(
                ordinary.values,
                y_jitter[: len(ordinary)] if mask is None else y_jitter[(~mask).values],
                s=10,
                facecolors="none",
                edgecolors="black",
                linewidths=0.6,
                zorder=3,
            )
            if mask is not None and mask.any():
                ax.scatter(
                    new[mask].values,
                    y_jitter[mask.values],
                    s=26,
                    marker="D",
                    facecolors="black",
                    edgecolors="black",
                    zorder=4,
                    label="row unlike the training rows",
                )
        ax.set_yticks(positions)
        ax.set_yticklabels(["training", "new rows"][: len(stats)], fontsize=8)
        ax.set_title(col, fontsize=9, loc="left", pad=2)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        for ax in axes[1:]:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    if handles:
        fig.legend(handles[:1], labels[:1], loc="lower right", fontsize=8, frameon=False)
    fig.suptitle(
        "Predictor spread: training rows (filled) and rows being predicted (hatched)", fontsize=9, x=0.02, ha="left"
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    return out_path
