"""Statistical fills are fitted on the training split, not the whole file."""

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from python_scripts.helpers import preprocess_data
from python_scripts.preprocessing.split_fill import FillPlan
from tests.unit.test_pipelines import _train

warnings.filterwarnings("ignore")


def _frame_with_a_gap(seed=42, n=200):
    """One feature with NaN in every fifth row; the feature's mean depends on which rows are seen."""
    rng = np.random.RandomState(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = x1 * 3 + rng.normal(0, 0.1, n)
    y = 2 * x1 - x2 + rng.normal(0, 0.2, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})
    df.loc[df.index % 5 == 0, "x3"] = np.nan
    return df


def test_fill_plan_defers_mean_and_fits_on_train_rows_only():
    df = _frame_with_a_gap()
    plan = FillPlan()
    out = preprocess_data(
        df,
        target_cols=["y"],
        indicator_cols=["x1", "x2", "x3"],
        drop_missing="indicator",
        impute_strategy="mean",
        drop_zero="none",
        zero_handle="none",
        fill_plan_out=plan,
    )
    # Deferred: the NaN survives preprocessing, and the plan names the column.
    assert out["x3"].isna().sum() == 40
    assert "x3" in plan.nan_columns and plan.nan_strategy == "mean"

    train_idx, _ = train_test_split(out.index, test_size=0.2, random_state=7)
    filled = plan.fit_transform_by_index(out, train_idx)
    train_only_mean = out.loc[train_idx, "x3"].mean()
    whole_file_mean = out["x3"].mean()
    assert plan.nan_fills["x3"] == float(train_only_mean)
    assert plan.nan_fills["x3"] != float(whole_file_mean)
    assert filled["x3"].isna().sum() == 0


def test_training_run_records_train_split_fills_and_reuses_them(tmp_path):
    df = _frame_with_a_gap()
    csv = tmp_path / "gap.csv"
    df.to_csv(csv, index=False)
    store = {}
    result, _ = _train(store, str(csv), [0, 1, 2], [3], "Linear", dropMissing="indicator", imputeStrategy="mean")
    fills = store["impute_fill_values"]
    assert "x3" in fills
    # The stored fill is the training-split mean under the same seed and test size the
    # pipeline used, not the whole-file mean.
    rows = df.dropna(subset=["x1", "x2", "y"])
    train_idx, _ = train_test_split(rows.index, test_size=0.2, random_state=42)
    assert fills["x3"] == float(rows.loc[train_idx, "x3"].mean())
    assert fills["x3"] != float(rows["x3"].mean())
    # Inference on a file with its own gaps uses the stored value.
    new = pd.DataFrame({"x1": [0.0, 1.0], "x2": [0.0, 1.0], "x3": [np.nan, 4.0]})
    filled = preprocess_data(
        new,
        target_cols=None,
        indicator_cols=["x1", "x2", "x3"],
        drop_missing="indicator",
        impute_strategy="mean",
        drop_zero="none",
        zero_handle="none",
        fill_values=fills,
    )
    assert filled.loc[0, "x3"] == fills["x3"]
