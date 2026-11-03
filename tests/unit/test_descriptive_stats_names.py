"""Descriptive statistics must name the column, not the first ten characters of it.

The table cut every name to ten characters server-side, so bill_length_mm read as
bill_lengt and body_mass_g as body_mass_. The width belongs to the stylesheet; the
payload should carry the name the file uses.
"""

import warnings
from pathlib import Path

import pandas as pd

from python_scripts.app_exploration import handle_corr

warnings.filterwarnings("ignore")

NUMERIC = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]


def test_the_payload_carries_the_whole_column_name(tmp_path):
    frame = pd.read_csv("examples/penguins.csv").dropna()[NUMERIC]
    payload, status = handle_corr(
        {"data": frame},
        {"colsIgnore": "all", "dropMissing": "none", "imputeStrategy": "none", "dropZero": "none"},
        Path(tmp_path),
        lambda p: p,
    )
    assert status == 200, payload
    names = [row["column"] for row in payload["descriptive_stats"]]
    assert set(names) == set(NUMERIC), names
    # The three names the report showed cut.
    assert not any(n in ("bill_lengt", "flipper_le", "body_mass_") for n in names), names
    assert max(len(n) for n in names) > 10, "no name here is long enough to prove the point"
