"""Score new rows with a DiGiTerra model bundle, without the web app.

The bundle's fitted objects are unpickled, so run this only on bundles from a
source you trust. The script applies the training-time preprocessing recorded in
the manifest (drop or fill rules, transformer, scaler, feature order) and writes
one Predicted_<target> column per target.

Usage:
    .venv/bin/python scripts/score_bundle.py BUNDLE.zip NEW_ROWS.csv [--out predictions.csv]
    .venv/bin/python scripts/score_bundle.py BUNDLE.zip --show    # print the manifest only

Input: a csv, xlsx, xlsm, or parquet file with the feature columns the manifest
lists under model.features. Extra columns are ignored; missing ones stop the run.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_sup = Path(tempfile.mkdtemp(prefix="dgt_score_"))
os.environ.setdefault("DIGITERRA_APP_SUPPORT_DIR", str(_sup))
os.environ.setdefault("DIGITERRA_OUTPUT_DIR", str(_sup / "vis"))
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd  # noqa: E402

from core.helpers import normalize_predict_preprocess_mode, read_table_bounded  # noqa: E402
from python_scripts import config  # noqa: E402
from python_scripts.helpers import prediction, preprocess_data  # noqa: E402
from python_scripts.reporting.bundle import load_bundle, package_drift, read_manifest  # noqa: E402


def score(bundle_path: str, rows_path: str) -> pd.DataFrame:
    manifest, objects = load_bundle(bundle_path, force=True)
    drift = package_drift(manifest)
    if drift:
        print(f"note: installed versions differ from the bundle for {', '.join(drift)}", file=sys.stderr)
    data = read_table_bounded(rows_path, max_cells=0, max_frame_mb=0)
    cfg = objects.get("inference_config") or {}
    indicator_names = cfg.get("indicator_names") or list(objects["feature_order"])
    drop_zero = normalize_predict_preprocess_mode(cfg.get("drop_zero", "none"))
    zero_handle = cfg.get("zero_handle")
    if zero_handle is None:
        zero_handle = "drop" if drop_zero != "none" else "none"
    elif isinstance(zero_handle, str) and zero_handle in {"0", "0.01"}:
        zero_handle = float(zero_handle)
    if drop_zero == "none":
        zero_handle = "none"
    impute_strategy = cfg.get("impute_strategy", "none")
    if impute_strategy in {"0", "0.01"}:
        impute_strategy = float(impute_strategy)
    cols = [c for c in indicator_names if c in data.columns] or data.columns.tolist()
    df = preprocess_data(
        data,
        target_cols=None,
        indicator_cols=cols,
        drop_missing=normalize_predict_preprocess_mode(cfg.get("drop_missing", "none")),
        impute_strategy=impute_strategy,
        drop_zero=drop_zero,
        zero_handle=zero_handle,
        fill_values=objects.get("impute_fill_values") or None,
        zero_fill_values=objects.get("zero_fill_values") or None,
    )
    missing = sorted(set(objects["feature_order"]) - set(df.columns))
    if missing:
        raise SystemExit(f"Missing features in {rows_path}: {missing}")
    targets = objects.get("predictor_names")
    if targets is not None and hasattr(targets, "tolist"):
        targets = targets.tolist()
    # prediction() writes predictions.csv into the session output directory.
    out_dir = config.VIS_DIR.current()
    prediction(
        df,
        best_model=objects["model"],
        training_features=list(objects["feature_order"]),
        X_scaler=objects.get("X_scaler"),
        y_scaler=objects.get("y_scaler"),
        feature_order=list(objects["feature_order"]),
        target_names=targets,
    )
    return pd.read_csv(out_dir / "predictions.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle")
    ap.add_argument("rows", nargs="?")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--show", action="store_true", help="print the manifest and exit")
    args = ap.parse_args()
    if args.show or not args.rows:
        print(json.dumps(read_manifest(args.bundle), indent=2))
        return 0
    frame = score(args.bundle, args.rows)
    frame.to_csv(args.out, index=False)
    print(f"wrote {len(frame)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
