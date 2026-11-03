"""Prediction (inference) handler for DiGiTerra.

Runs the trained model on uploaded data and builds the response for the inference results page.
Invoked by app.py predict route. Receives store, request, and helpers to avoid circular imports.
"""

import json
import logging
import os
import uuid
from pathlib import Path

import pandas as pd
from werkzeug.utils import secure_filename

from core.constants import MAX_CELLS, MAX_FRAME_MB
from core.helpers import FrameTooLarge
from core.worker import WorkerError, WorkerLimitUnavailable, read_table_isolated
from python_scripts.helpers import prediction, preprocess_data
from python_scripts.plotting.plot_predictor_spread import SPREAD_FILE, plot_predictor_spread

logger = logging.getLogger(__name__)


def run_predict(
    store: dict,
    request,
    upload_folder: str,
    user_vis_dir: Path,
    with_prefix_fn,
    allowed_file_fn,
    normalize_preprocess_mode_fn,
):
    """Run prediction on uploaded file and build response.

    Returns (response_dict, status_code). On error, response_dict has 'error' key.
    """
    if "predictFile" not in request.files:
        return ({"error": "The request has no file. Select a file and upload it again."}, 400)
    file = request.files["predictFile"]
    if file.filename == "":
        return ({"error": "No file is selected. Select a file and click Upload."}, 400)
    if not allowed_file_fn(file.filename):
        return ({"error": "DiGiTerra cannot read this file type. Use a .csv, .xlsx, .xlsm, or .parquet file."}, 400)
    safe_filename_str = secure_filename(file.filename)
    if not safe_filename_str:
        return ({"error": "The file name is not valid. Rename the file and upload it again."}, 400)
    unique_prefix = uuid.uuid4().hex[:8]
    stored_filename = f"{unique_prefix}_{safe_filename_str}"
    filepath = os.path.join(upload_folder, stored_filename)
    file.save(filepath)
    try:
        data = read_table_isolated(filepath, max_cells=MAX_CELLS, max_frame_mb=MAX_FRAME_MB)
    except FrameTooLarge as e:
        return ({"error": str(e)}, 400)
    except WorkerLimitUnavailable as e:
        logger.error("Isolated parse unavailable on this platform: %s", e)
        return (
            {
                "error": "DiGiTerra did not read the prediction file. This installation cannot apply "
                "a memory limit to the parse. Ask the operator to run the server profile on a "
                "platform whose kernel applies RLIMIT_AS. Linux and the BSDs apply it."
            },
            500,
        )
    except WorkerError as e:
        return ({"error": f"DiGiTerra cannot read the file within the limits for one upload. {e}"}, 400)
    except pd.errors.EmptyDataError:
        return ({"error": "The file is empty. Upload a file that contains data."}, 400)
    except pd.errors.ParserError as e:
        logger.error("Parsing error: %s", e)
        return ({"error": f"DiGiTerra cannot parse the file. {str(e)}"}, 400)
    except Exception as e:
        logger.error("Error reading file: %s", e, exc_info=True)
        return ({"error": f"DiGiTerra cannot read the file. {str(e)}"}, 400)
    finally:
        # Read once into memory; the copy on disk has no further use.
        try:
            os.remove(filepath)
        except OSError:
            logger.debug("Temporary prediction upload already removed or unavailable: %s", filepath)
    if data.empty:
        return ({"error": "The file has no data rows. Upload a file that contains data."}, 400)
    if data.columns.duplicated().any():
        logger.warning("Duplicate column names detected in prediction file. Renaming duplicates.")
        new_columns = []
        seen = {}
        for col in data.columns:
            if col in seen:
                seen[col] += 1
                new_columns.append(f"{col}.{seen[col]}")
            else:
                seen[col] = 0
                new_columns.append(col)
        data.columns = new_columns
    if "model" not in store or store["model"] is None:
        return ({"error": "DiGiTerra has no trained model. Train a model, then run the prediction again."}, 400)
    if "feature_order" not in store or not store["feature_order"]:
        return (
            {"error": "DiGiTerra has no recorded feature order. Train a model, then run the prediction again."},
            400,
        )
    # Cluster models: only some support assigning new points to clusters (e.g. KMeans, GMM). Spectral, DBSCAN, OPTICS,
    # etc. do not.
    if store.get("model_type") == "cluster":
        m = store["model"]
        if not (hasattr(m, "predict") and callable(getattr(m, "predict", None))):
            return (
                {
                    "error": "This cluster model (Spectral, DBSCAN, or OPTICS) has no predict method. KMeans and GMM "
                    "assign new rows to clusters."
                },
                400,
            )
    infer_cfg = store.get("inference_config", {})
    indicator_names = infer_cfg.get("indicator_names")
    drop_missing = normalize_preprocess_mode_fn(infer_cfg.get("drop_missing", "none"))
    impute_strategy = infer_cfg.get("impute_strategy", "none")
    drop_zero = normalize_preprocess_mode_fn(infer_cfg.get("drop_zero", "none"))
    zero_handle = infer_cfg.get("zero_handle")
    if zero_handle is None:
        zero_handle = "drop" if drop_zero != "none" else "none"
    elif isinstance(zero_handle, str) and zero_handle in {"0", "0.01"}:
        zero_handle = float(zero_handle)
    if drop_zero == "none":
        zero_handle = "none"
    if impute_strategy in {"0", "0.01"}:
        impute_strategy = float(impute_strategy)
    preprocess_indicator_cols = None
    if indicator_names:
        preprocess_indicator_cols = [c for c in indicator_names if c in data.columns]
    if not preprocess_indicator_cols:
        preprocess_indicator_cols = data.columns.tolist()
    df = preprocess_data(
        data,
        target_cols=None,
        indicator_cols=preprocess_indicator_cols,
        drop_missing=drop_missing,
        impute_strategy=impute_strategy,
        drop_zero=drop_zero,
        zero_handle=zero_handle,
        fill_values=store.get("impute_fill_values") or None,
        zero_fill_values=store.get("zero_fill_values") or None,
    )
    try:
        required_features = list(store["feature_order"])
        missing = sorted(set(required_features) - set(df.columns))
        if missing:
            return (
                {
                    "error": f"The prediction file has no column for these features. {missing}. "
                    "Add the columns, then upload the file again."
                },
                400,
            )
        y_scaler = store.get("y_scaler", None)
        target_names = store.get("predictor_names", None)
        if target_names is not None and hasattr(target_names, "tolist"):
            target_names = target_names.tolist()
        prediction(
            df,
            best_model=store["model"],
            training_features=required_features,
            X_scaler=store.get("X_scaler", None),
            y_scaler=y_scaler,
            feature_order=required_features,
            target_names=target_names,
            applicability=store.get("applicability"),
            conformal=store.get("conformal"),
            model_type=store.get("model_type"),
        )
        pred_path = user_vis_dir / "predictions.csv"
        applicability_summary = None
        uncertainty_summary = None
        summary = []
        predictions_preview = {}
        pred_df = None
        pred_cols = []
        if pred_path.exists():
            try:
                pred_df = pd.read_csv(pred_path)
                # How many of these predictions rest on data the model has seen.
                if "novel_row" in pred_df.columns:
                    out_of_range = pred_df.get("out_of_range_columns")
                    applicability_summary = {
                        "rows": int(len(pred_df)),
                        "novel": int(pred_df["novel_row"].fillna(False).astype(bool).sum()),
                        "out_of_range": (
                            int(out_of_range.fillna("").astype(str).str.len().gt(0).sum())
                            if out_of_range is not None
                            else 0
                        ),
                    }
                # An interval half-width, or the least confident class probability.
                lower = [c for c in pred_df.columns if c.startswith("lower_90")]
                upper = [c for c in pred_df.columns if c.startswith("upper_90")]
                if lower and upper:
                    widths = (pred_df[upper[0]] - pred_df[lower[0]]) / 2.0
                    uncertainty_summary = {
                        "kind": "interval",
                        "half_width": float(widths.dropna().iloc[0]) if widths.notna().any() else None,
                        "coverage": 90,
                        "units": str(store.get("inference_config", {}).get("units") or ""),
                    }
                elif "max_probability" in pred_df.columns:
                    lowest = pd.to_numeric(pred_df["max_probability"], errors="coerce").dropna()
                    uncertainty_summary = {
                        "kind": "probability",
                        "lowest": float(lowest.min()) if not lowest.empty else None,
                    }
                pred_cols = [c for c in pred_df.columns if c.startswith("Predicted_")]
                for col in pred_cols:
                    s = pred_df[col]
                    if pd.api.types.is_numeric_dtype(s):
                        q = s.quantile([0.25, 0.5, 0.75]).tolist()
                        summary.append(
                            {
                                "column": col,
                                "n": int(s.count()),
                                "min": float(s.min()) if s.count() else None,
                                "max": float(s.max()) if s.count() else None,
                                "mean": float(s.mean()) if s.count() else None,
                                "std": float(s.std()) if s.count() and len(s) > 1 else None,
                                "25": float(q[0]) if len(q) > 0 else None,
                                "50": float(q[1]) if len(q) > 1 else None,
                                "75": float(q[2]) if len(q) > 2 else None,
                                "100": float(s.max()) if s.count() else None,
                            }
                        )
                        predictions_preview[col] = s.dropna().head(200).tolist()
                    else:
                        vc = s.value_counts()
                        summary.append(
                            {
                                "column": col,
                                "n": int(s.count()),
                                "value_counts": [{"value": str(k), "count": int(v)} for k, v in vc.items()],
                            }
                        )
                        predictions_preview[col] = s.dropna().head(200).tolist()
            except Exception as e:
                logger.warning("Could not build prediction summary: %s", e)
        # A check against measured values, when the file carries them. The person
        # decides: a file with the target column, or the target name followed by
        # _measured, _true or _actual, is scored; a file without one is scored
        # blind, and the reply names the column that would have made the check.
        check = _score_against_measured(pred_df, target_names, store.get("model_type", "regression"))
        training_visualization = store.get("training_visualization")
        if training_visualization and not (user_vis_dir / training_visualization).exists():
            training_visualization = None
        training_visualization_version = store.get("training_visualization_version")
        model_type = store.get("model_type", "regression")
        training_target_summary = None
        try:
            summary_path = user_vis_dir / "training_target_summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    training_target_summary = json.load(f)
        except Exception as e:
            logger.debug("Could not read training target summary: %s", e)
        inference_visualization = None
        inference_visualization_version = None
        if (
            model_type == "regression"
            and pred_df is not None
            and pred_cols
            and (user_vis_dir / "training_plot_data.json").exists()
        ):
            try:
                with open(user_vis_dir / "training_plot_data.json") as f:
                    plot_data = json.load(f)
                first_col = pred_cols[0]
                inference_pred = pred_df[first_col].dropna().values.ravel()
                if len(inference_pred) > 0:
                    from python_scripts.plotting.visualize_predictions import plot_inference_pred_vs_actual

                    out_path = user_vis_dir / "inference_pred_vs_actual_1.png"
                    plot_inference_pred_vs_actual(
                        plot_data["y_train_actual"],
                        plot_data["y_train_pred"],
                        plot_data["y_test_actual"],
                        plot_data["y_test_pred"],
                        inference_pred,
                        plot_data["target_name"],
                        plot_data.get("units", ""),
                        plot_data.get("model_name", "Model"),
                        out_path,
                    )
                    inference_visualization = "inference_pred_vs_actual_1.png"
                    inference_visualization_version = str(int(__import__("time").time() * 1000))
            except Exception as e:
                logger.debug("Could not regenerate inference plot: %s", e)
        # The spread of every predictor in the training rows against the rows just
        # predicted, with the rows the novelty detector marked drawn as points.
        spread_visualization = None
        spread_visualization_version = None
        if pred_df is not None and (user_vis_dir / SPREAD_FILE).exists():
            try:
                novel = pred_df["novel_row"] if "novel_row" in pred_df.columns else None
                out_path = user_vis_dir / "predictor_spread.png"
                if plot_predictor_spread(user_vis_dir / SPREAD_FILE, pred_df, novel, out_path) is not None:
                    spread_visualization = "predictor_spread.png"
                    spread_visualization_version = str(int(__import__("time").time() * 1000))
            except Exception as e:
                logger.debug("Could not draw the predictor spread: %s", e)
        filename = safe_filename_str[:31]
        if len(safe_filename_str) > 30:
            filename += "..."
        return (
            {
                "results": with_prefix_fn("/download/predictions.csv"),
                "spread_visualization": spread_visualization,
                "spread_visualization_version": spread_visualization_version,
                "filename": filename,
                "summary": summary,
                "check": check,
                "applicability_summary": applicability_summary,
                "uncertainty_summary": uncertainty_summary,
                "predictions_preview": predictions_preview,
                "model_type": model_type,
                "training_visualization": training_visualization,
                "training_visualization_version": training_visualization_version,
                "training_target_summary": training_target_summary,
                "inference_visualization": inference_visualization,
                "inference_visualization_version": inference_visualization_version,
            },
            200,
        )
    except Exception as e:
        logger.error("Error in predict: %s", e, exc_info=True)
        err_msg = str(e) if e else "An error occurred during prediction."
        return ({"error": err_msg}, 500)


def _score_against_measured(pred_df, target_names, model_type):
    """Score Predicted_* columns against a measured column in the scored file, if any.

    Returns a dict for the page: found=False with the column names that would be
    read, or found=True with the metrics. Regression: R2, RMSE, MAE and the count.
    Classification: accuracy, macro F1, and a confusion table over the union of
    labels. Rows with a blank on either side are left out and counted.
    """
    names = list(target_names) if target_names else []
    candidates = []
    for t in names or ["Target"]:
        candidates += [t, f"{t}_measured", f"{t}_true", f"{t}_actual"]
    if pred_df is None:
        return {"found": False, "accepted": candidates}
    pred_cols = [c for c in pred_df.columns if c.startswith("Predicted_")]
    if len(pred_cols) != 1:
        return {"found": False, "accepted": candidates}
    truth_col = next((c for c in candidates if c in pred_df.columns), None)
    if truth_col is None:
        return {"found": False, "accepted": candidates}
    y_true = pred_df[truth_col]
    y_pred = pred_df[pred_cols[0]]
    keep = y_true.notna() & y_pred.notna()
    n_dropped = int((~keep).sum())
    y_true, y_pred = y_true[keep], y_pred[keep]
    out = {"found": True, "column": truth_col, "n": int(len(y_true)), "dropped": n_dropped}
    if int(len(y_true)) < 2:
        out["error"] = "Fewer than two rows carry both a prediction and a measured value."
        return out
    if model_type == "regression" and pd.api.types.is_numeric_dtype(y_pred):
        yt = pd.to_numeric(y_true, errors="coerce")
        ok = yt.notna()
        yt, yp = yt[ok].astype(float), y_pred[ok].astype(float)
        resid = yt - yp
        ss_tot = float(((yt - yt.mean()) ** 2).sum())
        out.update(
            {
                "kind": "regression",
                "n": int(len(yt)),
                "r2": float(1.0 - float((resid**2).sum()) / ss_tot) if ss_tot > 0 else None,
                "rmse": float((resid**2).mean() ** 0.5),
                "mae": float(resid.abs().mean()),
                "bias": float(resid.mean() * -1.0),
            }
        )
        return out
    yt = y_true.astype(str).str.strip()
    yp = y_pred.astype(str).str.strip()
    labels = sorted(set(yt) | set(yp))
    matrix = [[int(((yt == a) & (yp == b)).sum()) for b in labels] for a in labels]
    per_class_f1 = []
    for i, _ in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(len(labels))) - tp
        fn = sum(matrix[i]) - tp
        per_class_f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    out.update(
        {
            "kind": "classification",
            "accuracy": float((yt == yp).mean()),
            "macro_f1": float(sum(per_class_f1) / len(per_class_f1)) if per_class_f1 else None,
            "labels": labels,
            "confusion": matrix,
        }
    )
    return out
