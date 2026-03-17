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

from python_scripts.helpers import preprocess_data, prediction

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
        return ({"error": "No file uploaded"}, 400)
    file = request.files["predictFile"]
    if file.filename == "":
        return ({"error": "No file selected"}, 400)
    if not allowed_file_fn(file.filename):
        return ({"error": "Invalid file type. Only CSV files are allowed."}, 400)
    safe_filename_str = secure_filename(file.filename)
    if not safe_filename_str:
        return ({"error": "Invalid filename."}, 400)
    unique_prefix = uuid.uuid4().hex[:8]
    stored_filename = f"{unique_prefix}_{safe_filename_str}"
    filepath = os.path.join(upload_folder, stored_filename)
    file.save(filepath)
    try:
        data = pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        return ({"error": "The uploaded file is empty."}, 400)
    except pd.errors.ParserError as e:
        logger.error("CSV parsing error: %s", e)
        return ({"error": f"Error parsing CSV file: {str(e)}"}, 400)
    except Exception as e:
        logger.error("Error reading CSV file: %s", e, exc_info=True)
        return ({"error": f"Error reading file: {str(e)}"}, 400)
    if data.empty:
        return ({"error": "The uploaded file contains no data."}, 400)
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
        return ({"error": "No trained model found. Train a model first."}, 400)
    if "feature_order" not in store or not store["feature_order"]:
        return ({"error": "No feature_order found. Train a model first."}, 400)
    # Cluster models: only some support assigning new points to clusters (e.g. KMeans, GMM). Spectral, DBSCAN, OPTICS, etc. do not.
    if store.get("model_type") == "cluster":
        m = store["model"]
        if not (hasattr(m, "predict") and callable(getattr(m, "predict", None))):
            return ({"error": "Inference on new data is not supported for this cluster model (e.g. Spectral, DBSCAN, OPTICS). Use a model that supports prediction (e.g. KMeans, GMM) to assign new points to clusters."}, 400)
    infer_cfg = store.get("inference_config", {})
    indicator_names = infer_cfg.get("indicator_names")
    drop_missing = normalize_preprocess_mode_fn(infer_cfg.get("drop_missing", "none"))
    impute_strategy = infer_cfg.get("impute_strategy", "none")
    drop_zero = normalize_preprocess_mode_fn(infer_cfg.get("drop_zero", "none"))
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
    )
    try:
        required_features = list(store["feature_order"])
        missing = sorted(set(required_features) - set(df.columns))
        if missing:
            return ({"error": f"Missing features in prediction file: {missing}"}, 400)
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
        )
        pred_path = user_vis_dir / "predictions.csv"
        summary = []
        predictions_preview = {}
        pred_df = None
        pred_cols = []
        if pred_path.exists():
            try:
                pred_df = pd.read_csv(pred_path)
                pred_cols = [c for c in pred_df.columns if c.startswith("Predicted_")]
                for col in pred_cols:
                    s = pred_df[col]
                    if pd.api.types.is_numeric_dtype(s):
                        q = s.quantile([0.25, 0.5, 0.75]).tolist()
                        summary.append({
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
                        })
                        predictions_preview[col] = s.dropna().head(200).tolist()
                    else:
                        vc = s.value_counts()
                        summary.append({
                            "column": col,
                            "n": int(s.count()),
                            "value_counts": [{"value": str(k), "count": int(v)} for k, v in vc.items()],
                        })
                        predictions_preview[col] = s.dropna().head(200).tolist()
            except Exception as e:
                logger.warning("Could not build prediction summary: %s", e)
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
        if model_type == "regression" and pred_df is not None and pred_cols and (user_vis_dir / "training_plot_data.json").exists():
            try:
                with open(user_vis_dir / "training_plot_data.json") as f:
                    plot_data = json.load(f)
                first_col = pred_cols[0]
                inference_pred = pred_df[first_col].dropna().values.ravel()
                if len(inference_pred) > 0:
                    from python_scripts.plotting.visualize_predictions import plot_inference_pred_vs_actual
                    out_path = user_vis_dir / "inference_pred_vs_actual_1.png"
                    plot_inference_pred_vs_actual(
                        plot_data["y_train_actual"], plot_data["y_train_pred"],
                        plot_data["y_test_actual"], plot_data["y_test_pred"],
                        inference_pred,
                        plot_data["target_name"], plot_data.get("units", ""),
                        plot_data.get("model_name", "Model"), out_path,
                    )
                    inference_visualization = "inference_pred_vs_actual_1.png"
                    inference_visualization_version = str(int(__import__("time").time() * 1000))
            except Exception as e:
                logger.debug("Could not regenerate inference plot: %s", e)
        filename = safe_filename_str[:31]
        if len(safe_filename_str) > 30:
            filename += "..."
        return (
            {
                "results": with_prefix_fn("/download/predictions.csv"),
                "filename": filename,
                "summary": summary,
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
