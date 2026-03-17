"""Model preprocessing route."""
import logging

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request

from app_constants import COLUMN_NAME_DISPLAY_LENGTH, HTTP_BAD_REQUEST
from app_state import get_session_storage

bp = Blueprint("preprocess", __name__)
logger = logging.getLogger(__name__)


@bp.route("/preprocess", methods=["POST"])
def preprocess():
    """Preprocess data and return selected column names."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), HTTP_BAD_REQUEST
    if "indicators" not in data or "predictors" not in data or "stratify" not in data:
        return jsonify({"error": "Missing required fields: indicators, predictors, stratify"}), HTTP_BAD_REQUEST

    store = get_session_storage()
    if "data" not in store:
        return jsonify({"error": "No data uploaded. Please upload a file first."}), HTTP_BAD_REQUEST

    required_fields = ["indicators", "predictors", "stratify"]
    missing_fields = [f for f in required_fields if f not in data]
    if missing_fields:
        return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), HTTP_BAD_REQUEST

    df = store["data"]
    try:
        selected_indicators = data["indicators"]
        selected_predictors = data["predictors"]
        stratify = data["stratify"]
        if isinstance(selected_predictors, (int, np.integer)):
            selected_predictors = [selected_predictors]
        if isinstance(selected_indicators, (int, np.integer)):
            selected_indicators = [selected_indicators]
        predictor_names = df.columns.take(selected_predictors).tolist()
        indicator_names = df.columns.take(selected_indicators).tolist()
    except (KeyError, IndexError, TypeError) as e:
        try:
            predictor_names = df.columns[selected_predictors].tolist()
            indicator_names = df.columns[selected_indicators].tolist()
        except Exception as e2:
            logger.error("Error in preprocess: %s", e2, exc_info=True)
            return jsonify({"error": f"Invalid column indices provided. This may be due to duplicate column names in your CSV file: {str(e2)}"}), HTTP_BAD_REQUEST

    try:
        if isinstance(stratify, (int, np.integer)):
            stratify_name = df.columns.take([stratify]).tolist()[0]
        else:
            stratify_name = df.columns[stratify]
            if isinstance(stratify_name, pd.Index):
                stratify_name = stratify_name.tolist()[0] if len(stratify_name) > 0 else str(stratify_name)
            else:
                stratify_name = str(stratify_name)
    except (KeyError, IndexError):
        stratify_name = str(df.columns[stratify]) if stratify < len(df.columns) else ""

    output_type = (data.get("outputType") or "").strip()
    if output_type != "Cluster":
        strat_idx = stratify if isinstance(stratify, (int, np.integer)) else None
        if strat_idx is not None and selected_predictors is not None:
            pred_list = selected_predictors if isinstance(selected_predictors, (list, tuple)) else [selected_predictors]
            if strat_idx in pred_list:
                return jsonify({
                    "error": "Do not stratify by your target column. That would leak target information into the train/test split. Choose a different column to stratify by."
                }), HTTP_BAD_REQUEST

    store["last_preprocess_request"] = {
        "indicators": selected_indicators,
        "predictors": selected_predictors,
        "stratify": stratify,
    }
    return jsonify({
        "predictors": [str(p)[:COLUMN_NAME_DISPLAY_LENGTH] for p in predictor_names],
        "indicators": [str(i)[:COLUMN_NAME_DISPLAY_LENGTH] for i in indicator_names],
        "stratify": stratify_name[:COLUMN_NAME_DISPLAY_LENGTH] if stratify_name else "",
    })
