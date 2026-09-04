"""Model preprocessing route."""

import logging

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request

from core.constants import COLUMN_NAME_DISPLAY_LENGTH, HTTP_BAD_REQUEST
from core.state import get_session_storage

bp = Blueprint("preprocess", __name__)
logger = logging.getLogger(__name__)


def _column_letter(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA."""
    letters = ""
    while index >= 0:
        letters = chr(index % 26 + 65) + letters
        index = index // 26 - 1
    return letters


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
        return jsonify({"error": "No data uploaded. Upload a file first."}), HTTP_BAD_REQUEST

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
    except (KeyError, IndexError, TypeError):
        try:
            predictor_names = df.columns[selected_predictors].tolist()
            indicator_names = df.columns[selected_indicators].tolist()
        except Exception as e2:
            logger.error("Error in preprocess: %s", e2, exc_info=True)
            n_cols = len(df.columns)
            return jsonify(
                {
                    "error": (
                        f"A selected column is outside the file, which has {n_cols} columns "
                        f"(A to {_column_letter(n_cols - 1)}). Column letters are written like A-D or A-D,F,H."
                    )
                }
            ), HTTP_BAD_REQUEST

    try:
        if isinstance(stratify, (int, np.integer)):
            stratify_name = df.columns.take([stratify]).tolist()[0]
        else:
            stratify_name = df.columns[stratify]
            if isinstance(stratify_name, pd.Index):
                stratify_name = stratify_name.tolist()[0] if len(stratify_name) > 0 else str(stratify_name)
            else:
                stratify_name = str(stratify_name)
    except (KeyError, IndexError, TypeError, ValueError):
        # A stratify value that names no column is a client error, not a server fault.
        return jsonify(
            {"error": f"Stratify selection {stratify!r} does not match a column in this file."}
        ), HTTP_BAD_REQUEST

    output_type = (data.get("outputType") or "").strip()
    # Clustering has no supervised target. Classification stratifies on the class
    # labels, which is what train_test_split's own stratify argument does, so naming
    # the target there is correct rather than a leak. The guard is for regression,
    # where the target is cut into bins and the bins would carry it into the split.
    # The client sends stratify as a bare index or a one-element list; both are checked.
    if output_type not in ("Cluster", "Classifier") and selected_predictors is not None:
        strat_list = stratify if isinstance(stratify, (list, tuple)) else [stratify]
        strat_idx = {int(s) for s in strat_list if isinstance(s, (int, np.integer))}
        pred_raw = selected_predictors if isinstance(selected_predictors, (list, tuple)) else [selected_predictors]
        pred_list = {int(p) for p in pred_raw if isinstance(p, (int, np.integer))}
        if strat_idx & pred_list:
            return jsonify(
                {
                    "error": "The stratify column is the target column, which leaks the target into the split. Select "
                    "a different column."
                }
            ), HTTP_BAD_REQUEST

    store["last_preprocess_request"] = {
        "indicators": selected_indicators,
        "predictors": selected_predictors,
        "stratify": stratify,
    }
    return jsonify(
        {
            "predictors": [str(p)[:COLUMN_NAME_DISPLAY_LENGTH] for p in predictor_names],
            "indicators": [str(i)[:COLUMN_NAME_DISPLAY_LENGTH] for i in indicator_names],
            "stratify": stratify_name[:COLUMN_NAME_DISPLAY_LENGTH] if stratify_name else "",
        }
    )
