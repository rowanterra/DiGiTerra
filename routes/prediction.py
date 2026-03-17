"""Prediction (inference) route."""
from flask import Blueprint, current_app, jsonify, request

from core.helpers import ensure_user_vis_dir, normalize_predict_preprocess_mode, with_prefix, allowed_file
from core.state import get_session_storage
from python_scripts.app_prediction import run_predict

bp = Blueprint("prediction", __name__)


@bp.route("/predict", methods=["POST"])
def predict():
    """Generate predictions using a trained model."""
    store = get_session_storage()
    result, status = run_predict(
        store,
        request,
        current_app.config["UPLOAD_FOLDER"],
        ensure_user_vis_dir(),
        with_prefix,
        allowed_file,
        normalize_predict_preprocess_mode,
    )
    return jsonify(result), status
