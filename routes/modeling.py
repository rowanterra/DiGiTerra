"""Model training (process) route."""
import threading
import uuid

from flask import Blueprint, jsonify, request

from core.constants import HTTP_ACCEPTED, HTTP_BAD_REQUEST
from core.state import get_session_id, get_session_storage
from python_scripts.app_model_training import run_model_training
from python_scripts.preprocessing.progress_tracker import get_tracker

bp = Blueprint("modeling", __name__)


def _run_model_training(session_id: str, data: dict, storage_session_id: str):
    """Run model training in a background thread."""
    run_model_training(session_id, data, storage_session_id, get_storage=get_session_storage)


@bp.route("/process", methods=["POST"])
def process_columns():
    """Start model training with progress tracking. Returns session_id for SSE."""
    if not request.json:
        return jsonify({"error": "No data provided"}), HTTP_BAD_REQUEST
    store = get_session_storage()
    last_preprocess = store.get("last_preprocess_request")
    if not last_preprocess:
        return jsonify({"error": "Run Model Preprocessing before Modeling."}), HTTP_BAD_REQUEST
    req_indicators = request.json.get("indicators")
    req_predictors = request.json.get("predictors")
    if req_indicators != last_preprocess.get("indicators") or req_predictors != last_preprocess.get("predictors"):
        return jsonify({"error": "Selections changed since preprocessing. Re-run Model Preprocessing before Modeling."}), HTTP_BAD_REQUEST

    session_id = str(uuid.uuid4())
    storage_session_id = get_session_id()
    get_tracker(session_id)
    data = request.json
    threading.Thread(
        target=_run_model_training,
        args=(session_id, data, storage_session_id),
        daemon=True,
    ).start()
    return jsonify({"session_id": session_id, "status": "processing"}), HTTP_ACCEPTED
