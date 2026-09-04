"""Model training (process) route."""

import threading
import uuid

from flask import Blueprint, jsonify, request

from core import jobs
from core.constants import HTTP_ACCEPTED, HTTP_BAD_REQUEST, HTTP_TOO_MANY_REQUESTS
from core.state import get_session_id, get_session_storage
from python_scripts import config
from python_scripts.app_model_training import run_model_training, validate_training_request
from python_scripts.preprocessing.progress_tracker import get_tracker

bp = Blueprint("modeling", __name__)


def _run_model_training(session_id: str, data: dict, storage_session_id: str):
    """Run model training in a background thread.

    A new thread starts with default context variables, so the output scope has to
    be set here or the run would write into the shared base directory.
    """
    config.set_session_scope(storage_session_id)
    try:
        run_model_training(session_id, data, storage_session_id, get_storage=get_session_storage)
    finally:
        jobs.release(storage_session_id, session_id)


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
        return jsonify(
            {"error": "Selections changed since preprocessing. Re-run Model Preprocessing before Modeling."}
        ), HTTP_BAD_REQUEST

    data = request.json
    # Check the request before a thread exists, so a bad or oversized value costs
    # a parse rather than a training run.
    try:
        validate_training_request(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), HTTP_BAD_REQUEST

    session_id = str(uuid.uuid4())
    storage_session_id = get_session_id()
    # One run per session, and a fixed number across the process.
    try:
        jobs.acquire(storage_session_id, session_id)
    except jobs.SlotUnavailable as e:
        return jsonify({"error": e.reason}), HTTP_TOO_MANY_REQUESTS

    get_tracker(session_id, storage_session_id)
    try:
        threading.Thread(
            target=_run_model_training,
            args=(session_id, data, storage_session_id),
            daemon=True,
        ).start()
    except Exception:
        jobs.release(storage_session_id, session_id)
        raise
    return jsonify({"session_id": session_id, "status": "processing"}), HTTP_ACCEPTED
