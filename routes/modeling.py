"""Model training (process) route."""

import logging
import os
import threading
import uuid

from flask import Blueprint, jsonify, request

from core import jobs, queue
from core.constants import HTTP_ACCEPTED, HTTP_BAD_REQUEST, HTTP_TOO_MANY_REQUESTS
from core.state import get_session_id, get_session_storage
from python_scripts import config
from python_scripts.app_model_training import run_model_training, validate_training_request
from python_scripts.preprocessing.progress_tracker import get_tracker

bp = Blueprint("modeling", __name__)
logger = logging.getLogger(__name__)


def _run_model_training(session_id: str, data: dict, storage_session_id: str, job_id: str | None = None):
    """Run model training in a background thread.

    A new thread starts with default context variables, so the output scope has to
    be set here or the run would write into the shared base directory.
    """
    config.set_session_scope(storage_session_id)
    try:
        run_model_training(session_id, data, storage_session_id, get_storage=get_session_storage)
        if job_id:
            queue.finish(job_id, queue.DONE)
    except BaseException as e:
        if job_id:
            queue.finish(job_id, queue.FAILED, str(e)[:500])
        raise
    finally:
        jobs.release(storage_session_id, session_id)
        # A slot has come free, so the next waiting run can start.
        if queue.queue_enabled():
            _start_next_waiting()


def _start_next_waiting() -> None:
    """Take the oldest waiting run, if a slot is free, and start it."""
    claimed = None
    try:
        queue.requeue_lost()
        pending = queue.claim(worker=f"pid-{os.getpid()}")
        if pending is None:
            return
        claimed = pending
        try:
            jobs.acquire(pending["session_id"], pending["id"])
        except jobs.SlotUnavailable:
            # No slot after all: put it back at the front of the line.
            queue.finish(pending["id"], queue.QUEUED)
            return
        get_tracker(pending["id"], pending["session_id"])
        threading.Thread(
            target=_run_model_training,
            args=(pending["id"], pending["request"], pending["session_id"], pending["id"]),
            daemon=True,
        ).start()
    except Exception as e:
        logger.warning("A waiting run could not be started: %s", e)
        if claimed is not None:
            queue.finish(claimed["id"], queue.FAILED, str(e)[:500])


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
        # Where runs wait for each other, a full slot means the run is queued
        # rather than lost. Where they do not, the answer is still no.
        if not queue.queue_enabled():
            return jsonify({"error": e.reason}), HTTP_TOO_MANY_REQUESTS
        try:
            queued = queue.enqueue(session_id, storage_session_id, data)
        except queue.QueueFull as full:
            return jsonify({"error": str(full)}), HTTP_TOO_MANY_REQUESTS
        get_tracker(session_id, storage_session_id)
        return jsonify(
            {
                "session_id": session_id,
                "status": "queued",
                "position": queued["position"],
                "message": f"{e.reason} This run is waiting; it starts when a slot is free.",
            }
        ), HTTP_ACCEPTED

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


@bp.route("/queue/<job_id>")
def queue_status(job_id):
    """Where a waiting run stands, for a page that was told its run is queued."""
    job = queue.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown run."}), HTTP_BAD_REQUEST
    if job["session_id"] != get_session_id():
        return jsonify({"error": "Unknown run."}), HTTP_BAD_REQUEST
    return jsonify(
        {
            "id": job["id"],
            "state": job["state"],
            "position": queue.position(job_id),
            "detail": job.get("detail"),
        }
    )
