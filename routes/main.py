"""Main page and progress SSE."""

import json
import logging
import os
import time
import uuid

from flask import Blueprint, Response, jsonify, render_template, send_file, stream_with_context

from core import jobs
from core.constants import (
    ALLOW_UNSAFE_OVERRIDE,
    HTTP_NOT_FOUND,
    HTTP_TOO_MANY_REQUESTS,
    PROGRESS_COMPLETE_THRESHOLD,
    PROGRESS_FINAL_UPDATE_DELAY_SECONDS,
    PROGRESS_STREAM_TIMEOUT_SECONDS,
    PROGRESS_UPDATE_INTERVAL_SECONDS,
)
from core.helpers import with_prefix
from core.state import get_session_id
from python_scripts import config
from python_scripts.preprocessing.progress_tracker import get_result, peek_tracker, remove_tracker

logger = logging.getLogger(__name__)

bp = Blueprint("main", __name__)

# The example files the welcome screen may load. An allowlist rather than a path
# taken from the request, so the route cannot be asked for anything else.
EXAMPLE_FILES = {
    "penguins.csv",
    "iris.csv",
    "wine.csv",
    "breast_cancer.csv",
    "diabetes.csv",
    "linnerud.csv",
    "synthetic_regression.csv",
    "iris_clustering.csv",
    "wine_clustering.csv",
    "breast_cancer_clustering.csv",
}


@bp.route("/example-data/<name>", methods=["GET"])
def example_data(name):
    """Serve one example file so the page can load it through the ordinary upload.

    The welcome screen fetches this, builds a File from it, and submits the upload
    form, so an example goes through the same parse, the same limits and the same
    data checks as a file a person chooses.
    """
    if name not in EXAMPLE_FILES:
        return jsonify({"error": "That example is not available."}), HTTP_NOT_FOUND
    path = os.path.join(config.BASE_DIR, "examples", name)
    if not os.path.exists(path):
        return jsonify({"error": "That example is not installed."}), HTTP_NOT_FOUND
    return send_file(path, mimetype="text/csv", as_attachment=False, download_name=name)


@bp.route("/")
def index():
    """Render the main HTML page."""
    use_js_bundle = os.environ.get("DIGITERRA_USE_JS_BUNDLE", "").strip().lower() in ("1", "true", "yes")
    return render_template(
        "index.html",
        api_root=config.URL_PREFIX,
        static_root=with_prefix("/static"),
        use_js_bundle=use_js_bundle,
        allow_unsafe_override=ALLOW_UNSAFE_OVERRIDE,
    )


@bp.route("/progress/<session_id>")
def progress_stream(session_id):
    """Server-Sent Events endpoint for progress updates."""
    # Resolve the tracker before opening the stream. An unknown id must 404 rather
    # than hold a connection open until the stream timeout.
    tracker = peek_tracker(session_id)
    if tracker is None:
        return jsonify({"error": "Unknown training session."}), HTTP_NOT_FOUND
    # A run answers only the browser session that started it.
    if tracker.owner_session_id and tracker.owner_session_id != get_session_id():
        return jsonify({"error": "Unknown training session."}), HTTP_NOT_FOUND

    # One stream per run and a fixed number across the process, so streams cannot
    # hold every request thread. The slot is released when the generator ends or
    # the response is closed, whichever happens; a never-started generator is
    # covered by call_on_close.
    stream_token = uuid.uuid4().hex
    try:
        jobs.stream_acquire(session_id, stream_token)
    except jobs.SlotUnavailable as e:
        return jsonify({"error": e.reason}), HTTP_TOO_MANY_REQUESTS

    def release():
        jobs.stream_release(session_id, stream_token)

    def generate():
        start_time = time.time()
        try:
            yield from _progress_events(tracker, session_id, start_time)
        finally:
            release()

    response = Response(stream_with_context(generate()), mimetype="text/event-stream")
    response.call_on_close(release)
    return response


def _progress_events(tracker, session_id, start_time):
    while True:
        if time.time() - start_time > PROGRESS_STREAM_TIMEOUT_SECONDS:
            # End the stream without an error payload. EventSource reconnects on
            # its own, the tracker is still there, and the run continues. This
            # keeps one long run from holding a worker thread indefinitely.
            logger.info("Progress stream for %s reached its time limit; the client may reconnect.", session_id)
            break
        try:
            progress = tracker.get_progress()
            yield f"data: {json.dumps(progress)}\n\n"
            if progress["overall_progress"] >= PROGRESS_COMPLETE_THRESHOLD:
                time.sleep(PROGRESS_FINAL_UPDATE_DELAY_SECONDS)
                result = get_result(session_id)
                for _ in range(3):
                    if result is not None:
                        break
                    time.sleep(0.5)
                    result = get_result(session_id)
                if result is not None:
                    yield f"data: {json.dumps({'type': 'result', 'data': result})}\n\n"
                else:
                    yield f"data: {json.dumps({'error': 'Result not available after training completed'})}\n\n"
                remove_tracker(session_id)
                break
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            break
        time.sleep(PROGRESS_UPDATE_INTERVAL_SECONDS)
