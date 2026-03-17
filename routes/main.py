"""Main page and progress SSE."""
import json
import time

from flask import Blueprint, Response, render_template, stream_with_context

from app_constants import (
    PROGRESS_COMPLETE_THRESHOLD,
    PROGRESS_FINAL_UPDATE_DELAY_SECONDS,
    PROGRESS_STREAM_TIMEOUT_SECONDS,
    PROGRESS_UPDATE_INTERVAL_SECONDS,
)
from app_helpers import with_prefix
from python_scripts import config
from python_scripts.preprocessing.progress_tracker import get_result, get_tracker, remove_tracker

bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    """Render the main HTML page."""
    return render_template(
        "index.html",
        api_root=config.URL_PREFIX,
        static_root=with_prefix("/static"),
    )


@bp.route("/progress/<session_id>")
def progress_stream(session_id):
    """Server-Sent Events endpoint for progress updates."""
    def generate():
        tracker = get_tracker(session_id)
        start_time = time.time()
        while True:
            if time.time() - start_time > PROGRESS_STREAM_TIMEOUT_SECONDS:
                yield f"data: {json.dumps({'error': 'Progress stream timeout'})}\n\n"
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

    return Response(stream_with_context(generate()), mimetype="text/event-stream")
