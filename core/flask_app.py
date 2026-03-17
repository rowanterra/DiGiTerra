# DiGiTerra Flask app. Routes in routes/; session state in core.state. See docs/HANDOFF.md.

import os
import uuid
import logging
from pathlib import Path

from flask import Flask, g, request, send_from_directory

from python_scripts import config

BASE_DIR = config.BASE_DIR
APP_SUPPORT_DIR = config.APP_SUPPORT_DIR
UPLOAD_DIR = config.UPLOAD_DIR
LOG_DIR = config.LOG_DIR
URL_PREFIX = config.URL_PREFIX

_user_vis_path = Path(os.environ.get("DIGITERRA_OUTPUT_DIR", str(APP_SUPPORT_DIR / "user_visualizations")))
os.environ.setdefault("DIGITERRA_OUTPUT_DIR", str(_user_vis_path))
_user_vis_path.mkdir(parents=True, exist_ok=True)
config.update_vis_dir(_user_vis_path)


def _configure_logging():
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S", force=True)
    root = logging.getLogger()
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        log_file = LOG_DIR / f"digiterra_{datetime.now().strftime('%Y-%m-%d')}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        root.addHandler(fh)
    except OSError:
        pass
    return logging.getLogger(__name__)


logger = _configure_logging()

from core.state import SESSION_COOKIE_NAME
from routes import register_blueprints

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.before_request
def _ensure_session_cookie():
    existing = request.cookies.get(SESSION_COOKIE_NAME)
    if existing:
        g.session_id = existing
        g.new_session_id = None
        return
    new_id = uuid.uuid4().hex
    g.session_id = new_id
    g.new_session_id = new_id


@app.after_request
def _set_session_cookie(response):
    new_id = getattr(g, "new_session_id", None)
    if new_id:
        response.set_cookie(SESSION_COOKIE_NAME, new_id, httponly=True, samesite="Lax")
    return response


register_blueprints(app)

if URL_PREFIX:
    @app.route(f"{URL_PREFIX}/static/<path:filename>")
    def prefixed_static(filename):
        return send_from_directory(app.static_folder, filename)


def create_app():
    return app


def run_app(host=None, port=None, debug=False):
    host = host or os.environ.get("DIGITERRA_HOST", "127.0.0.1")
    port = port or int(os.environ.get("DIGITERRA_PORT", "5000"))
    app.run(host=host, port=port, debug=debug)
