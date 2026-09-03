"""JSON error handling.

Every route in this app answers a fetch() call, and the client parses the body as
JSON. A Flask default error page is HTML, so an unhandled exception reaches the
browser as an unparseable body and the user sees a generic failure with no cause.
These handlers keep the content type stable and keep server internals out of the
response.

Adapted from the errors module of the July restructure.
"""

from __future__ import annotations

import logging

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

logger = logging.getLogger(__name__)


class ApiError(Exception):
    """An error safe to surface to the client with a specific status code."""

    def __init__(self, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


def register_error_handlers(app: Flask) -> None:
    """Attach JSON handlers for ApiError, HTTP errors and unexpected exceptions."""

    @app.errorhandler(ApiError)
    def _handle_api_error(exc: ApiError):
        return jsonify({"error": exc.message}), exc.status

    @app.errorhandler(RequestEntityTooLarge)
    def _handle_too_large(exc: RequestEntityTooLarge):
        limit = app.config.get("MAX_CONTENT_LENGTH") or 0
        mb = limit / (1024 * 1024)
        return jsonify({"error": f"File is larger than the {mb:.0f} MB upload limit."}), 413

    @app.errorhandler(HTTPException)
    def _handle_http(exc: HTTPException):
        # Page requests still get the HTML error page; API calls get JSON.
        if request.path == "/" or request.accept_mimetypes.accept_html and not request.is_json:
            if exc.code == 404 and not request.path.startswith("/api"):
                return exc
        return jsonify({"error": exc.description}), exc.code or 500

    @app.errorhandler(Exception)
    def _handle_unexpected(exc: Exception):
        logger.exception("Unhandled error on %s", request.path)
        return jsonify({"error": "The server could not complete that request. See the log for details."}), 500
