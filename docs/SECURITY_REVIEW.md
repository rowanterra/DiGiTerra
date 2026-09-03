# Security posture

Reviewed against the `audit-fixes` branch on 2026-09-03. The statements below describe the code on that branch. DiGiTerra is a single-user desktop application that also runs as a web app behind gunicorn. There is no authentication, CSRF protection, or rate limiting. A multi-user deployment needs those before exposure to untrusted clients.

## Uploads

Only `.csv` files are accepted. `allowed_file` in `core/helpers.py` checks the extension, and `routes/upload.py` rejects anything else with a 400.

Uploaded filenames pass through `werkzeug.utils.secure_filename` before the file is written under `UPLOAD_DIR`.

`MAX_CONTENT_LENGTH` is set in `core/flask_app.py`. The limit is `DIGITERRA_MAX_UPLOAD_MB` when that variable is a positive number, and 256 MB otherwise. An oversized request body is rejected before the file is read, and the handler in `core/errors.py` answers with a JSON 413.

`routes/upload.py` also displays a warning for files larger than `FILE_SIZE_WARNING_MB` (`DIGITERRA_WARN_UPLOAD_MB`, default 1 MB) and for high cell counts. The warning does not block the upload. A second, application-level size check against `DIGITERRA_MAX_UPLOAD_MB` runs only when `DIGITERRA_ENFORCE_UPLOAD_LIMIT` is true (`core/constants.py`).

The upload response returns the filename and the column names without HTML escaping, and the page inserts them with `innerHTML`. This is an open cross-site scripting issue for a shared deployment where one user can upload a file that another user views. The issue is not fixed on this branch.

## Downloads

`routes/assets.py` serves generated files from the session's output directory. The requested path is resolved with `Path.resolve()` and checked with `relative_to` against the resolved base, so a path outside the directory returns 404. Only regular files are served, and the download name passes through `secure_filename`.

## Sessions and generated files

Each browser session has a cookie named `digiterra_session` (`core/state.py`) whose value is 32 hex characters issued by the server. `core/flask_app.py` replaces any cookie value that does not have that form, so a crafted value cannot name another session's storage or output directory. The cookie is `HttpOnly` and `SameSite=Lax`, and `Secure` when the request is HTTPS or `DIGITERRA_COOKIE_SECURE` is set.

Generated files are written under a per-session subdirectory. `config.set_session_scope` in `python_scripts/config.py` sets a context variable that the request and the training thread both use, so one session's `predictions.csv` does not overwrite another's.

Session state is in process memory. `core/state.py` bounds the store to `DIGITERRA_MAX_SESSIONS` entries (default 32), evicts the oldest entry past that bound, and guards the store with a lock. Because the state is in one process, gunicorn must run with one worker.

## Progress streams

`GET /progress/<id>` in `routes/main.py` returns 404 for an id that has no tracker. Trackers are capped at 256 (`MAX_TRACKED_SESSIONS` in `python_scripts/preprocessing/progress_tracker.py`). A stream for a valid id stays open for up to `PROGRESS_STREAM_TIMEOUT_SECONDS` (3600), which on a single synchronous worker blocks other requests for that time.

## Error responses

`core/errors.py` registers handlers so `ApiError`, `RequestEntityTooLarge`, other `HTTPException` types, and unhandled exceptions return a JSON body with an `error` string. A 404 on a page request still returns the HTML error page. Stack traces go to the log, not the response.

## Desktop wrapper

`core/desktop.py` exposes `save_file` to page JavaScript through the pywebview `js_api`. The method resolves the requested name under the output directory and refuses a path that resolves outside that directory.

## Container

`deploy/docker/Dockerfile` builds from `python:3.11-slim`, creates an `app` user, and switches to that user before the `CMD`. Gunicorn binds `0.0.0.0:5000` with one worker.

## Open items

- HTML-escape the filename and column names in the upload response, and make `escapeHtml` in `static/js/core.js` escape quotes.
- Add CSRF protection, authentication, and rate limiting before a multi-user deployment.
- Put the app behind HTTPS and a reverse proxy for any network deployment.
