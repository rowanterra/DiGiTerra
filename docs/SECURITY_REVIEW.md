# Security posture

Reviewed against the `audit-fixes` branch on 2026-09-03. The statements below describe the code on that branch. DiGiTerra is a single-user desktop application that also runs as a web app behind gunicorn. There is no authentication, CSRF protection, or rate limiting. A multi-user deployment needs those before exposure to untrusted clients.

## Uploads

Only `.csv` files are accepted. `allowed_file` in `core/helpers.py` checks the extension, and `routes/upload.py` rejects anything else with a 400.

Uploaded filenames pass through `werkzeug.utils.secure_filename` before the file is written under `UPLOAD_DIR`.

Three limits apply in order. `MAX_CONTENT_LENGTH` in `core/flask_app.py` rejects an oversized request body before the file is read, and `core/errors.py` answers with a JSON 413; the value is `DIGITERRA_MAX_UPLOAD_MB` when that variable is positive and 32 MB otherwise. `read_csv_bounded` in `core/helpers.py` then reads the header to learn the column count and caps the rows it parses at `DIGITERRA_MAX_CELLS` divided by that width (default 5,000,000 cells), so a file over the budget is refused without the whole file being parsed into memory. The parsed frame is measured with `memory_usage(deep=True)` and refused above `DIGITERRA_MAX_FRAME_MB` (default 256), which catches object columns that hold far more bytes per cell than numeric ones. `tests/unit/test_limits.py` asserts the memory behavior by comparing the peak allocation of a rejected read against a full parse of the same file.

`desktop_app.py` sets a 1024 MB body limit and disables the cell and frame limits before the application module is imported, since a local user's own machine is not an attack surface.

The upload file on disk is removed once it has been read, on every exit path, in both `routes/upload.py` and `python_scripts/app_prediction.py`.

`routes/upload.py` also displays a warning for files larger than `FILE_SIZE_WARNING_MB` (`DIGITERRA_WARN_UPLOAD_MB`, default 1 MB) and for high cell counts. The warning does not block the upload. A second, application-level size check against `DIGITERRA_MAX_UPLOAD_MB` runs only when `DIGITERRA_ENFORCE_UPLOAD_LIMIT` is true (`core/constants.py`).

The upload response returns the filename after `secure_filename`, so it carries no markup. Column names are returned as data because the page posts them back for column selection; `escapeHtml` in `static/js/core.js` escapes `& < > " '` and every `innerHTML` sink that renders a column name or a server string passes through it. `tests/unit/test_limits.py` checks the filename, and `tests/integration/test_browser_advanced.py` loads the page in Chromium with no uncaught error.

## Downloads

`routes/assets.py` serves generated files from the session's output directory. The requested path is resolved with `Path.resolve()` and checked with `relative_to` against the resolved base, so a path outside the directory returns 404. Only regular files are served, and the download name passes through `secure_filename`.

## Sessions and generated files

Each browser session has a cookie named `digiterra_session` (`core/state.py`) whose value is 32 hex characters issued by the server. `core/flask_app.py` replaces any cookie value that does not have that form, so a crafted value cannot name another session's storage or output directory. The cookie is `HttpOnly` and `SameSite=Lax`, and `Secure` when the request is HTTPS or `DIGITERRA_COOKIE_SECURE` is set.

Generated files are written under a per-session subdirectory. `config.set_session_scope` in `python_scripts/config.py` sets a context variable that the request and the training thread both use, so one session's `predictions.csv` does not overwrite another's.

Session state is in process memory. `core/state.py` bounds the store to `DIGITERRA_MAX_SESSIONS` entries (default 32), evicts the oldest entry past that bound, and guards the store with a lock. Because the state is in one process, gunicorn must run with one worker. The bound is a count, not a byte total, so the worst case is the session count times the cell cap.

Session output directories older than `DIGITERRA_STALE_SESSION_HOURS` (default 24; 0 disables) and stray upload files of the same age are removed by `config.sweep_stale_outputs`. A directory is marked as in use whenever a path is built from it, and a session with a run in flight is skipped whatever its age, so a job longer than the age limit does not lose its output to an unrelated visitor's sweep. The sweep runs on a timer started in `core/flask_app.py` and also when a new session cookie is issued.

## Progress streams

`GET /progress/<id>` in `routes/main.py` returns 404 for an id that has no tracker and for an id that a different browser session started; each tracker records the session cookie that created it. Trackers are capped at 256 (`MAX_TRACKED_SESSIONS` in `python_scripts/preprocessing/progress_tracker.py`).

A stream occupies a worker thread while it is open. The Dockerfile runs gunicorn with one worker process and `--worker-class gthread --threads 8`, which keeps session state in one process while letting an open stream coexist with other requests; the default synchronous worker does not. `scripts/check_stream_starvation.py` measures both: recorded on 2026-09-03, an unrelated request was not answered within ten seconds under the synchronous worker and was answered in 0.01 s under the threaded worker. The thread count is the ceiling on concurrent requests including open streams. A stream ends after `DIGITERRA_PROGRESS_TIMEOUT_SECONDS` (900) without an error payload, so the browser's EventSource reconnects and a longer run still delivers its result.

## Training request bounds and admission control

`validate_training_request` in `python_scripts/app_model_training.py` runs in the request handler, before `routes/modeling.py` creates a thread, so a malformed or out-of-range request costs a JSON parse. It checks the required fields, that the named model is in the registry, and the ranges held in `core/constants.py`: `testSize` in [`DIGITERRA_MIN_TEST_SIZE`, `DIGITERRA_MAX_TEST_SIZE`] (0.05 to 0.90), cross-validation folds 2 to `DIGITERRA_MAX_CV_FOLDS` (20), search folds 2 to `DIGITERRA_MAX_SEARCH_CV_FOLDS` (10), and search iterations 1 to `DIGITERRA_MAX_SEARCH_ITERATIONS` (100). A value outside its range returns 400.

`core/jobs.py` limits how many runs exist at once: one per browser session, and `DIGITERRA_MAX_CONCURRENT_TRAINING` (default 2) across the process. A request past either limit returns 429 without starting a thread. The search and cross-validation paths use every core (`n_jobs=-1`), so this is the ceiling on the CPU one client can occupy.

There is no rate limiting, so a client can still send requests that are rejected cheaply, and no queue, so a refused run is not deferred. Those need an external limiter and a job queue.

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
