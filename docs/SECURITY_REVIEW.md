# Security posture

Reviewed against the `audit-fixes` branch on 2026-09-03. The statements below describe the code on that branch. DiGiTerra is a single-user desktop application that also runs as a web app behind gunicorn. There is no authentication, CSRF protection, or rate limiting. A multi-user deployment needs those before exposure to untrusted clients.

## Uploads

`.csv`, `.xlsx`, `.xlsm`, and `.parquet` files are accepted. `allowed_file` in `core/helpers.py` checks the extension, and `routes/upload.py` rejects anything else with a 400.

Uploaded filenames pass through `werkzeug.utils.secure_filename` before the file is written under `UPLOAD_DIR`.

Three limits apply in order. `MAX_CONTENT_LENGTH` in `core/flask_app.py` rejects an oversized request body before the file is read, and `core/errors.py` answers with a JSON 413; the value is `DIGITERRA_MAX_UPLOAD_MB` when that variable is positive and 32 MB otherwise. `read_table_bounded` in `core/helpers.py` then learns the column count from the header (CSV), the first row in read-only mode (Excel), or the file metadata (Parquet) and caps the rows it parses at `DIGITERRA_MAX_CELLS` divided by that width (default 5,000,000 cells), so a file over the budget is refused without the whole file being parsed into memory. For Parquet a row count in the metadata above the budget is refused with no data read. For Excel, pandas stops the row iterator at the cap, but openpyxl parses the workbook's shared-strings table in full first, so the bound is on rows, not on a string-heavy workbook's string table. The parsed frame is measured with `memory_usage(deep=True)` and refused above `DIGITERRA_MAX_FRAME_MB` (default 256), which catches object columns that hold far more bytes per cell than numeric ones. `tests/unit/test_limits.py` asserts the memory behavior by comparing the peak allocation of a rejected read against a full parse of the same file.

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

A stream occupies a worker thread while it is open. The Dockerfile runs gunicorn with one worker process and `--worker-class gthread --threads 8`, which keeps session state in one process while letting an open stream coexist with other requests; the default synchronous worker does not. `scripts/check_stream_starvation.py` measures both: recorded on 2026-09-03, an unrelated request was not answered within ten seconds under the synchronous worker and was answered in 0.01 s under the threaded worker. A stream ends after `DIGITERRA_PROGRESS_TIMEOUT_SECONDS` (900) without an error payload, so the browser's EventSource reconnects and a longer run still delivers its result.

Streams take a slot from `core/jobs.py`: one per training id, and `DIGITERRA_MAX_CONCURRENT_STREAMS` (default 4, half the thread pool) across the process. A request past either limit returns 429, and the slot is released when the generator ends or the response closes, so a client that never reads the body does not hold it. `tests/unit/test_capacity.py` opens two streams under a cap of two, checks that a third and a duplicate are refused while an unrelated request answers, and that closing one frees its slot.

## Training request bounds and admission control

`validate_training_request` in `python_scripts/app_model_training.py` runs in the request handler, before `routes/modeling.py` creates a thread, so a malformed or out-of-range request costs a JSON parse. It checks the required fields, that the named model is in the registry, and the ranges held in `core/constants.py`: `testSize` in [`DIGITERRA_MIN_TEST_SIZE`, `DIGITERRA_MAX_TEST_SIZE`] (0.05 to 0.90), cross-validation folds 2 to `DIGITERRA_MAX_CV_FOLDS` (20), search folds 2 to `DIGITERRA_MAX_SEARCH_CV_FOLDS` (10), and search iterations 1 to `DIGITERRA_MAX_SEARCH_ITERATIONS` (100). A value outside its range returns 400.

`core/jobs.py` limits how many runs exist at once: one per browser session, and `DIGITERRA_MAX_CONCURRENT_TRAINING` (default 2) across the process. A request past either limit returns 429 without starting a thread. Hyperparameter searches, feature-selection forests, and the finalist evaluation ask for `job_n_jobs()` cores: `DIGITERRA_JOB_CORES` when set, otherwise `cpu_count // DIGITERRA_MAX_CONCURRENT_TRAINING`, so the concurrent runs together ask for the host once. Estimators that take `n_jobs` from the form keep the form's value.

Data checks run before dispatch (`python_scripts/preprocessing/guardrails.py`). A blocking finding raises before any estimator is fit unless the request sets `allowUnsafe`; the override is recorded in the result and the run record.

## Session memory

`core/state.py` holds at most `DIGITERRA_MAX_SESSIONS` (32) session dictionaries and, with each, the `memory_usage(deep=True)` size of its frame. An upload is admitted only when the total across sessions stays within `DIGITERRA_MAX_TOTAL_FRAME_MB` (default 1024, 0 disables; the desktop launcher disables it). The oldest sessions without a run in flight are evicted first; when only sessions with running jobs remain, the upload is refused with 429. The session-count eviction also skips sessions with a run in flight. Without this budget, a client rotating cookies could hold `MAX_SESSIONS` frames at the per-frame cap.

## Run records and model bundles

Each run writes `manifest.json`, `model_card.json`, and `model_card.html` into the session directory (`python_scripts/reporting/bundle.py`). The card is generated with every value HTML-escaped and no script. `GET /export-bundle` zips those with the fitted objects serialized by joblib and records the SHA-256 of that artifact in the manifest.

`POST /import-bundle` loads a bundle's objects into the session for `/predict`. Loading unpickles the objects, and unpickling runs code from the file, so the route answers 403 unless `DIGITERRA_ALLOW_BUNDLE_IMPORT` is set. The desktop launcher sets it; the container and Helm defaults do not. When enabled, the route checks the bundle format string and the artifact hash before unpickling, which detects alteration of a bundle DiGiTerra wrote but does not make an untrusted bundle safe. `scripts/score_bundle.py` and `scripts/rerun_bundle.py` unpickle without the flag and are for bundles from a known source.

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
