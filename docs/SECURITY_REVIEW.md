# Security posture

Reviewed against the `audit-fixes` branch on 2026-09-04, at the commit that added the bundle signature and the compressed-input preflight. The statements below describe the code at that commit; `tests/unit/test_docs.py` fails when a `DIGITERRA_*` variable is added to the code without appearing here or in `docs/documentation.md`, so a new limit cannot ship undocumented. DiGiTerra is a single-user desktop application that also runs as a web app behind gunicorn. There is no authentication, CSRF protection, or rate limiting. A multi-user deployment needs those before exposure to untrusted clients.

## Uploads

`.csv`, `.xlsx`, `.xlsm`, and `.parquet` files are accepted. `allowed_file` in `core/helpers.py` checks the extension, and `routes/upload.py` rejects anything else with a 400.

Uploaded filenames pass through `werkzeug.utils.secure_filename` before the file is written under `UPLOAD_DIR`.

Three limits apply in order. `MAX_CONTENT_LENGTH` in `core/flask_app.py` rejects an oversized request body before the file is read, and `core/errors.py` answers with a JSON 413; the value is `DIGITERRA_MAX_UPLOAD_MB` when that variable is positive and 32 MB otherwise. `read_table_bounded` in `core/helpers.py` then learns the column count from the header (CSV), the first row in read-only mode (Excel), or the file metadata (Parquet) and caps the rows it parses at `DIGITERRA_MAX_CELLS` divided by that width (default 5,000,000 cells), so a file over the budget is refused without the whole file being parsed into memory. For Parquet a row count in the metadata above the budget is refused with no data read. For Excel the zip directory is checked before openpyxl opens anything (see Compressed inputs below), and pandas then stops the row iterator at the cap. The parsed frame is measured with `memory_usage(deep=True)` and refused above `DIGITERRA_MAX_FRAME_MB` (default 256), which catches object columns that hold far more bytes per cell than numeric ones. `tests/unit/test_limits.py` asserts the memory behavior by comparing the peak allocation of a rejected read against a full parse of the same file.

`desktop_app.py` sets limits at the scale of one machine before the application module is imported: `DIGITERRA_MAX_UPLOAD_MB=1024`, `DIGITERRA_MAX_CELLS=200000000`, `DIGITERRA_MAX_FRAME_MB=4096`, and `DIGITERRA_MAX_TOTAL_FRAME_MB=0`, no total across sessions because there is one session. It no longer disables them. It also enables bundle import and the data-check override, both described below.

The upload file on disk is removed once it has been read, on every exit path, in both `routes/upload.py` and `python_scripts/app_prediction.py`.

`routes/upload.py` also displays a warning for files larger than `FILE_SIZE_WARNING_MB` (`DIGITERRA_WARN_UPLOAD_MB`, default 1 MB) and for high cell counts. The warning does not block the upload. A second, application-level size check against `DIGITERRA_MAX_UPLOAD_MB` runs only when `DIGITERRA_ENFORCE_UPLOAD_LIMIT` is true (`core/constants.py`).

The upload response returns the filename after `secure_filename`, so it carries no markup. Column names are returned as data because the page posts them back for column selection; `escapeHtml` in `static/js/core.js` escapes `& < > " '` and every `innerHTML` sink that renders a column name or a server string passes through it. `tests/unit/test_limits.py` checks the filename, and `tests/integration/test_browser_advanced.py` loads the page in Chromium with no uncaught error.

## Downloads

`routes/assets.py` serves generated files from the session's output directory. The requested path is resolved with `Path.resolve()` and checked with `relative_to` against the resolved base, so a path outside the directory returns 404. Only regular files are served, and the download name passes through `secure_filename`.

## Sessions and generated files

Each browser session has a cookie named `digiterra_session` (`core/state.py`) whose value is 32 hex characters issued by the server. `core/flask_app.py` replaces any cookie value that does not have that form, so a crafted value cannot name another session's storage or output directory. The cookie is `HttpOnly` and `SameSite=Lax`, and `Secure` when the request is HTTPS or `DIGITERRA_COOKIE_SECURE` is set.

Generated files are written under a per-session subdirectory. `config.set_session_scope` in `python_scripts/config.py` sets a context variable that the request and the training thread both use, so one session's `predictions.csv` does not overwrite another's.

Session state is in process memory. `core/state.py` bounds the store to `DIGITERRA_MAX_SESSIONS` entries (default 32), guards it with a lock, and also holds a byte total; the Session memory section below describes it. Because the state is in one process, gunicorn must run with one worker.

Session output directories older than `DIGITERRA_STALE_SESSION_HOURS` (default 24; 0 disables) and stray upload files of the same age are removed by `config.sweep_stale_outputs`. A directory is marked as in use whenever a path is built from it, and a session with a run in flight is skipped whatever its age, so a job longer than the age limit does not lose its output to an unrelated visitor's sweep. The sweep runs on a timer started in `core/flask_app.py` and also when a new session cookie is issued.

## Progress streams

`GET /progress/<id>` in `routes/main.py` returns 404 for an id that has no tracker and for an id that a different browser session started; each tracker records the session cookie that created it. Trackers are capped at 256 (`MAX_TRACKED_SESSIONS` in `python_scripts/preprocessing/progress_tracker.py`).

A stream occupies a worker thread while it is open. The Dockerfile runs gunicorn with one worker process and `--worker-class gthread --threads 8`, which keeps session state in one process while letting an open stream coexist with other requests; the default synchronous worker does not. `scripts/check_stream_starvation.py` measures both: recorded on 2026-09-03, an unrelated request was not answered within ten seconds under the synchronous worker and was answered in 0.01 s under the threaded worker. A stream ends after `DIGITERRA_PROGRESS_TIMEOUT_SECONDS` (900) without an error payload, so the browser's EventSource reconnects and a longer run still delivers its result.

Streams take a slot from `core/jobs.py`: one per training id, and `DIGITERRA_MAX_CONCURRENT_STREAMS` (default 4, half the thread pool) across the process. A request past either limit returns 429, and the slot is released when the generator ends or the response closes, so a client that never reads the body does not keep it. `tests/unit/test_capacity.py` opens two streams under a cap of two, checks that a third and a duplicate are refused while an unrelated request answers, and that closing one frees its slot.

## Training request bounds and admission control

`validate_training_request` in `python_scripts/app_model_training.py` runs in the request handler, before `routes/modeling.py` creates a thread, so a malformed or out-of-range request costs a JSON parse. It checks the required fields, that the named model is in the registry, and the ranges held in `core/constants.py`: `testSize` in [`DIGITERRA_MIN_TEST_SIZE`, `DIGITERRA_MAX_TEST_SIZE`] (0.05 to 0.90), cross-validation folds 2 to `DIGITERRA_MAX_CV_FOLDS` (20), search folds 2 to `DIGITERRA_MAX_SEARCH_CV_FOLDS` (10), and search iterations 1 to `DIGITERRA_MAX_SEARCH_ITERATIONS` (100). A value outside its range returns 400.

`core/jobs.py` limits how many runs exist at once: one per browser session, and `DIGITERRA_MAX_CONCURRENT_TRAINING` (default 2) across the process. A request past either limit returns 429 without starting a thread. Hyperparameter searches, feature-selection forests, and the finalist evaluation ask for `job_n_jobs()` cores: `DIGITERRA_JOB_CORES` when set, otherwise `cpu_count // DIGITERRA_MAX_CONCURRENT_TRAINING`, so the concurrent runs together ask for the host once. Estimators that take `n_jobs` from the form keep the form's value.

Data checks run before dispatch (`python_scripts/preprocessing/guardrails.py`). A blocking finding raises before any estimator is fit. The override (`allowUnsafe`) is a server-side switch, `DIGITERRA_ALLOW_UNSAFE_OVERRIDE`, off by default and on in the desktop launcher; a request that asks for it on a deployment without the switch is refused in `validate_training_request`, and one that asks for it without an `allowUnsafeReason` of at least 10 characters is refused the same way. The control is rendered only when the switch is on. A run trained past a blocking finding carries `decision_use: "not valid for decision use"` and the reason in the result, the manifest, and the model card, and the card's title says so.

## Session memory

`core/state.py` holds at most `DIGITERRA_MAX_SESSIONS` (32) session dictionaries and, with each, the `memory_usage(deep=True)` size of its frame and the serialized size of its fitted objects (measured by a counting joblib dump after each run). An upload is admitted only when the total across sessions stays within `DIGITERRA_MAX_TOTAL_FRAME_MB` (default 1024, 0 disables; the desktop launcher disables it). The check runs twice: before the parse against the decoded size taken from metadata (`core.helpers.estimate_decoded_bytes`: file size for CSV, the sum of zip members' inflated sizes for Excel, the sum of row-group byte sizes for Parquet), and after the parse against the measured frame. The oldest sessions without a run in flight are evicted first; when only sessions with running jobs remain, the upload is refused with 429. The session-count eviction also skips sessions with a run in flight.

After a run the fitted objects are serialized to a counting sink to measure them and charged with `admit_model`, which counts the model beside the frame it was trained on, evicts idle sessions, and raises when the pair does not fit (a measurement failure is treated the same way); the caller then drops the fitted objects and the result carries a note that inference is unavailable for that run, so an over-budget model is not retained. Measuring after the fit is weaker than a memory limit on the process that does the fit. The budget covers frames and fitted objects, not transient allocations inside a parse or a training run, and the process is not isolated: two runs at their core budget with large intermediates can exceed it. Parsing and training in a separate worker under a memory and time limit is the remaining control and is not done; the container memory limit is the outer bound.

## Compressed inputs

A 32 MB request body says little about an Excel or Parquet file, which can inflate far past its size on disk. Before a workbook is opened, `core/helpers.py` reads the zip directory and refuses more than 10,000 members, any member over 1 MB that inflates more than 200 times, and a total inflated size over eight times `DIGITERRA_MAX_FRAME_MB`; only then does openpyxl open it. Before a Parquet file is decoded, the footer's row-group byte sizes are summed against the same budget, and within budget the rows are read in batches of at most 65,536 up to the row cap, so a single oversized row group is not materialized whole. `tests/unit/test_formats.py` builds a workbook under 1 MB on disk with a 40 MB comment inside and a Parquet file that decodes to 9.6 MB, and measures with tracemalloc that both are refused below 8 MB and 2 MB of peak allocation.

The desktop launcher no longer disables the limits; it sets them at the scale of one machine (1 GB body, 200 million cells, 4 GB frame).

## Run records and model bundles

Each run writes `manifest.json`, `model_card.json`, and `model_card.html` into the session directory (`python_scripts/reporting/bundle.py`). The card is generated with every value HTML-escaped and no script. `GET /export-bundle` zips those with the fitted objects serialized by joblib and records the SHA-256 of that artifact in the manifest.

A bundle is a zip, and the signature can only be checked after `model.joblib` has been read, so the zip is bounded first: `core.helpers.preflight_zip` refuses more than 32 members, any name outside the six a bundle may hold, a member over `DIGITERRA_MAX_BUNDLE_MEMBER_MB` (512), a total over `DIGITERRA_MAX_BUNDLE_TOTAL_MB` (1024), or an inflation ratio over 200, all from the central directory. The directory is data from the file and can lie, so `read_zip_member_bounded` streams each member and stops at the same cap. The manifest is read that way as well, capped at 4 MB. The workbook reader uses the same preflight.

`POST /import-bundle` loads a bundle's objects into the session for `/predict`. Loading unpickles the objects, and unpickling runs code from the file. A hash in the manifest proves nothing about who wrote the file, so every exported artifact is signed with HMAC-SHA256 under a key that belongs to the installation: `DIGITERRA_BUNDLE_KEY` when set, otherwise `bundle_signing.key` generated once (32 random bytes, owner-only permissions) in the app support directory. Import verifies the signature before anything is unpickled and refuses a bundle that is unsigned, altered, or signed by another installation, naming both key ids in the message. Two installations that set the same `DIGITERRA_BUNDLE_KEY` can exchange bundles; that is the trust decision, made by the people who share the key. The route also answers 403 unless `DIGITERRA_ALLOW_BUNDLE_IMPORT` is set; the desktop launcher sets it, the container and Helm defaults do not. `tests/unit/test_bundle.py` exports under one key and shows the import refused under another and with the signature removed. `scripts/score_bundle.py` checks the same signature and has `--trust-unsigned` for a file whose origin the operator has verified another way; `scripts/rerun_bundle.py` reads only the manifest and unpickles nothing.

The fitted objects are still pickles. A signature stops a stranger's bundle from being loaded; it does not make the format safe against a signer who is compromised. A pickle-free model format is the remaining step and is not done.

There is no rate limiting, so a client can still send requests that are rejected cheaply, and no queue, so a refused run is not deferred. Those need an external limiter and a job queue.

## Error responses

`core/errors.py` registers handlers so `ApiError`, `RequestEntityTooLarge`, other `HTTPException` types, and unhandled exceptions return a JSON body with an `error` string. A 404 on a page request still returns the HTML error page. Stack traces go to the log, not the response.

## Desktop wrapper

`core/desktop.py` exposes `save_file` to page JavaScript through the pywebview `js_api`. The method resolves the requested name under the output directory and refuses a path that resolves outside that directory.

## Container

`deploy/docker/Dockerfile` builds from `python:3.11-slim`, creates an `app` user, and switches to that user before the `CMD`. Gunicorn binds `0.0.0.0:5000` with one worker.

## Open items

- Add CSRF protection, authentication, and rate limiting before a multi-user deployment.
- Put the app behind HTTPS and a reverse proxy for any network deployment.
