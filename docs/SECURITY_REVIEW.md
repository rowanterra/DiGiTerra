# Security posture

Reviewed against the `audit-fixes` branch on 2026-09-04, at the commit that added the deployment profile. The statements below describe the code at that commit; `tests/unit/test_docs.py` fails when a `DIGITERRA_*` variable is added to the code without appearing here or in `docs/documentation.md`, so a new limit cannot reach a release undocumented. DiGiTerra is a single-user desktop application that also runs as a web app behind gunicorn, and the two need opposite defaults. `DIGITERRA_PROFILE` picks a set rather than leaving an operator to set each control correctly on its own.

## Deployment profile

`desktop` (the default, and what `desktop_app.py` runs) asks for no identity, no CSRF token, and applies no rate limit, because the only user is the person at the keyboard. `server` requires all three. Each control also answers to its own variable, so a pilot can turn one on while its proxy is being arranged: `DIGITERRA_REQUIRE_AUTH`, `DIGITERRA_REQUIRE_CSRF`, `DIGITERRA_RATE_LIMIT`.

Identity is delegated, not implemented. `core/security.py` reads the name in `DIGITERRA_AUTH_HEADER` (default `X-Forwarded-User`) only when the connection comes from an address in `DIGITERRA_TRUSTED_PROXIES`, which is the arrangement an EDX or comparable environment provides by terminating single sign-on at its proxy; a header from any other peer is ignored, since anyone who can reach the port can set one. `DIGITERRA_SHARED_SECRET` accepts a constant-time match on `X-DiGiTerra-Token` for a deployment with no such proxy. Neither gives per-user sessions, a login page, logout, or a record of who did what: the session cookie still identifies the browser rather than the person, so two people behind one proxy identity share a session's data. An account system remains to be built.

CSRF uses a double-submit token: the server sets `digiterra_csrf`, readable by script on purpose, and the page echoes it in `X-DiGiTerra-CSRF` on every unsafe method. `static/js/core.js` wraps `fetch` once so no call site has to remember it. A cross-site form can cause the cookie to be sent but cannot read it to set the header.

The rate limit is a token bucket per identity, or per address when there is none: `DIGITERRA_RATE_PER_MINUTE` (120) with `DIGITERRA_RATE_BURST` (40). `/process`, `/predict`, and `/upload` take ten tokens each, the rest take one, so a run costs what a run costs. The buckets live in this process, which is where the session state already lives; a deployment across more than one process needs a shared store for both.

## Uploads

`.csv`, `.xlsx`, `.xlsm`, and `.parquet` files are accepted. `allowed_file` in `core/helpers.py` checks the extension, and `routes/upload.py` rejects anything else with a 400.

Uploaded filenames pass through `werkzeug.utils.secure_filename` before the file is written under `UPLOAD_DIR`.

Three limits run in order. `MAX_CONTENT_LENGTH` in `core/flask_app.py` rejects an oversized request body before the file is read, and `core/errors.py` answers with a JSON 413; the value is `DIGITERRA_MAX_UPLOAD_MB` when that variable is positive and 32 MB otherwise. `read_table_bounded` in `core/helpers.py` then learns the column count from the header (CSV), the first row in read-only mode (Excel), or the file metadata (Parquet) and caps the rows it parses at `DIGITERRA_MAX_CELLS` divided by that width (default 5,000,000 cells), so a file over the budget is refused without the whole file being parsed into memory. For Parquet a row count in the metadata above the budget is refused with no data read. For Excel the zip directory is checked before openpyxl opens anything (see Compressed inputs below), and pandas then stops the row iterator at the cap. The parsed frame is measured with `memory_usage(deep=True)` and refused above `DIGITERRA_MAX_FRAME_MB` (default 256), which catches object columns that take far more bytes per cell than numeric ones. `tests/unit/test_limits.py` asserts the memory behavior by comparing the peak allocation of a rejected read against a full parse of the same file.

`desktop_app.py` sets limits at the scale of one machine before the application module is imported: `DIGITERRA_MAX_UPLOAD_MB=1024`, `DIGITERRA_MAX_CELLS=200000000`, `DIGITERRA_MAX_FRAME_MB=4096`, and `DIGITERRA_MAX_TOTAL_FRAME_MB=0`, no total across sessions because there is one session. It no longer disables them. It also enables bundle import and the data-check override, both described below.

The upload file on disk is removed once it has been read, on every exit path, in both `routes/upload.py` and `python_scripts/app_prediction.py`.

`routes/upload.py` also displays a warning for files larger than `FILE_SIZE_WARNING_MB` (`DIGITERRA_WARN_UPLOAD_MB`, default 1 MB) and for high cell counts. The warning does not block the upload. A second, application-level size check against `DIGITERRA_MAX_UPLOAD_MB` runs only when `DIGITERRA_ENFORCE_UPLOAD_LIMIT` is true (`core/constants.py`).

The upload response returns the filename after `secure_filename`, so it contains no markup. Column names are returned as data because the page posts them back for column selection; `escapeHtml` in `static/js/core.js` escapes `& < > " '` and every `innerHTML` sink that renders a column name or a server string passes through it. `tests/unit/test_limits.py` checks the filename, and `tests/integration/test_browser_advanced.py` loads the page in Chromium with no uncaught error.

## Downloads

`routes/assets.py` serves generated files from the session's output directory. The requested path is resolved with `Path.resolve()` and checked with `relative_to` against the resolved base, so a path outside the directory returns 404. Only regular files are served, and the download name passes through `secure_filename`.

## Sessions and generated files

Each browser session has a cookie named `digiterra_session` (`core/state.py`) whose value is 32 hex characters issued by the server. `core/flask_app.py` replaces any cookie value that does not have that form, so a crafted value cannot name another session's storage or output directory. The cookie is `HttpOnly` and `SameSite=Lax`, and `Secure` when the request is HTTPS or `DIGITERRA_COOKIE_SECURE` is set.

Generated files are written under a per-session subdirectory. `config.set_session_scope` in `python_scripts/config.py` sets a context variable that the request and the training thread both use, so one session's `predictions.csv` does not overwrite another's.

Session state is in process memory. `core/state.py` bounds the store to `DIGITERRA_MAX_SESSIONS` entries (default 32), guards it with a lock, and also records a byte total; the Session memory section below describes it. Because the state is in one process, gunicorn must run with one worker.

Session output directories older than `DIGITERRA_STALE_SESSION_HOURS` (default 24; 0 disables) and stray upload files of the same age are removed by `config.sweep_stale_outputs`. A directory is marked as in use whenever a path is built from it, and a session with a run in flight is skipped whatever its age, so a job longer than the age limit does not lose its output to an unrelated visitor's sweep. The sweep runs on a timer started in `core/flask_app.py` and also when a new session cookie is issued.

## Progress streams

`GET /progress/<id>` in `routes/main.py` returns 404 for an id that has no tracker and for an id that a different browser session started; each tracker records the session cookie that created it. Trackers are capped at 256 (`MAX_TRACKED_SESSIONS` in `python_scripts/preprocessing/progress_tracker.py`).

A stream occupies a worker thread while it is open. The Dockerfile runs gunicorn with one worker process and `--worker-class gthread --threads 8`, which keeps session state in one process while letting an open stream coexist with other requests; the default synchronous worker does not. `scripts/check_stream_starvation.py` measures both: recorded on 2026-09-03, an unrelated request was not answered within ten seconds under the synchronous worker and was answered in 0.01 s under the threaded worker. A stream ends after `DIGITERRA_PROGRESS_TIMEOUT_SECONDS` (900) without an error payload, so the browser's EventSource reconnects and a longer run still delivers its result.

Streams take a slot from `core/jobs.py`: one per training id, and `DIGITERRA_MAX_CONCURRENT_STREAMS` (default 4, half the thread pool) across the process. A request past either limit returns 429, and the slot is released when the generator ends or the response closes, so a client that never reads the body does not keep it. `tests/unit/test_capacity.py` opens two streams under a cap of two, checks that a third and a duplicate are refused while an unrelated request answers, and that closing one frees its slot.

## Training request bounds and admission control

`validate_training_request` in `python_scripts/app_model_training.py` runs in the request handler, before `routes/modeling.py` creates a thread, so a malformed or out-of-range request costs a JSON parse. It checks the required fields, that the named model is in the registry, and the ranges set in `core/constants.py`: `testSize` in [`DIGITERRA_MIN_TEST_SIZE`, `DIGITERRA_MAX_TEST_SIZE`] (0.05 to 0.90), cross-validation folds 2 to `DIGITERRA_MAX_CV_FOLDS` (20), search folds 2 to `DIGITERRA_MAX_SEARCH_CV_FOLDS` (10), and search iterations 1 to `DIGITERRA_MAX_SEARCH_ITERATIONS` (100). A value outside its range returns 400.

`core/jobs.py` limits how many runs exist at once: one per browser session, and `DIGITERRA_MAX_CONCURRENT_TRAINING` (default 2) across the process. A request past either limit returns 429 without starting a thread. Hyperparameter searches, feature-selection forests, and the finalist evaluation ask for `job_n_jobs()` cores: `DIGITERRA_JOB_CORES` when set, otherwise `cpu_count // DIGITERRA_MAX_CONCURRENT_TRAINING`, so the concurrent runs together ask for the host once. Estimators that take `n_jobs` from the form keep the form's value.

Data checks run before dispatch (`python_scripts/preprocessing/guardrails.py`). A blocking finding raises before any estimator is fit. The override (`allowUnsafe`) is a server-side switch, `DIGITERRA_ALLOW_UNSAFE_OVERRIDE`, off by default and on in the desktop launcher; a request that asks for it on a deployment without the switch is refused in `validate_training_request`, and one that asks for it without an `allowUnsafeReason` of at least 10 characters is refused the same way. The control is rendered only when the switch is on. A run trained past a blocking finding records `decision_use: "not valid for decision use"` and the reason in the result, the manifest, and the model card, and the card's title says so.

## Session memory

`core/state.py` keeps at most `DIGITERRA_MAX_SESSIONS` (32) session dictionaries and, with each, the `memory_usage(deep=True)` size of its frame and the serialized size of its fitted objects (measured by a counting joblib dump after each run). An upload is admitted only when the total across sessions stays within `DIGITERRA_MAX_TOTAL_FRAME_MB` (default 1024, 0 disables; the desktop launcher disables it). The check runs twice: before the parse against the decoded size taken from metadata (`core.helpers.estimate_decoded_bytes`: file size for CSV, the sum of zip members' inflated sizes for Excel, the sum of row-group byte sizes for Parquet), and after the parse against the measured frame. The oldest sessions without a run in flight are evicted first; when only sessions with running jobs remain, the upload is refused with 429. The session-count eviction also skips sessions with a run in flight.

After a run the fitted objects are serialized to a counting sink to measure them and charged with `admit_model`, which counts the model beside the frame it was trained on, evicts idle sessions, and raises when the pair does not fit (a measurement failure is treated the same way); the caller then drops the fitted objects and the result states a note that inference is unavailable for that run, so an over-budget model is not retained. Measuring after the fit is weaker than a memory limit on the process that does the fit. The budget covers frames and fitted objects, not transient allocations inside a parse or a training run, and the process is not isolated: two runs at their core budget with large intermediates can exceed it. Parsing and training in a separate worker under a memory and time limit is the remaining control and is not done; the container memory limit is the outer bound.

## Compressed inputs

A 32 MB request body says little about an Excel or Parquet file, which can inflate far past its size on disk. Before a workbook is opened, `core/helpers.py` reads the zip directory and refuses more than 10,000 members, any member over 1 MB that inflates more than 200 times, and a total inflated size over eight times `DIGITERRA_MAX_FRAME_MB`; only then does openpyxl open it. Before a Parquet file is decoded, the footer's row-group byte sizes are summed against the same budget, and within budget the rows are read in batches of at most 65,536 up to the row cap, so a single oversized row group is not materialized whole. `tests/unit/test_formats.py` builds a workbook under 1 MB on disk with a 40 MB comment inside and a Parquet file that decodes to 9.6 MB, and measures with tracemalloc that both are refused below 8 MB and 2 MB of peak allocation.

The desktop launcher no longer disables the limits; it sets them at the scale of one machine (1 GB body, 200 million cells, 4 GB frame).

## Waiting runs

Admission control refuses a run when the slots are full. On a desktop that is right, because the person can wait and try again. On a shared installation it loses the request: nothing remembers that someone asked. `core/queue.py` records it instead, in SQLite under the application support directory, which gives the atomic claim a queue needs in one statement and survives the process that wrote it. A deployment across several machines would need a server rather than a file, and this is not that.

A job records the request, not the data. the frame is in the session store, which is where a worker in another process reads it. `POST /process` answers 202 with `status: queued` and a position when the slots are full and `DIGITERRA_QUEUE` is on, and 429 as before when it is not. `GET /queue/<id>` tells a page where its run stands, and answers only the session that started it. A slot coming free starts the oldest waiting run. A job whose worker stopped without finishing is returned to the queue after `DIGITERRA_QUEUE_STALE_SECONDS` (7200), which is what makes a restart recoverable rather than a loss, and `DIGITERRA_MAX_QUEUE_DEPTH` (50) bounds how many may wait.

`tests/unit/test_queue.py` races eight threads for one job and requires exactly one winner, stops and restarts the queue in a separate interpreter, and calls the route to see a full slot produce a queued run rather than a refusal.

## Where a session lives

`core/session_store.py` puts session contents behind two backends. `memory`, the desktop default, is the dictionary this application has always used. `disk`, the server default, writes one directory per session under the application support directory: the frame as parquet, the fitted objects through the artifact format, everything else as JSON, each written to a temporary file and renamed into place so a reader sees one version or the other and never half of either. A session id that is not the 32 hex characters this application issues is refused rather than used to build a path. `DIGITERRA_SESSION_BACKEND` overrides the profile.

The mapping a caller receives is the same either way: it records that it has changed, and `core.state.flush` writes it at the end of a request and at the end of a run rather than on every key, which would rewrite a parquet file for a one-word change. `tests/unit/test_session_store.py` starts a second interpreter and reads a session the first one wrote, which is the only way that claim can be checked.

The rate-limit buckets are still per process. Two workers therefore each allow the configured rate, so a deployment behind more than one should set the limit accordingly until the buckets move to the same store.

## Isolated work

A budget enforced inside the process doing the allocating cannot bound that process. by the time a frame or a fitted model is large enough to measure, the memory is already taken. `core/worker.py` runs a step in a child process started with the spawn method, under `RLIMIT_AS` and a wall clock, and the two ends exchange data through a file rather than a pipe, which would deadlock on a result larger than its buffer. An allocation past the limit raises `MemoryError` in the child or has it killed; either way the parent gets a refusal and its own memory is untouched. This is true only where the kernel applies `RLIMIT_AS`. Linux and the BSDs do; macOS does not. The next section says what happens elsewhere.

`RLIMIT_AS` counts address space, not resident pages, and a Python child spends several hundred megabytes of it on the interpreter, numpy, pandas and scikit-learn before it touches any data; measured here, about 512 MB before a small CSV is read. A step's limit is therefore `DIGITERRA_WORKER_BASE_MB` (768) plus what its data may take, and a limit below that baseline would refuse every step whatever its size.

Parsing goes through the worker under the server profile, or wherever `DIGITERRA_ISOLATE_WORK` says so; on the desktop it does not, since the machine belongs to the person using it. Training does not yet, because the fitted objects have to come back across the process boundary and that path is being built on the artifact format below.

### Where the memory bound holds

`RLIMIT_AS` is enforced by the Linux kernel and by the BSDs. It is not enforced on macOS: there `RLIMIT_AS` is a second name for `RLIMIT_RSS`, which the Darwin kernel does not apply. Measured on macOS 26 (Darwin 25.6, arm64), `setrlimit` returns `EINVAL` for every value below roughly 512 GiB, which is every budget a step would ask for, and the astronomical values above that which it does accept bound nothing. Windows has no equivalent and no `resource` module at all.

**The server profile therefore requires a platform whose kernel applies `RLIMIT_AS`, which is currently Linux and the BSDs.** On macOS, and on any platform not known to enforce the limit, `core/worker.py` refuses: `run_isolated` raises `WorkerLimitUnavailable` before a child process is started, and the message names the platform and the reason. A server-profile upload on macOS is answered with a 500 saying this installation cannot run an upload within a memory limit, rather than running the parse with no ceiling and reporting success. The refusal is checked in the parent, which is where it saves the work, and again in the child, which is the process that would do the allocating; a step that asks for a ceiling of zero is opting out deliberately and still runs.

This is the same rule the rest of the application follows: a fitted model whose size cannot be measured is discarded rather than kept uncharged (see Session memory), and a bundle whose signature cannot be verified is refused rather than imported. A limit that cannot be enforced is reported as absent, not assumed.

The desktop profile does not isolate work at all, so nothing about it changes on macOS: the machine belongs to the person using it, and `desktop_app.py` is unaffected.

`tests/unit/test_worker.py` asserts the outcome each platform actually offers rather than skipping on the one that cannot make the promise. Where the ceiling holds, an allocation of four gigabytes under a 768 MB budget stops the child while the parent survives and the next step still runs. Where it cannot be set, the same call is refused before any child starts. One test spawns a child that asks for the real budget and reports whether the kernel gave it, so the platform reading in the code cannot drift from what the allocating process finds; another simulates an unenforceable limit on any platform and checks that no child is created. The CI matrix runs `ubuntu-latest` and `macos-latest`, because an ubuntu-only matrix is what let the macOS behavior go untested.

## Run records and model bundles

Each run writes `manifest.json`, `model_card.json`, and `model_card.html` into the session directory (`python_scripts/reporting/bundle.py`). The card is generated with every value HTML-escaped and no script. `GET /export-bundle` zips those with the fitted objects serialized by joblib and records the SHA-256 of that artifact in the manifest.

A bundle is a zip, and the signature can only be checked after `model.joblib` has been read, so the zip is measured first: `core.helpers.preflight_zip` refuses more than 32 members, any name outside the six a bundle may contain, a member over `DIGITERRA_MAX_BUNDLE_MEMBER_MB` (512), a total over `DIGITERRA_MAX_BUNDLE_TOTAL_MB` (1024), or an inflation ratio over 200, all from the central directory. The directory is data from the file and can lie, so `read_zip_member_bounded` streams each member and stops at the same cap. The manifest is read that way as well, capped at 4 MB. The workbook reader uses the same preflight.

The artifact is written by `python_scripts/reporting/artifact.py` with `skops`, which writes the fitted objects as data with their types named and builds only the types the caller allows. The allowlist is the scikit-learn, numpy, scipy and pandas trees plus two names of this project's own: `apply_group_map`, which a fitted preprocessor refers to, and the RadiusNeighbors fallback estimator. A file naming anything else is refused before a single object is built, and the message names what it asked for. `scripts/check_artifact_roundtrip.py` trains every registered estimator, writes and reads its objects, and compares predictions, or labels for the clustering models that have no `predict`; `docs/artifact_roundtrip_2026-09-04.json` records the run: 67 estimators trained, 66 round-tripped with identical predictions, and one, `Birch`, cannot be written at all. Each row records the dataset and its SHA-256, the seed, the estimator class, the SHA-256 of the artifact, and the first eight predictions before and after the round trip at full precision, compared exactly rather than within a tolerance, so a reader can rerun one model and check the numbers rather than take the word "ok" for it. It found two things worth stating. The allowlist named the RadiusNeighbors fallback class wrongly, so that model's artifact would have been refused on load; the name is corrected, and the sweep is what would catch it again. And `Birch` cannot be written by skops at all, so a run of that model reports plainly that it cannot be exported as a bundle rather than falling back to a pickle; its results, model card and manifest are unaffected.

Bundles in the older format contain a joblib pickle instead. Reading one runs code from the file, so it is refused unless `DIGITERRA_ALLOW_LEGACY_PICKLE` is set, and the message says why.

`POST /import-bundle` loads a bundle's objects into the session for `/predict`. A hash in the manifest proves nothing about who wrote the file, so every exported artifact is also signed with HMAC-SHA256 under a key that belongs to the installation: `DIGITERRA_BUNDLE_KEY` when set, otherwise `bundle_signing.key` generated once (32 random bytes, owner-only permissions) in the app support directory. Import verifies the signature before anything is unpickled and refuses a bundle that is unsigned, altered, or signed by another installation, naming both key ids in the message. Two installations that set the same `DIGITERRA_BUNDLE_KEY` can exchange bundles; that is the trust decision, made by the people who share the key. The route also answers 403 unless `DIGITERRA_ALLOW_BUNDLE_IMPORT` is set; the desktop launcher sets it, the container and Helm defaults do not. `tests/unit/test_bundle.py` exports under one key and shows the import refused under another and with the signature removed. `scripts/score_bundle.py` checks the same signature and has `--trust-unsigned` for a file whose origin the operator has verified another way; `scripts/rerun_bundle.py` reads only the manifest and unpickles nothing.

A signature stops a stranger's bundle from being loaded; the allowlist stops a file, signed or not, from building a type this application does not fit. The two answer different questions and both are in place.

There is no rate limiting, so a client can still send requests that are rejected cheaply, and no queue, so a refused run is not deferred. Those need an external limiter and a job queue.

## Error responses

`core/errors.py` registers handlers so `ApiError`, `RequestEntityTooLarge`, other `HTTPException` types, and unhandled exceptions return a JSON body with an `error` string. A 404 on a page request still returns the HTML error page. Stack traces go to the log, not the response.

## Desktop wrapper

`core/desktop.py` exposes `save_file` to page JavaScript through the pywebview `js_api`. The method resolves the requested name under the output directory and refuses a path that resolves outside that directory.

## Container

`deploy/docker/Dockerfile` builds from `python:3.11-slim`, creates an `app` user, and switches to that user before the `CMD`. Gunicorn binds `0.0.0.0:5000` with one worker. The image is Linux, whose kernel applies `RLIMIT_AS`, which is what the memory bound on isolated work requires. Running the server profile directly on a macOS host is a configuration the application refuses rather than serves.

## Open items

- Build an account system with its own sessions, so that identity is per person rather than per browser and a proxy is not the only place a name can come from.
- Move the rate-limit buckets to the session store, so a limit means the same thing behind more than one worker.
- Run parsing and training in a worker with memory and wall-time limits, and put a durable queue in front of it, so a refused run is deferred rather than dropped.
- Run training in the isolated worker as parsing already is, which means moving the fitted objects across the process boundary through the artifact format and the session store, both of which now exist.
- Give the queue a worker process of its own, so a waiting run starts without a request arriving to notice the free slot.
- Put the app behind HTTPS and a reverse proxy for any network deployment.
