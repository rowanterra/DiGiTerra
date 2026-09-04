# Audit fixes: state and what is left

Branch `audit-fixes`, off `147d313` (ADA Boost, 2026-03-25).

## Done and verified

Verification scripts live in /tmp (regenerate if the container was recycled):
`verify_ml.py` (11 unit checks, all pass) and `e2e.py` (upload → preprocess →
train → predict through the Flask test client). `pytest tests/` is 106 green as of the seventh pass,
`npm run lint` is clean.

ML correctness
- Cross-validation reported RMSE and MAE as negative. The scoring dict renamed
  the scorers, so the `neg_` sign check never matched. `_build_cv_summary` now
  takes the scoring map.
- Inference broke whenever feature selection was on: the frame was subset to the
  selected columns before the scaler, which had been fit on the full transformed
  set. Scaling now happens first, against `X_scaler.feature_names_in_`.
- Feature importance charts labeled the wrong features. `feature_names` in the
  artifact was the pre-selection list; it is now the columns the model was fit on.
- A regression run's `y_scaler` survived in session storage and inverse-
  transformed the next run's cluster labels (281.65 instead of cluster 1). Cleared
  for non-regression runs, and inference now checks a recorded
  `_digiterra_problem_type`.
- Classification with a numeric target crashed: the ordinal vocabulary is built
  from `str(v)` but `y_test` kept its dtype. Both sides are strings now.
- `y` was never realigned after `dropna`, so any residual NaN produced
  "inconsistent numbers of samples". Realigned in both supervised pipelines.
- Stratify-by-bins raised KeyError on a hardcoded `QUANTILE_STRATIFY`. The
  pipeline now records which column the split actually used.
- Clustering reported a `best_k` from a silhouette sweep the fitted model never
  used, and built agglomerative centroids at that k, giving NaN rows. `best_k` is
  now the clusters the fit produced; centroids come from the labels themselves.
- Precision and recall were swapped in the fallback PR curve.
- Permutation importance scored unscaled targets against a model predicting in
  scaled space.
- Hyperparameter search ran twice (the fitted search object was refit).
- Seeds: RF classifier, ElasticNet, IsolationForest, feature-selection estimators,
  the silhouette sweep, the search's `random_state`, and SHAP sampling all take
  the run seed. Search CV folds are shuffled.
- Imputation fill values are recorded at training and reused at inference, and
  the `fillna(inplace=True)` chained assignment (a silent no-op under
  copy-on-write) is gone.

Web layer
- `GET /progress/<anything>` created a tracker on demand and held an SSE stream
  for an hour, which on the shipped single sync worker is a full outage.
  `peek_tracker` returns 404 for an unknown id; trackers are capped at 256.
- A failed training stored its error and then deleted it with the tracker, so the
  client sat at a spinner until the one-hour timeout. `ProgressTracker.fail()`
  drives progress to the terminal value and keeps the result.
- Generated files had fixed names in one shared directory, so one user's
  `predictions.csv` overwrote another's at the same URL. Each session now gets its
  own subdirectory through a context variable that the request and the training
  thread both set (`config.set_session_scope`).
- Session cookies the server never issued are replaced rather than trusted.
- `SESSION_STORE` is bounded (32) and lock-guarded.
- `MAX_CONTENT_LENGTH` is set; oversized bodies are rejected before the save.
- JSON error handlers (ported from the July tree's `errors.py`) so a 500 is not
  an HTML page the client cannot parse.
- The stratify-by-target guard accepted `[2]` and any `outputType` bypass.
- Desktop `save_file` accepted an absolute path from page JavaScript.
- `DIGITERRA_DEBUG=1` did nothing: `basicConfig` without `force=True` after
  flask_app had already configured logging.

Frontend
- 178 Advanced-mode controls were dead. `templates/index.html` had 259 duplicate
  ids, so `getElementById` always returned the hidden Simple-mode twin. The
  Advanced copies were renamed to the existing `advanced*` convention (530
  attribute rewrites) and `getModeScopedElement()` resolves by mode.
- 41 hyperparameter field groups existed only inside `#advancedOptimization`,
  which every code path hides. They were moved into the Advanced section and the
  legacy panel (1,359 lines) removed. Duplicate ids: 259 -> 0 live.
- 12 Perceptron reads were `document.getElementById('')`; one SGD branch read an
  undeclared `nJobs`.
- `.eslintrc.json` was dead and `.eslintrc.cjs` did not extend
  `eslint:recommended`, so `no-undef` was off and the ReferenceError above linted
  clean. `npm run lint` now builds the bundle, lints the split files, and lints the
  bundle with `no-undef` on. `npm run check-bundle` fails if the committed bundle
  has drifted.

## Seventh pass, 2026-09-04: the implementation re-audit at bfaca2a

`IMPLEMENTATION_RE_AUDIT_2026-09-04.md` had four findings, all sound, two of them
high. Bundle import in the desktop build was arbitrary code execution from a shared
zip, because a hash in the manifest says nothing about who wrote the file: bundles are
now signed with HMAC-SHA256 under a per-installation key and import verifies before
unpickling. Excel and Parquet inflated before any cap applied, and the desktop build
turned the caps off: the zip directory and the Parquet footer are checked against a
decoded-size budget before anything is opened, Parquet reads in batches, and the
desktop limits are large rather than absent. The session budget now runs against a
metadata estimate before the parse and counts fitted objects after a run. The
data-check override is a server-side switch that requires a reason and labels the run
not valid for decision use. Not done from that review: a pickle-free model format and
parsing or training in an isolated worker. `pytest tests/` is 106 green.

## Sixth pass, 2026-09-04 (commits 10 to 13): the market-gap items and the implementation review

Commits bc404d0, c22768f, 4216fdd, 4153210. `pytest tests/` is 97 green, `ruff check .`,
`ruff format --check .`, and `npm run lint` are clean.

Market-gap items from `MARKET_GAP_ANALYSIS.md` that are in the code now: Excel and Parquet
input with an Excel export; time-ordered and grouped holdouts; data checks before
training with a recorded override; a baseline on the same validation rows; a run
record (manifest and model card) per run; a portable bundle with export, gated import,
a scoring script, and a rerun script. `audit_handoff_2026-09-03/GAP_RESPONSE.md` has the
item-by-item disposition, including the items left out and why.

Two defects surfaced while testing the bundle. The fitted preprocessor's group-map step
was a closure and could not be pickled. `prediction()` used the scaler's
`feature_names_in_` array in a boolean `or`, which raises for more than one column, so
`/predict` returned 500 for every scaled regression model since commit ccf017f, my own
first-pass change. No test had run `/predict` after a real training run;
`tests/unit/test_inference.py` does now, for five model and scaler combinations.

The reviewer's `IMPLEMENTATION_REVIEW.md` had four findings, all sound and all
addressed in 4153210: progress-stream slots (one per run, four across the process),
a process-wide budget for uploaded data with job-protected eviction, a per-job core
budget in place of `n_jobs=-1`, and a portable stream-starvation script that exits
nonzero on failure. `tests/unit/test_capacity.py` covers the first three.

## Fifth pass, 2026-09-03 night (commit 9): the re-audit

The second reviewer re-audited commit 8 and filed `RE_AUDIT.md`: three high
findings and one medium, all sound, plus one verification failure that was a
real defect in my own test.

- The test compared floating-point means with exact equality. It now uses
  pytest.approx for the equality and a tolerance for the inequality.
- Unbounded job creation. `validate_training_request` now runs in the request
  handler before any thread exists, and `core/jobs.py` allows one run per
  session and two across the process, returning 429 past either.
- The cell cap ran after `pd.read_csv` had already parsed the file.
  `read_table_bounded` in `core/helpers.py` reads the header for the column count,
  caps the rows it parses at the cell budget, and measures the parsed frame
  against a byte budget. A rejected read peaks far below a full parse, which the
  tests assert. Served defaults dropped to 32 MB body and 5,000,000 cells.
- One stream starved the only synchronous worker. The container now runs
  gunicorn with `--worker-class gthread --threads 8`, one process, so session
  state stays put while an open stream coexists with other requests.
  `scripts/check_stream_starvation.py` measures it: the synchronous worker did
  not answer an unrelated request within ten seconds, the threaded worker
  answered in 0.01 s. The stream also ends after 15 minutes without an error
  payload, so the browser reconnects and a longer run still delivers.
- The sweep could delete a running job's directory. It now skips a session with
  a run in flight, refreshes a directory's age as it is used, and runs on a
  timer.

Still not done, and still correct as the reviewer's conclusion: authentication,
CSRF tokens, rate limiting, a durable job queue, and out-of-process session
state. Without those the network deployment is not for untrusted users.

## Fourth pass, 2026-09-03 late (commit 8): the external review handoff

A second reviewer left `audit_handoff_2026-09-03/` beside the repo with four
findings and three patches. All three patches were sound and are applied,
adapted to this tree: uploads are deleted once parsed (0001); testSize, CV
folds, search folds and search iterations are bounded server-side, with the
constants in core/constants.py and the UI inputs carrying the same ranges
(0002); a progress stream answers only the browser session that started the
run (0003). The two findings without a patch got code as well: a frame above
DIGITERRA_MAX_CELLS cells is rejected before it is stored, the served default
body limit is 64 MB with desktop_app.py raising it, and
config.sweep_stale_outputs removes session directories and upload leftovers
older than DIGITERRA_STALE_SESSION_HOURS. tests/unit/test_limits.py covers each
item and caught one more defect on the way: reset_session_scope raised on the
second teardown of a streamed response. DEPLOYMENT_OPTIONS.md is strategy (EDX
and public deployments, values files not to be written until names are known)
and needed no code beyond the escaping regression test it asked for.

Not done from the handoff, by its own scoping: authentication, CSRF tokens,
rate limiting, a job queue, a shared backing store. The residual-risk
paragraph in FINDINGS.md still holds for an untrusted network deployment.

## Third pass, 2026-09-03 night (commits 3 to 5, unattended)

Done: ruff 607 -> 0 with `ruff format` over the tree and the rest by hand; CI
now fails on `ruff check` and `ruff format --check`. Two defects surfaced by
the unused-variable cleanup: the regression pipeline returned "" in place of
the per-bin evaluation table (the "quantile Bin Results" sheet never existed),
and the user's bin labels were parsed and discarded. Imputation leakage is
fixed: `preprocessing/split_fill.py` holds a FillPlan that preprocess_data
records mean/median NaN and zero fills into; each pipeline fits the plan on its
training rows after train_test_split and applies it to both halves; the fitted
values go to session storage and inference reuses them
(tests/unit/test_split_fill.py proves the stored fill is the train-split mean,
not the whole-file mean). Regression heading 33 -> 32 models. Helm
image.repository is `required`. shap.summary_plot takes rng=seed.

RadiusNeighbors: a row with no training row inside the radius took NaN from
scikit-learn; it now takes the value of its nearest training row, and
`n_fallback_` on the model counts how many rows that was.

Still open: the `n_estmators` id spelling is consistent across HTML and JS and
was left.

## Second pass, 2026-09-03 evening (commit 2)

Done: XSS closed (escapeHtml escapes quotes; every innerHTML sink that takes
server or CSV data is wrapped; the echoed filename comes from secure_filename);
pairplot and the other nine fetch calls show an error in the page on failure;
implicit globals declared; clustering now applies outlier handling and reports
feature selection, search and CV as "Not applicable to clustering" in both the
tracker and the result (`skipped_options`, rendered on the results panel); the
every-model sweep (scripts/train_every_model.py) trains 67 of 67 offered
estimators after fixes to SVR/NuSVR/SVC/NuSVC gamma, MLP hidden_layer_sizes,
LassoCV/ElasticNetCV n_alphas (removed in scikit-learn 1.9), PassiveAggressive
n_jobs, and SGDClassifier eta0; Perceptron (a classifier registered as a
regressor) is now a rejected entry like Polynomial; a Playwright test
(tests/integration/test_browser_advanced.py) drives the page and proves Ridge,
Logistic classifier and the relocated AdaBoost group reach the /process payload
in Advanced mode, which also caught 11 slider gates that still read the Simple
id; docs rewritten for truth and register (four stale architecture docs
deleted), requirements pinned, LICENSE, CHANGELOG, CITATION.cff,
THIRD_PARTY_NOTICES.md (generated from installed metadata), Makefile, ruff
config, CI matrix 3.11-3.12 with the bundle-drift check; UI copy cut from 6,678
to 4,899 characters of field notes under the plain-technical-prose rules.

Still open: imputation leakage (unchanged, see item 2 below); RadiusNeighbors
regression returns NaN when no training row is inside the radius (needs a
larger default radius or a fallback); ruff baseline is 592 errors (354
auto-fixable) and the CI step is continue-on-error; Helm deployment.yaml does
not `required`-guard an empty image.repository; the UI still names "(33
models)" for regression where the registry has 32 trainable.

## Not done (first pass list, items 1 and 3-7 now closed)

1. XSS. `routes/upload.py` echoes the raw filename and column names, and
   `static/js/upload.js` interpolates them with `innerHTML` (lines 85, 96, 104,
   108, 206, 226) as does `modeling.js` (2870, 3131, 3341). `escapeHtml` in
   core.js does not escape quotes, so it is unsafe for the attribute case at
   upload.js:226. Fix: escape server-side in the response and harden `escapeHtml`.
2. Imputation leakage. Mean/median fills are still computed on the whole dataset
   before the train/test split, so test rows inform training fills. Fixing it
   properly means moving the fill inside the pipelines after the split, which
   collides with the `dropna` the pipelines do first. Left deliberately; the
   inference half is fixed.
3. Clustering ignores every advanced option. `app_model_training.py` builds a
   separate `base_c` kwargs dict without feature selection, outlier handling,
   hyperparameter search or CV, while the progress panel still shows those stages
   and the AutoML summary claims they ran. Either pass them through or hide the
   cards for Cluster output.
4. Documentation. `docs/SECURITY_REVIEW.md` cites line numbers in a 6-line
   `app.py`; `HANDOFF.md`, `SOFTWARE_ALIGNMENT.md` and
   `ARCHITECTURE_RECOMMENDATIONS.md` describe a pre-blueprint architecture and
   recommend work already done; `CONTRIBUTING.md:31` has the registry tuple
   backwards; `requirements.txt` claims to pin and does not;
   `docs/DEPENDENCIES.md:23` contradicts it on numpy; `examples/README.md`
   references a generator script that does not exist; three referenced docs are
   missing; the Helm default image `ghcr.io/netl/digiterra` is not pullable.
5. July tree. Only `errors.py` has been folded in so far. Still portable: the MIT
   LICENSE (the repo has none), CHANGELOG.md, the bounded thread-safe store
   pattern, ruff and mypy config, and the 3.10-3.12 CI matrix.
6. Headless browser check of the renamed Advanced controls has not been run. The
   rename and the resolver are verified by lint and by id-uniqueness, not yet by
   driving the page.
7. `nJobs`-class implicit globals (`model1`, `stratifyColumnNumber`,
   `stratifyBool`, `quantileBinDict`, `corrColsIndices`, `stratifyStr`) are
   assigned without declaration and registered in `.eslintrc.bundle.cjs` so
   `no-undef` still guards the rest. Worth declaring properly.
