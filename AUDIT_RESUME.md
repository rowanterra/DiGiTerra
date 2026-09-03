# Audit fixes: state and what is left

Branch `audit-fixes`, off `147d313` (ADA Boost, 2026-03-25).

## Done and verified

Verification scripts live in /tmp (regenerate if the container was recycled):
`verify_ml.py` (11 unit checks, all pass) and `e2e.py` (upload → preprocess →
train → predict through the Flask test client). `pytest tests/` is 15 green,
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

## Not done

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
