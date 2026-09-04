# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

Model training and inference

- Cross-validation reported RMSE and MAE with a negative sign. The summary now receives the scoring map and undoes the `neg_` prefix.
- Inference with feature selection on failed because the frame was subset before the scaler ran. Scaling now runs first, against `X_scaler.feature_names_in_`.
- Feature importance charts labeled the pre-selection column list. The artifact now records the columns the model was fit on.
- A regression run's `y_scaler` stayed in session storage and inverse-transformed the cluster labels of the next run. Non-regression runs clear the scaler, and inference checks the recorded `_digiterra_problem_type`.
- Classification with a numeric target failed because the ordinal vocabulary used `str(v)` while `y_test` kept the numeric dtype. Both sides are strings now.
- `y` was not realigned after `dropna`, so a residual NaN produced an inconsistent sample count. Both supervised pipelines realign.
- Stratify-by-bins raised `KeyError` on a hardcoded column name. The pipeline records the column the split used.
- Clustering reported a `best_k` from a silhouette sweep that the fitted model did not use, and computed agglomerative centroids at that k. `best_k` is the cluster count of the fit, and centroids come from the labels.
- Precision and recall were swapped in the fallback precision-recall curve.
- Permutation importance scored unscaled targets against a model that predicts in scaled space.
- Hyperparameter search ran twice because the fitted search object was refit.
- The run seed now reaches the random forest classifier, ElasticNet, IsolationForest, feature-selection estimators, silhouette sweep, search `random_state`, and SHAP sampling. Search CV folds are shuffled.
- Mean and median imputation, and mean or median replacement of zeros, were computed over the whole file before the train/test split, so test rows informed training values. `preprocess_data` now records those columns in a `FillPlan` (`python_scripts/preprocessing/split_fill.py`); each pipeline fits the plan on its training rows after the split, applies the same values to the test rows, and the fitted values are stored and reused at inference. Constant fills and row drops still happen before the split. The `fillna(inplace=True)` chained assignment, a silent no-op under copy-on-write, is gone.
- The regression pipeline computed the per-bin evaluation table and then returned an empty string in its place, so the "quantile Bin Results" sheet was never written. The bin labels a user types were parsed and discarded; they now name the rows of that table.
- RadiusNeighborsRegressor returned NaN for a row with no training row inside the radius. Such a row now takes the value of its nearest training row, and the model records how many rows that was.
- 67 of 67 offered estimators train from an empty hyperparameter form (`scripts/train_every_model.py`): `gamma=None` in the four SVM trainers, an empty MLP `hidden_layer_sizes`, the `n_alphas` argument removed in scikit-learn 1.9, `n_jobs` on PassiveAggressiveRegressor, and `eta0=0` on SGDClassifier. Perceptron, a classifier registered as a regressor, is a rejected entry like Polynomial.
- `prediction()` in `python_scripts/helpers.py` used the scaler's `feature_names_in_` array in a boolean `or`, which raises for more than one column, so `/predict` returned 500 for every scaled regression model. `tests/unit/test_inference.py` runs `/predict` after a regression run with and without a scaler, a random forest, a classifier, and k-means.
- The fitted preprocessor's group-map step was a closure and could not be pickled. `apply_group_map` in `python_scripts/preprocessing/utilities.py` is a module-level function with the maps passed as `kw_args`.

Web layer

- The CSV reader takes the column count from the header and caps the rows it parses at the cell budget, so a file over the limit is refused without being parsed into memory first, and the parsed frame is refused above `DIGITERRA_MAX_FRAME_MB`. The served body limit is 32 MB and the cell budget 5,000,000.
- The training request is validated in the request handler, before a thread exists, and `core/jobs.py` allows one run per session and `DIGITERRA_MAX_CONCURRENT_TRAINING` across the process. A request past either returns 429.
- The container runs gunicorn with one worker process and eight threads. An open progress stream no longer blocks other requests; `scripts/check_stream_starvation.py` measures both worker classes. A stream ends after 15 minutes without an error payload so the browser reconnects.
- The output sweep skips a session with a run in flight, refreshes a directory's age as it is used, and runs on a timer rather than only when a new session cookie is issued.
- Uploaded CSVs are removed from disk once parsed, in both the upload and prediction routes, and session output directories older than `DIGITERRA_STALE_SESSION_HOURS` (default 24) are swept when a new session is issued.
- A frame above `DIGITERRA_MAX_CELLS` cells (default 25,000,000) is rejected before it is stored; the default request-body limit is 64 MB for a served deployment, and `desktop_app.py` raises both for local use.
- `testSize`, cross-validation folds, search folds, and search iterations are bounded server-side (`core/constants.py`), and the UI inputs carry the same ranges. An out-of-range value ends the run with an error instead of scheduling unbounded work.
- A progress stream answers only the browser session that started the run.
- `reset_session_scope` tolerated a token reused by the second teardown of a streamed response; it now clears the scope instead of raising.
- `GET /progress/<id>` created a tracker for any id and held an SSE stream for an hour. An unknown id now returns 404, and trackers are capped at 256.
- A failed training deleted the stored error with the tracker, so the client waited at a spinner until the timeout. `ProgressTracker.fail()` moves progress to the terminal value and keeps the result.
- Generated files had fixed names in one shared directory. Each session now writes to a subdirectory selected through `config.set_session_scope`, which the request and the training thread both set.
- Session cookies the server did not issue are replaced.
- The session store is bounded (32 sessions by default) and lock-guarded.
- `MAX_CONTENT_LENGTH` is set, so an oversized body is rejected before the save.
- Error handlers return JSON, so a 500 is not an HTML page the client cannot parse.
- The stratify-by-target guard accepted `[2]` and any `outputType` value.
- Desktop `save_file` accepted an absolute path from page JavaScript.
- `DIGITERRA_DEBUG=1` did nothing because `basicConfig` ran without `force=True` after the Flask app had configured logging.

Frontend

- 178 Advanced-mode controls did not respond because `templates/index.html` had 259 duplicate ids and `getElementById` returned the hidden Simple-mode twin. The Advanced copies use the `advanced*` naming and `getModeScopedElement()` resolves by mode.
- 41 hyperparameter field groups were inside a panel every code path hides. They are in the Advanced section and the unused panel is removed.
- 12 Perceptron reads used `document.getElementById('')`, and one SGD branch read an undeclared `nJobs`.
- ESLint did not extend `eslint:recommended`, so `no-undef` was off. `npm run lint` builds the bundle and lints the split files and the bundle. `npm run check-bundle` fails when the committed bundle has drifted.

### Added

- Data checks before training (`python_scripts/preprocessing/guardrails.py`): a feature identical or near-identical to the target, the target listed as a feature, a categorical column that maps one-to-one onto the class, constant and identifier columns, heavy missingness, duplicate rows, class imbalance, and too few rows for the requested split or folds. Blocking findings stop the run with the reasons unless the request sets `allowUnsafe` (Advanced, Data Checks); every finding is returned as `data_checks` and shown above the result.
- A baseline scored on the same validation rows (`python_scripts/preprocessing/baseline.py`): the training mean for regression, the most common class for classification. The result says whether the model beats it.
- Time-ordered and grouped holdouts (`python_scripts/preprocessing/splitting.py`; Advanced, Train/Test Split). Excel (`.xlsx`, `.xlsm`) and Parquet input; `GET /download/predictions.xlsx`.
- A run record per training run: `manifest.json` and `model_card.html` in the session directory, with the SHA-256 of the uploaded bytes, the request, seed, estimator and parameters, preprocessing and split, metrics, baseline, data checks, and package versions (`python_scripts/reporting/bundle.py`). `GET /export-bundle` zips the record with the fitted objects and the artifact hash. `POST /import-bundle` loads a bundle for `/predict` and is off unless `DIGITERRA_ALLOW_BUNDLE_IMPORT` is set; the desktop launcher sets it.
- `scripts/score_bundle.py` scores a new file from a bundle without the web app. `scripts/rerun_bundle.py` checks the data hash, replays the recorded request, and compares the metrics.
- Progress-stream slots: one stream per run and `DIGITERRA_MAX_CONCURRENT_STREAMS` (4) across the process, 429 past either. A process-wide budget for uploaded data, `DIGITERRA_MAX_TOTAL_FRAME_MB` (1024), evicting idle sessions first and refusing with 429 when only sessions with running jobs remain. Searches and feature-selection forests use `cpu_count // DIGITERRA_MAX_CONCURRENT_TRAINING` cores, or `DIGITERRA_JOB_CORES`.
- Bundles are signed with HMAC-SHA256 under the installation's key (`DIGITERRA_BUNDLE_KEY`, or `bundle_signing.key` generated in the app support directory); import verifies the signature before unpickling and refuses unsigned, altered, or foreign bundles. `scripts/score_bundle.py --trust-unsigned` is the command-line escape.
- Excel and Parquet files are measured before they are opened: zip member count, inflation ratio, and total inflated size for workbooks; row-group byte sizes from the footer for Parquet, which is then read in batches up to the row cap. `core.helpers.estimate_decoded_bytes` feeds the same numbers to the session budget before the parse, and fitted objects count toward the budget after each run. The desktop launcher sets limits at the scale of one machine instead of disabling them.
- The data-check override is a server-side switch (`DIGITERRA_ALLOW_UNSAFE_OVERRIDE`, on in the desktop launcher), requires `allowUnsafeReason`, and labels a forced run "not valid for decision use" in the result, manifest, and model card.
- A column selection that is not of the form `A-D` or `A-D,F,H` is refused in the browser with the offending value named; the server's fallback message states the file's column range in letters.
- One bounded zip preflight (`core.helpers.preflight_zip`) serves both the workbook reader and the bundle loader: allowed member names, member count, per-member and total decoded caps, inflation ratio, all from the central directory. `read_zip_member_bounded` streams each member and enforces the cap again, since a directory can lie. A bundle is bounded before its signature can be checked, because the signature is over bytes that must first be read.
- A fitted model is charged against the memory budget with `admit_model` after the fit; when it does not fit, the objects are dropped and the result says inference is unavailable rather than the run being discarded. The counting sink that measures the model lacked `tell`, so the previous release measured nothing.
- MIT license.
- This changelog.
- Makefile with `install`, `dev`, `lint`, `test`, `check`, and `build-bundle` targets.
- ruff configuration in `pyproject.toml` and a CI matrix over Python 3.11 and 3.12.

### Changed

- `requirements.txt` pins every package to the version recorded by `pip freeze`. `requests` is removed because nothing imports it.
- `docs/SECURITY_REVIEW.md` describes the current upload, download, and session handling.
- `docs/HANDOFF.md`, `docs/SOFTWARE_ALIGNMENT.md`, `docs/ARCHITECTURE_RECOMMENDATIONS.md`, and `docs/CLEANUP_CHECKLIST.md` are removed. The still-true content is in `docs/documentation.md` and `CONTRIBUTING.md`.
- Helm chart `appVersion` is 0.1.0, and `image.repository` has no default.
