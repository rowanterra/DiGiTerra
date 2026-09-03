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

Web layer

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

- MIT license.
- This changelog.
- Makefile with `install`, `dev`, `lint`, `test`, `check`, and `build-bundle` targets.
- ruff configuration in `pyproject.toml` and a CI matrix over Python 3.11 and 3.12.

### Changed

- `requirements.txt` pins every package to the version recorded by `pip freeze`. `requests` is removed because nothing imports it.
- `docs/SECURITY_REVIEW.md` describes the current upload, download, and session handling.
- `docs/HANDOFF.md`, `docs/SOFTWARE_ALIGNMENT.md`, `docs/ARCHITECTURE_RECOMMENDATIONS.md`, and `docs/CLEANUP_CHECKLIST.md` are removed. The still-true content is in `docs/documentation.md` and `CONTRIBUTING.md`.
- Helm chart `appVersion` is 0.1.0, and `image.repository` has no default.
