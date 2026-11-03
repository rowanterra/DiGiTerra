# DiGiTerra documentation

## Workflow

1. Upload a CSV file. The file becomes the working dataset.
2. Data Exploration computes correlation matrices, pairplots, and descriptive statistics.
3. Model Preprocessing selects targets and indicators, cleaning, and transformations.
4. Modeling trains the model and computes metrics.
5. Inference applies the trained model to a new dataset. The Inference Results page shows the inference summary, training plot (Predicted vs Actual and Residuals) with the inference points drawn on the scatter panel, and inference distribution histogram.
6. Downloads provide PDF figures and spreadsheet summaries.

## Terms

Indicators are the input columns. Targets are the output columns to predict or cluster. Transformers encode non-numeric columns for modeling. Stratification balances the split or the evaluation on a selected variable.

## Regression stability across seeds

`python_scripts/preprocessing/evaluate_regression_robustness.py` repeats the train/test split and cross-validation over a set of seeds, keeps the split consistent within each seed, and writes per-seed metrics for every model and feature-set combination together with aggregate statistics (median, mean, standard deviation, and 95% empirical interval). A run of 30 seeds is a reasonable screen. The finalist is then confirmed on an external validation dataset.

## File format and column entry

DiGiTerra reads CSV, Excel (.xlsx and .xlsm, first sheet only), and Parquet files. The first row of a CSV or Excel file is the header, and the Parquet schema names the columns.

Columns are selected with spreadsheet-style letters (A, B, C, ...). A range is written `A-D`, and several selections are separated with commas, for example `A-D, F, H`.

## Process map

```
Upload  ->  Data Exploration  ->  Model Preprocessing  ->  Modeling  ->  Outputs
              |                        |                      |
              +------ Backend pipelines +---------------------+
```

## Code map

- Page: `templates/index.html`, with `static/js/` modules (`init.js`, `core.js`, `upload.js`, `preprocess.js`, `modeling.js`, `inference.js`, `app.js`) or `static/js/app.bundle.js` when `DIGITERRA_USE_JS_BUNDLE=1`.
- Styling: `static/style.css`.
- Application object, session cookie, and upload size limit: `core/flask_app.py`.
- Routes: `routes/` (`main.py`, `upload.py`, `exploration.py`, `preprocess.py`, `modeling.py`, `prediction.py`, `assets.py`).
- Session store, error handlers, constants: `core/state.py`, `core/errors.py`, `core/constants.py`.
- Training orchestration: `python_scripts/app_model_training.py`, with the model lookup in `python_scripts/model_registry.py`.
- Data exploration (correlation, pairplot, auto-detect): `python_scripts/app_exploration.py`.
- Inference: `python_scripts/app_prediction.py`.
- Preprocessing and plotting: `python_scripts/preprocessing/`, `python_scripts/plotting/`.
- Plot style: `python_scripts/plotting/plot_style.py`. `apply_plot_style()` runs at the start of `visualize_predictions()` and `plot_regression_bundle()`, so training and export figures share one style.
- Paths and URL prefix: `python_scripts/config.py`.

## Running the web app

`python app.py` starts the Flask development server on `127.0.0.1:5000`. `DIGITERRA_HOST` and `DIGITERRA_PORT` override the address.

For production, gunicorn with one worker:

```bash
gunicorn --bind 0.0.0.0:5000 --workers 1 --worker-class gthread --threads 8 --timeout 120 app:app
```

Session state is in process memory, so a second worker would answer "No data uploaded" for a session the first worker created.

`URL_PREFIX` serves the app under a prefix as well as at the root:

```bash
export URL_PREFIX=digiterra
python app.py
```

The page is then at `http://127.0.0.1:5000/digiterra/` and at `http://127.0.0.1:5000/`. The Dockerfile sets `URL_PREFIX=digiterra`.

Other environment variables: `DIGITERRA_BASE_DIR`, `DIGITERRA_APP_SUPPORT_DIR`, `DIGITERRA_OUTPUT_DIR` (paths), `LOG_LEVEL`, `DIGITERRA_MAX_UPLOAD_MB`, `DIGITERRA_WARN_UPLOAD_MB`, `DIGITERRA_ENFORCE_UPLOAD_LIMIT`, `DIGITERRA_MAX_SESSIONS`, `DIGITERRA_MAX_CELLS`, `DIGITERRA_MAX_FRAME_MB`, `DIGITERRA_MAX_TOTAL_FRAME_MB`, `DIGITERRA_MAX_CONCURRENT_TRAINING`, `DIGITERRA_MAX_CONCURRENT_STREAMS`, `DIGITERRA_JOB_CORES`, `DIGITERRA_DEFAULT_MAX_UPLOAD_MB`, `DIGITERRA_ALLOW_BUNDLE_IMPORT`, `DIGITERRA_BUNDLE_KEY`, `DIGITERRA_MAX_BUNDLE_MEMBER_MB`, `DIGITERRA_MAX_BUNDLE_TOTAL_MB`, `DIGITERRA_ALLOW_UNSAFE_OVERRIDE`, `DIGITERRA_PROFILE`, `DIGITERRA_REQUIRE_AUTH`, `DIGITERRA_REQUIRE_CSRF`, `DIGITERRA_RATE_LIMIT`, `DIGITERRA_RATE_PER_MINUTE`, `DIGITERRA_RATE_BURST`, `DIGITERRA_AUTH_HEADER`, `DIGITERRA_TRUSTED_PROXIES`, `DIGITERRA_SHARED_SECRET`, `DIGITERRA_ISOLATE_WORK`, `DIGITERRA_WORKER_MEMORY_MB`, `DIGITERRA_WORKER_BASE_MB`, `DIGITERRA_WORKER_TIMEOUT_SECONDS`, `DIGITERRA_PARSE_TIMEOUT_SECONDS`, `DIGITERRA_ALLOW_LEGACY_PICKLE`, `DIGITERRA_SESSION_BACKEND`, `DIGITERRA_QUEUE`, `DIGITERRA_QUEUE_PATH`, `DIGITERRA_MAX_QUEUE_DEPTH`, `DIGITERRA_QUEUE_STALE_SECONDS`, `DIGITERRA_PROGRESS_TIMEOUT_SECONDS`, `DIGITERRA_RESULT_WAIT_SECONDS` (how long the progress stream waits for a result after the progress reads 100, 120 by default), `DIGITERRA_STALE_SESSION_HOURS`, `DIGITERRA_MIN_TEST_SIZE`, `DIGITERRA_MAX_TEST_SIZE`, `DIGITERRA_MAX_CV_FOLDS`, `DIGITERRA_MAX_SEARCH_CV_FOLDS`, `DIGITERRA_MAX_SEARCH_ITERATIONS`, and `DIGITERRA_COOKIE_SECURE`. `docs/SECURITY_REVIEW.md` describes what each limit does. `DIGITERRA_CHROMIUM` names a Chromium binary for the browser tests when Playwright's own is absent.

## How far the new rows are from the training data

A model returns a prediction for any row whose columns parse, including a row nothing
like the data it was trained on. Two sets of columns in `predictions.csv` say which of
those a prediction is.

`novelty_score` comes from an IsolationForest fitted on the training rows, on the same
features the model reads. A higher score means a row more like the training rows.
`novel_row` is true when the score falls below the first percentile of the training
rows' own scores. That threshold is recorded when the model is trained and is not
recomputed from the new file, because a file of unusual rows would otherwise look
ordinary against itself. `out_of_range_columns` lists the numeric columns whose value
for that row falls outside the range seen in training, and is empty when none do.

A regression run also writes `lower_90` and `upper_90`, a 90 percent interval from
split conformal. The half-width is the quantile of the absolute residuals on the
validation rows, in the units of the target. The interval contains the observed value for at least nine
rows in ten, provided the new rows are exchangeable with the validation rows. Where they are
not, that coverage is not guaranteed, and the novelty score is the check on that: a row reported as
unlike the training data is a row whose interval should not be relied on.

A classification run writes one `probability_` column per class and `max_probability`.
An estimator with no probability of its own writes neither, and the model card records
that it could not.

The model card records the detector, the threshold, the training ranges, alpha and the
validation size, so a reader of a run sees the same numbers without the file.

## Accessibility

### ARIA attributes

`templates/index.html` uses `role="navigation"` on the navigation, `role="status"` on status messages, and `role="region"` with labels on the main sections. Dynamic updates use `aria-live` regions (`polite` for general updates, `assertive` for errors). Controls without visible text have `aria-label`. Form inputs reference help text with `aria-describedby`, required fields have `aria-required="true"`, and the scripts set `aria-invalid="true"` on a field that fails validation.

### Keyboard navigation

A "Skip to main content" link appears on focus. The scripts move focus to the active tab and to error messages. Interactive elements have visible focus outlines, and the tab order follows the page order.

### Screen readers

Status changes and errors are announced through the live regions. Every form control and button has an accessible name. Extra descriptions use the `sr-only` class, which hides the text visually. Toggle switches receive ARIA labels from the adjacent text.

### Visual structure

Focus states use a high-contrast outline. Headings follow a hierarchy, and the document declares `lang="en"`.
