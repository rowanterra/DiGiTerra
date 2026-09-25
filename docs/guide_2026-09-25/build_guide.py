"""Build docs/digiterra_guide_2026-09-25.html, the DiGiTerra 0.1.0 guide.

Inputs (relative to this folder, docs/guide_2026-09-25/):
  cayman.css        the stylesheet of the Terrahedral guide of 2026-09-24 (prettydoc cayman copy)
  shots/*.png       screenshots written by drive_guided.py on 2026-09-25
  fig/*.png         docs/audit_2026-09-25/figures/*.pdf rendered with pdftoppm -r 150
  ../../static/doc_banner.jpg   the contour masthead of the Documentation view
Output: ../digiterra_guide_2026-09-25.html, one self-contained file with every image embedded.

Run from the repository root: python3 docs/guide_2026-09-25/build_guide.py
"""

import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent


def data_uri(path: Path) -> str:
    """Return the file as a base64 data URI with the type set from the suffix."""
    kind = {"png": "image/png", "jpg": "image/jpeg"}[path.suffix[1:]]
    return f"data:{kind};base64," + base64.b64encode(path.read_bytes()).decode()


def figure(path: str, alt: str, caption: str, width: str = "100%") -> str:
    """Return a figure block with an embedded image and a caption."""
    return (f'<figure><img src="{data_uri(HERE / path)}" alt="{alt}" style="width:{width}">'
            f"<figcaption>{caption}</figcaption></figure>")


# ---- stylesheet: the Terrahedral guide's cayman copy, masthead set to the app's banner
css = (HERE / "cayman.css").read_text()
css = css.replace(
    "background-image:linear-gradient(180deg, rgba(12,32,25,0.58), rgba(12,32,25,0.74)),none;",
    "background-image:linear-gradient(180deg, rgba(12,32,25,0.58), rgba(12,32,25,0.74)),"
    f"url('{data_uri(HERE.parent.parent / 'static' / 'doc_banner.jpg')}');",
)
css += """
.main-content table.plain th { background-color: transparent; color: #24503d;
  border-bottom: 1px solid #24503d; }
.main-content table.plain tr:nth-child(odd) { background-color: transparent; }
.main-content table.plain td.num, .main-content table.plain th.num { text-align: right; }
.main-content details.more { margin-bottom: 1rem; }
.main-content details.more > summary { cursor: pointer; color: #24503d; }
.main-content table.plain { font-size: 0.92rem; }
.main-content td.num { white-space: nowrap; }
.main-content p code, .main-content li code { overflow-wrap: anywhere; }
@media screen and (max-width: 42em) {
  .main-content table.plain { display: block; overflow-x: auto; }
  .tabs { flex-wrap: wrap; }
  .tabs button { padding: 0.45rem 0.7rem; font-size: 0.9rem; }
}
"""

HEAD = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DiGiTerra 0.1.0 Guide</title>
<style>{css}</style>
</head>
<body>
<section class="page-header">
<h1 class="title toc-ignore project-name">DiGiTerra: Tabular Machine Learning in the Browser with scikit-learn</h1>
<h3 class="subtitle project-tagline">A walkthrough of DiGiTerra 0.1.0 on the guided and classic pages, from installation to predictions on new rows, with the verification record</h3>
<h4 class="author project-author">Paige E. DiGirolamo, Rowan R. Terra, Djuna M. Gulliver, and Nancy J. Trun, 2026-09-25</h4>
</section>
<div class="tabs" role="tablist">
<button type="button" role="tab" id="tab-guide" aria-controls="pane-guide" aria-selected="true">How to use DiGiTerra</button>
<button type="button" role="tab" id="tab-verify" aria-controls="pane-verify" aria-selected="false">Verification record</button>
<button type="button" role="tab" id="tab-refs" aria-controls="pane-refs" aria-selected="false">References</button>
</div>
<div class="tabs-rule"><div></div></div>
"""

GUIDE = f"""
<section class="main-content" id="pane-guide" role="tabpanel" aria-labelledby="tab-guide">

<p>DiGiTerra is a Flask application for data exploration, preprocessing, model training, and inference on tabular data in CSV, Excel, and Parquet files. A file is uploaded in the browser, a scikit-learn estimator (Pedregosa et al., 2011) is fitted through one training route, and every run writes a record that names the SHA-256 of the uploaded file, seed, request, fitted parameters, metrics, data-check findings, and installed package versions. The application runs in a browser against a local or hosted server, or in a desktop window on macOS, Windows, and Linux. This tab follows one example from installation to predictions on new rows with the Palmer penguin file in <code>examples/</code>. The second tab records what the numbers were checked against, and the third tab lists the works cited.</p>

<div class="contents"><strong>Contents</strong><ul>
<li><a href="#install">Installation</a></li>
<li><a href="#first">A first model</a></li>
<li><a href="#pages">The guided and classic pages</a></li>
<li><a href="#hyper">Hyperparameters</a></li>
<li><a href="#record">Run record and bundle</a></li>
<li><a href="#inference">Inference on new rows</a></li>
<li><a href="#server">Running the web app</a></li>
<li><a href="#seeds">Regression stability across seeds</a></li>
<li><a href="#code">Code map</a></li>
<li><a href="#access">Accessibility</a></li>
<li><a href="#cite">Version and citation</a></li>
</ul></div>

<h2 id="install">Installation</h2>
<p>DiGiTerra 0.1.0 runs on Python 3.11 and 3.12. From the project root, the directory with <code>app.py</code> and <code>requirements.txt</code>, the pinned runtime dependencies install with pip, and either entry point starts the application.</p>
<div class="lang">SHELL</div><pre><code>pip install -r requirements.txt
python app.py            # web app at http://127.0.0.1:5000, guided page at /guided
python desktop_app.py    # the same application in a desktop window</code></pre>
<p>The desktop window needs pywebview, which is pinned in <code>requirements.txt</code>. The test suite installs from <code>requirements-dev.txt</code> and runs with <code>MPLBACKEND=Agg pytest tests/ -q</code> or <code>make test</code>. <code>make check</code> runs lint, tests, the JavaScript bundle check, and the requirements check.</p>
<p>The prebuilt application for each platform contains the Python interpreter and every dependency, so an end user installs neither Python nor pip. Each build is downloaded from the Releases page of the repository.</p>
<ol>
<li>macOS: drag <code>DiGiTerra.app</code> into <code>/Applications</code> and double-click the app.</li>
<li>Windows: extract the <code>DiGiTerra</code> folder, for example to <code>C:\\Program Files\\DiGiTerra</code>, and double-click <code>DiGiTerra.exe</code>.</li>
<li>Linux: extract the <code>DiGiTerra</code> folder, run <code>chmod +x DiGiTerra/DiGiTerra</code>, and start <code>./DiGiTerra/DiGiTerra</code>.</li>
</ol>
<p>The desktop launcher picks a random open port, and <code>DIGITERRA_PORT</code> sets a fixed one. With <code>DIGITERRA_DEBUG=1</code> the launcher writes a log to <code>~/Library/Logs/DiGiTerra/digiterra.log</code> on macOS, <code>%APPDATA%\\DiGiTerra\\Logs\\digiterra.log</code> on Windows, and <code>~/.cache/DiGiTerra/logs/digiterra.log</code> on Linux.</p>

<h2 id="first">A first model</h2>
<p><code>examples/penguins.csv</code> holds the Palmer Archipelago penguin measurements of Gorman et al. (2014): 344 rows, a text target <code>species</code> (Adelie 152, Gentoo 124, Chinstrap 68), two text columns, four numeric columns, and 11 rows with at least one blank. The steps below were carried out on the guided page on 2026-09-25, and the output blocks are the text the page showed.</p>
<ol>
<li>Start <code>python app.py</code> and open <code>http://127.0.0.1:5000/guided</code>.</li>
<li>In step 1, choose <code>examples/penguins.csv</code> and press Upload. Step 2 opens.</li>
<li>In step 2, choose <code>species</code> as the target. The page proposes the last column of the file, which for this file is <code>sex</code>.</li>
<li>In step 3, keep the proposed task and model and press Fit the model. Step 4 opens when the run ends.</li>
</ol>
<div class="lang">OUTPUT, STEPS 1 AND 2</div><pre><code>penguins.csv: 344 rows, 7 columns.
species has 3 distinct values. Proposed task: classification.</code></pre>
<div class="lang">OUTPUT, STEP 4</div><pre><code>Metric                Value
Accuracy              1.000
Precision (weighted)  1.000
Recall (weighted)     1.000
F1 (weighted)         1.000

Baseline (Predicting the most common training class (Adelie, 115 of 266 training rows)):
accuracy 0.4627 against the model's 1. The model beat the baseline.

Data checks: no findings on 333 rows and 6 predictors.</code></pre>
{figure("shots/step4.png", "Step 4 of the guided page with the metrics table, the baseline line, the data-check line, and the bundle and model card links", "Step 4 of the guided page after the random forest run on penguins.csv, in Guided mode at 1280 by 800 pixels. Screenshot of 2026-09-25 at commit c6b9526.")}
<p>The proposed model is a random forest (<code>RF_classifier</code>, scikit-learn <code>RandomForestClassifier</code> with 100 trees and <code>random_state</code> 42). The 11 rows with a blank were removed under the default blank rule, and the remaining 333 rows were split at random into 266 training rows and 67 test rows. The metrics are computed on the 67 test rows. The baseline predicts the most common training class for every test row, so a model that does not beat 0.4627 on this file has learned nothing that the class counts do not already show.</p>
<div class="note"><p>The guided page of 0.1.0 fits on the numeric predictor columns only. <code>island</code> and <code>sex</code> are text columns, and the run above read <code>bill_length_mm</code>, <code>bill_depth_mm</code>, <code>flipper_length_mm</code>, and <code>body_mass_g</code>. The list under <code>model.features</code> in <code>manifest.json</code> names the columns the estimator read, while the data-check line counts the six columns in the request. The classic page encodes text columns through the transformer step on Model Preprocessing.</p></div>

<h2 id="pages">The guided and classic pages</h2>
<p>The guided page at <code>/guided</code> is four steps in one column: upload a table, choose the target, choose the model and run, and read the result. One step is on screen at a time and fills the window, so the page does not scroll. The step buttons in the header and the Back and Next buttons move between steps that are already open. A step opens when the previous step has produced the input the next step needs, and the upload opens steps 2 and 3 together because the target and the model are filled with proposals.</p>
<p>A two-state control at the top selects Guided or Full, and the browser remembers the choice. Guided fills every setting with a default and shows the four steps. Full opens every setting the training route accepts in the same steps: the predictor columns in step 2, and in step 3 the split method (random, time-ordered, or grouped holdout) and split column, test fraction, stratification column, scaler, blank and zero handling, random seed, decimal places, feature selection with k, outlier method, cross-validation and folds, hyperparameter search, and the hyperparameter table of the chosen model. Every control has a help button beside the control's label. The explanation opens on hover, focus, or click, closes on Escape, and is tied to the control with <code>aria-describedby</code>.</p>
<p>The classic page at <code>/</code> is the original interface, with the green masthead, a Documentation view, and a Tutorial. Try an Example loads <code>penguins.csv</code> from the server and opens Data Exploration without an upload. The classic page runs the same training route through six stages.</p>
<ol>
<li>Upload a CSV, Excel (<code>.xlsx</code> and <code>.xlsm</code>, first sheet only), or Parquet file. The first row of a CSV or Excel file is the header, and the Parquet schema names the columns.</li>
<li>Data Exploration computes correlation matrices, pairplots, and descriptive statistics.</li>
<li>Model Preprocessing selects the indicators and targets, cleaning, and transformations.</li>
<li>Modeling trains the model and computes the metrics.</li>
<li>Inference applies the trained model to a new file. The Inference Results page shows the inference summary, the training plot (predicted against actual, and residuals) with the inference points drawn on the scatter panel, and the inference distribution histogram.</li>
<li>Downloads provide PDF figures and spreadsheet summaries.</li>
</ol>
<p>On Model Preprocessing, columns are selected with spreadsheet-style letters. A range is written <code>A-D</code>, and several selections are separated with commas, as in <code>A-D, F, H</code>. For <code>penguins.csv</code>, <code>B-G</code> as indicators and <code>A</code> as the target exercise one-hot encoding, missing-value handling, a stratified split, and three-class classification in one run. Indicators are the input columns. Targets are the output columns to predict or cluster. Transformers encode non-numeric columns for modeling, and stratification balances the split or the evaluation on a selected variable. <code>examples/README.md</code> lists the letters for the other nine example files.</p>
{figure("shots/classic.png", "The classic page with the green masthead, the Documentation and Tutorial buttons, and the Start Modeling and Try an Example buttons", "The classic page at the root path. Screenshot of 2026-09-25 at commit c6b9526.")}

<h2 id="hyper">Hyperparameters</h2>
<p>In Full mode, step 3 shows a table of the chosen estimator's parameters with the default of each. The route <code>/model-defaults/&lt;name&gt;</code> reads the names and defaults from the scikit-learn estimator itself, so the table follows the installed scikit-learn. A blank cell keeps the default. A typed value reaches the estimator and is recorded in the run record. The route <code>/models</code> marks which models have an editable table. At commit c6b9526 the table is editable for 19 of 23 classification models, 1 of 32 regression models (<code>Linear</code>), and 0 of 12 clustering models. The trainers of the other models use parameter names that differ from scikit-learn's, and the page states this and points to the classic page, where the Advanced controls set those parameters.</p>
{figure("shots/step3_full_knn.png", "Step 3 of the guided page in Full mode with KNeighbors chosen, the hyperparameter table showing n_neighbors typed as 3, and the split, preprocessing, selection, outlier, and validation settings", "Step 3 in Full mode with KNeighbors and n_neighbors set to 3 on penguins.csv. Screenshot of 2026-09-25 at commit c6b9526.")}
<p>The KNeighbors run in the figure shows the pass-through. The value 3 typed for <code>n_neighbors</code> is recorded as <code>{{"n_neighbors": 3}}</code> under <code>request.hyperparameters</code>, and the fitted estimator's parameters under <code>model.params</code> hold <code>n_neighbors</code> 3. The run scored accuracy 0.806 on the same 67 test rows against the same baseline of 0.4627. The predictors were not scaled for this run, and a distance model on columns in millimeters and grams is dominated by body mass, so the scaler setting in the same step is the first thing to change for this model.</p>
<p>Hyperparameter search replaces hand-typed values with a cross-validated search. Randomized search tries a sample of combinations, and grid search tries every combination. The values the search chose are written to the run record.</p>

<h2 id="record">Run record and bundle</h2>
<p>Every training run writes <code>manifest.json</code> and <code>model_card.html</code> into the session's output directory. The result links the model card and <code>GET /export-bundle</code>, a zip file of the record with the fitted objects. The bundle from the penguin run holds <code>manifest.json</code>, <code>model_card.html</code>, <code>model_card.json</code>, <code>model.skops</code>, and <code>visualizations.pdf</code>. The manifest has the format name <code>digiterra-bundle/2</code> and these sections.</p>
<table class="plain">
<thead><tr><th>Section</th><th>Contents in the penguin run</th></tr></thead>
<tbody>
<tr><td><code>data</code></td><td>File name, SHA-256 (<code>ced42ea2a83f</code>, first 12 of 64 characters), 15,010 bytes, 344 rows, 7 columns, upload time</td></tr>
<tr><td><code>model</code></td><td>Model name, task, estimator class, fitted parameters, the features read, targets, seed 42</td></tr>
<tr><td><code>request</code></td><td>The request as sent by the page, including the hyperparameters typed</td></tr>
<tr><td><code>preprocessing</code></td><td>Blank and zero rules, fill values, split note, feature selection, outliers</td></tr>
<tr><td><code>applicability</code></td><td>The novelty detector, detector threshold, and training range of each feature</td></tr>
<tr><td><code>uncertainty</code></td><td>Class probabilities for classification, the conformal interval for regression</td></tr>
<tr><td><code>metrics</code>, <code>cross_validation</code>, <code>baseline</code></td><td>Test metrics, cross-validation scores when run, the baseline comparison</td></tr>
<tr><td><code>data_checks</code></td><td>Rows and predictors checked, blocking findings, warnings</td></tr>
<tr><td><code>environment</code></td><td>Python, NumPy, pandas, scikit-learn, SciPy, SHAP, joblib and DiGiTerra versions, and the platform</td></tr>
<tr><td><code>artifact</code></td><td>Artifact file, SHA-256, size, serializer (skops 0.14.0), HMAC-SHA256 signature, key id</td></tr>
</tbody></table>
<p>Loading a bundle rebuilds fitted objects from a file. The artifact is therefore written with skops (skops developers, n.d.), which stores the objects as data with the type of each object named, and the loader refuses any type outside an exact allowlist (<code>python_scripts/reporting/artifact_allowlist.txt</code>, written by <code>scripts/artifact_type_census.py</code> from every registered estimator). Every bundle is also signed under the installation's key. Import refuses a bundle that is unsigned, altered, or signed by another installation, and installations that share <code>DIGITERRA_BUNDLE_KEY</code> can exchange bundles. The in-app import (<code>POST /import-bundle</code>) is on in the desktop build and off in the container unless <code>DIGITERRA_ALLOW_BUNDLE_IMPORT</code> is set.</p>
<p>Two scripts use a bundle without the web app. <code>scripts/rerun_bundle.py</code> checks the data hash, replays the recorded request, and compares each metric with the recorded value. <code>scripts/score_bundle.py</code> scores a new file. The scoring script runs in a temporary support directory, so the installation's key file is not found there, and the key is passed with <code>DIGITERRA_BUNDLE_KEY</code> or by pointing <code>DIGITERRA_APP_SUPPORT_DIR</code> at the installation's directory (<code>~/.local/share/DiGiTerra</code> on Linux).</p>
<div class="lang">SHELL</div><pre><code>python scripts/rerun_bundle.py bundle.zip examples/penguins.csv
DIGITERRA_APP_SUPPORT_DIR=~/.local/share/DiGiTerra \\
  python scripts/score_bundle.py bundle.zip examples/penguins_new.csv --out scored.csv</code></pre>
<div class="lang">OUTPUT</div><pre><code>accuracy         recorded        1.000  rerun        1.000  ok
precision        recorded        1.000  rerun        1.000  ok
recall           recorded        1.000  rerun        1.000  ok
f1score          recorded        1.000  rerun        1.000  ok
macro_precision  recorded        1.000  rerun        1.000  ok
macro_recall     recorded        1.000  rerun        1.000  ok
macro_f1score    recorded        1.000  rerun        1.000  ok
support          recorded       67.000  rerun       67.000  ok
all recorded metrics reproduced</code></pre>

<h2 id="inference">Inference on new rows</h2>
<p>A model returns a prediction for any row whose columns parse, including a row unlike the training data. Step 4 of the guided page and the Inference page of the classic page write <code>predictions.csv</code> with the input columns, the prediction, and columns that say how far each row is from the training rows. <code>examples/penguins_new.csv</code> is a seeded synthetic file of 22 rows drawn from normal distributions fitted to each species (8 Adelie, 6 Chinstrap, 8 Gentoo), and <code>examples/penguins_new_key.csv</code> records the species each row was drawn from.</p>
<div class="lang">OUTPUT, predictions.csv, FIRST TWO ROWS</div><pre><code>row_id,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex,Predicted_Target,
  probability_Adelie,probability_Chinstrap,probability_Gentoo,max_probability,novelty_score,novel_row,out_of_range_columns
1,Biscoe,40.7,13.6,204,4600,Female,Gentoo,0.44,0.08,0.48,0.48,-0.5443835443542763,False,
2,Dream,46.7,18.7,197,3975,Female,Chinstrap,0.05,0.95,0.0,0.95,-0.4345095890995424,False,</code></pre>
<p>For the random forest of the first model, the predicted species matched the key for 22 of 22 rows. No row was marked novel, and no row had a column outside the training range. Row 1 is the least certain prediction in the file, with a probability of 0.48 for Gentoo and 0.44 for Adelie.</p>
<p><code>novelty_score</code> comes from an isolation forest (Liu et al., 2008) fitted on the training rows, on the features the model reads. A higher score means a row more like the training rows. <code>novel_row</code> is true when the score falls below the first percentile of the training rows' own scores. The threshold is recorded when the model is trained and is not recomputed from the new file, because a file of unusual rows would otherwise look ordinary against itself. <code>out_of_range_columns</code> lists the numeric columns whose value for that row falls outside the range seen in training, and is empty when none do.</p>
<p>A regression run also writes <code>lower_90</code> and <code>upper_90</code>, a 90 percent interval from split conformal prediction (Lei et al., 2018). The half-width is the quantile of the absolute residuals on the validation rows, in the units of the target. The interval covers at least nine new rows in ten on average, provided the new rows are exchangeable with the validation rows. The novelty score is the check on that condition, and the interval of a row marked novel should not be relied on. A classification run writes one <code>probability_</code> column per class and <code>max_probability</code>. An estimator with no probability output writes neither, and the model card records the omission. The model card also records the detector, the threshold, the training ranges, alpha, and the validation size.</p>
<p><code>scripts/score_bundle.py</code> writes the prediction and probability columns and stops at <code>max_probability</code>. The novelty and range columns are written by the web app only.</p>

<h2 id="server">Running the web app</h2>
<p><code>python app.py</code> starts the Flask development server on <code>127.0.0.1:5000</code>, and <code>DIGITERRA_HOST</code> and <code>DIGITERRA_PORT</code> override the address. For production, gunicorn runs with one worker.</p>
<div class="lang">SHELL</div><pre><code>gunicorn --bind 0.0.0.0:5000 --workers 1 --worker-class gthread --threads 8 --timeout 120 app:app</code></pre>
<p>Session state is kept in process memory, so a second worker would answer "No data uploaded" for a session that the first worker created. <code>URL_PREFIX</code> serves the app under a prefix as well as at the root. With <code>URL_PREFIX=digiterra</code> the page is at <code>http://127.0.0.1:5000/digiterra/</code> and at <code>http://127.0.0.1:5000/</code>, and the Dockerfile sets that prefix. Docker and Helm assets are under <code>deploy/</code>, and <code>deploy/README.md</code> describes the build and the install.</p>
<details class="more"><summary>Other environment variables</summary>
<p><code>DIGITERRA_BASE_DIR</code>, <code>DIGITERRA_APP_SUPPORT_DIR</code>, and <code>DIGITERRA_OUTPUT_DIR</code> set paths. The remaining variables set logging, size and concurrency limits, bundle handling, authentication, rate limits, the job queue, and split and search bounds: <code>LOG_LEVEL</code>, <code>DIGITERRA_MAX_UPLOAD_MB</code>, <code>DIGITERRA_WARN_UPLOAD_MB</code>, <code>DIGITERRA_ENFORCE_UPLOAD_LIMIT</code>, <code>DIGITERRA_MAX_SESSIONS</code>, <code>DIGITERRA_MAX_CELLS</code>, <code>DIGITERRA_MAX_FRAME_MB</code>, <code>DIGITERRA_MAX_TOTAL_FRAME_MB</code>, <code>DIGITERRA_MAX_CONCURRENT_TRAINING</code>, <code>DIGITERRA_MAX_CONCURRENT_STREAMS</code>, <code>DIGITERRA_JOB_CORES</code>, <code>DIGITERRA_DEFAULT_MAX_UPLOAD_MB</code>, <code>DIGITERRA_ALLOW_BUNDLE_IMPORT</code>, <code>DIGITERRA_BUNDLE_KEY</code>, <code>DIGITERRA_MAX_BUNDLE_MEMBER_MB</code>, <code>DIGITERRA_MAX_BUNDLE_TOTAL_MB</code>, <code>DIGITERRA_ALLOW_UNSAFE_OVERRIDE</code>, <code>DIGITERRA_PROFILE</code>, <code>DIGITERRA_REQUIRE_AUTH</code>, <code>DIGITERRA_REQUIRE_CSRF</code>, <code>DIGITERRA_RATE_LIMIT</code>, <code>DIGITERRA_RATE_PER_MINUTE</code>, <code>DIGITERRA_RATE_BURST</code>, <code>DIGITERRA_AUTH_HEADER</code>, <code>DIGITERRA_TRUSTED_PROXIES</code>, <code>DIGITERRA_SHARED_SECRET</code>, <code>DIGITERRA_ISOLATE_WORK</code>, <code>DIGITERRA_WORKER_MEMORY_MB</code>, <code>DIGITERRA_WORKER_BASE_MB</code>, <code>DIGITERRA_WORKER_TIMEOUT_SECONDS</code>, <code>DIGITERRA_PARSE_TIMEOUT_SECONDS</code>, <code>DIGITERRA_ALLOW_LEGACY_PICKLE</code>, <code>DIGITERRA_SESSION_BACKEND</code>, <code>DIGITERRA_QUEUE</code>, <code>DIGITERRA_QUEUE_PATH</code>, <code>DIGITERRA_MAX_QUEUE_DEPTH</code>, <code>DIGITERRA_QUEUE_STALE_SECONDS</code>, <code>DIGITERRA_PROGRESS_TIMEOUT_SECONDS</code>, <code>DIGITERRA_STALE_SESSION_HOURS</code>, <code>DIGITERRA_MIN_TEST_SIZE</code>, <code>DIGITERRA_MAX_TEST_SIZE</code>, <code>DIGITERRA_MAX_CV_FOLDS</code>, <code>DIGITERRA_MAX_SEARCH_CV_FOLDS</code>, <code>DIGITERRA_MAX_SEARCH_ITERATIONS</code>, and <code>DIGITERRA_COOKIE_SECURE</code>. <code>docs/SECURITY_REVIEW.md</code> describes each limit.</p>
</details>

<h2 id="seeds">Regression stability across seeds</h2>
<p><code>python_scripts/preprocessing/evaluate_regression_robustness.py</code> repeats the train and test split and the cross-validation over a range of seeds. The split is held fixed within each seed, and the script writes per-seed metrics for every model and feature-set combination together with the median, mean, standard deviation, and 95 percent empirical interval.</p>
<div class="lang">SHELL</div><pre><code>python python_scripts/preprocessing/evaluate_regression_robustness.py \\
  --input /path/to/data.xlsx --sheet Literature --target TREE \\
  --features pH,Fe,Mn,Al,SO4 --keep-feature pH --stratify-col pH \\
  --seed-start 0 --seed-count 30 --output-dir output --output-stem tree_robustness</code></pre>
<p>The run writes <code>output/tree_robustness_per_seed.csv</code>, with one row per seed, model, and feature set, and <code>output/tree_robustness_summary.csv</code>. Thirty seeds are enough for a screen, and the finalist is then confirmed on an external validation dataset.</p>

<h2 id="code">Code map</h2>
<details class="more"><summary>Where each part of the application is in the repository</summary>
<table class="plain">
<thead><tr><th>Part</th><th>Files</th></tr></thead>
<tbody>
<tr><td>Guided page</td><td><code>templates/guided.html</code>, <code>static/js/guided.js</code>, <code>routes/guided.py</code></td></tr>
<tr><td>Classic page</td><td><code>templates/index.html</code> with the modules in <code>static/js/</code>, or <code>static/js/app.bundle.js</code> when <code>DIGITERRA_USE_JS_BUNDLE=1</code>; styling in <code>static/style.css</code></td></tr>
<tr><td>Application object, session cookie, upload limit</td><td><code>core/flask_app.py</code></td></tr>
<tr><td>Routes</td><td><code>routes/</code> (<code>main.py</code>, <code>upload.py</code>, <code>exploration.py</code>, <code>preprocess.py</code>, <code>modeling.py</code>, <code>prediction.py</code>, <code>assets.py</code>, <code>guided.py</code>)</td></tr>
<tr><td>Session store, error handlers, constants</td><td><code>core/state.py</code>, <code>core/errors.py</code>, <code>core/constants.py</code></td></tr>
<tr><td>Training</td><td><code>python_scripts/app_model_training.py</code>, model lookup in <code>python_scripts/model_registry.py</code></td></tr>
<tr><td>Exploration and inference</td><td><code>python_scripts/app_exploration.py</code>, <code>python_scripts/app_prediction.py</code></td></tr>
<tr><td>Preprocessing and plotting</td><td><code>python_scripts/preprocessing/</code>, <code>python_scripts/plotting/</code>; <code>apply_plot_style()</code> in <code>plot_style.py</code> sets one style for training and export figures</td></tr>
<tr><td>Bundle and run record</td><td><code>python_scripts/reporting/</code></td></tr>
<tr><td>Paths and URL prefix</td><td><code>python_scripts/config.py</code></td></tr>
</tbody></table>
</details>

<h2 id="access">Accessibility</h2>
<p>The guided page is built to WCAG 2.2 at level AA (World Wide Web Consortium, 2024). The page has landmarks, a visible label on every control, errors announced through an assertive live region and progress through a polite one, a visible focus ring, focus moved to each step's heading when the step opens, table headers with <code>scope</code>, a skip link, and no time limit. No meaning is carried by color alone, so each data-check severity is also written as a word. The page is plain HTML and one script, so the accessibility tree is the document tree. <code>tests/integration/test_browser_guided.py</code> runs axe-core in Chromium under the WCAG A and AA rule sets, walks the page by keyboard, and reads a typed hyperparameter back from the fitted estimator. These checks are automated, and no screen-reader session with an assistive-technology user has been run. <code>docs/ACCESSIBILITY.md</code> states what is checked and what is not claimed.</p>
<p>The classic page uses <code>role="navigation"</code>, <code>role="status"</code>, and labeled <code>role="region"</code> sections, live regions for updates (<code>polite</code>) and errors (<code>assertive</code>), <code>aria-label</code> on controls without visible text, <code>aria-describedby</code> for help text, <code>aria-required</code> on required fields, and <code>aria-invalid</code> on a field that fails validation. A skip link appears on focus, focus moves to the active tab and to error messages, and the tab order follows the page order.</p>

<h2 id="cite">Version and citation</h2>
<p>This guide describes DiGiTerra 0.1.0, released 2026-09-24, at commit <code>c6b9526</code> on branch <code>audit-fixes</code>. The software is released under the MIT license. <code>CITATION.cff</code> in the repository holds the citation: DiGirolamo, P. E., Terra, R. R., Gulliver, D. M., and Trun, N. J. (2026), <em>DiGiTerra: Tabular Machine Learning in the Browser with scikit-learn</em> (Version 0.1.0), <a href="https://github.com/rowanterra/DiGiTerra">https://github.com/rowanterra/DiGiTerra</a>. Please also cite scikit-learn (Pedregosa et al., 2011) for the estimators, preprocessing steps, and metrics, and Gorman et al. (2014) for the penguin data. <code>THIRD_PARTY_NOTICES.md</code> lists every runtime dependency with the dependency's version and license, including Flask (Pallets, 2026).</p>
<p></p>
</section>
"""

VERIFY = f"""
<section class="main-content" id="pane-verify" role="tabpanel" aria-labelledby="tab-verify" hidden>

<p>Each count below is read from a record file in the repository at commit <code>c6b9526</code>, and the file is named with the count. The examples on the first tab were run on 2026-09-25 in a Linux x86_64 container with Python 3.11.15, NumPy 1.26.4, pandas 2.3.3, scikit-learn 1.9.0, SciPy 1.17.1, and skops 0.14.0, the versions that the <code>environment</code> section of the penguin run's manifest records.</p>

<div class="verification">
<table class="plain">
<thead><tr><th>Check</th><th class="num">Result</th><th>Criterion</th><th>Record</th></tr></thead>
<tbody>
<tr><td>Registered estimators that train from an empty form</td><td class="num">67 of 69</td><td>Fit completes through the training route</td><td><code>docs/every_model_sweep_2026-09-24.json</code></td></tr>
<tr><td>Fitted estimators that reload with the same predictions</td><td class="num">66 of 67</td><td>Predictions for the first eight rows equal before and after the round trip, compared exactly</td><td><code>docs/artifact_roundtrip_2026-09-24.json</code></td></tr>
<tr><td>Differential test E1, DiGiTerra cells classed other</td><td class="num">0 of 530</td><td>Relative difference from the arbiter below 1e-6, or the arbiter value rounded to the printed decimals</td><td><code>docs/differential_test_E1/RESULTS.csv</code></td></tr>
<tr><td>Probes of the adversarial check that pass</td><td class="num">10 of 10</td><td>Pass condition stated per probe in the record</td><td><code>docs/adversarial_check_2026-09-25.json</code></td></tr>
</tbody></table>
</div>

<h2 id="e1">Differential test E1</h2>
<p>E1 is a black-box differential test. The same fixed train and test files go into DiGiTerra and four programs written by other people, and every printed value is compared with a plain scikit-learn 1.9.0 script (<code>scripts/oracle.py</code>) that serves as the arbiter. The arbiter is not assumed correct, and the independent engines in R 4.5.1, Excel 16.03, and Altair AI Studio 2026.0.1 are what give a disagreement weight. The diabetes file (Efron et al., 2004) was split once into 353 training and 89 test rows, and the iris file (Fisher, 1936) into 120 and 30, with the split written to disk so that no program performs a separate split, scaling, imputation, or search.</p>
<p>Five cases were run. D1 and D2 are descriptive statistics of the two files. R1 is ordinary least squares on the diabetes training rows, compared on the intercept, ten coefficients, test RMSE, MAE, and R<sup>2</sup>, and the 89 test predictions. C1 is k-nearest neighbors on iris with k of 5, compared on the 30 predicted classes. C2 is multinomial logistic regression on iris, compared on the classes and the 90 class probabilities; R and AI Studio fit a differently regularized model and were excluded from C2 by plan, and Excel has no classifier.</p>
{figure("fig/fig_e1_agreement.png", "Horizontal bar charts of compared cells by program and case, with fills for same, last digit, rounded to six decimals, near zero, and other", "E1 agreement by program and case. Each bar counts the compared cells, and the fill marks the class. Source: <code>docs/differential_test_E1/RESULTS.csv</code>.")}
<table class="plain">
<thead><tr><th>Program</th><th class="num">Cells</th><th class="num">Same</th><th class="num">Last digit</th><th class="num">Rounded 6 dp</th><th class="num">Near zero</th><th class="num">Other</th></tr></thead>
<tbody>
<tr><td>DiGiTerra 0.1.0</td><td class="num">530</td><td class="num">457</td><td class="num">7</td><td class="num">56</td><td class="num">10</td><td class="num">0</td></tr>
<tr><td>R 4.5.1</td><td class="num">448</td><td class="num">428</td><td class="num">0</td><td class="num">0</td><td class="num">20</td><td class="num">0</td></tr>
<tr><td>Excel 16.03</td><td class="num">446</td><td class="num">426</td><td class="num">0</td><td class="num">0</td><td class="num">20</td><td class="num">0</td></tr>
<tr><td>Orange 3.39.0</td><td class="num">630</td><td class="num">430</td><td class="num">0</td><td class="num">0</td><td class="num">20</td><td class="num">180</td></tr>
<tr><td>AI Studio 2026.0.1</td><td class="num">382</td><td class="num">362</td><td class="num">0</td><td class="num">0</td><td class="num">20</td><td class="num">0</td></tr>
</tbody></table>
<p>In R1 the largest relative difference from the arbiter was 3.6e-11 for DiGiTerra (R<sup>2</sup> printed to ten decimals), 5.1e-14 for R, 2.0e-14 for Excel, 2.1e-14 for Orange, and 3.8e-13 for AI Studio. All 91 DiGiTerra cells in C2, the class row and the 90 probabilities, are classed same. The 180 cells classed other are Orange's C2 probabilities over two runs, which differ from the arbiter by up to 7.9e-3 while all 30 predicted classes agree. Orange 3.39.0 bundles scikit-learn 1.5.2 on arm64, and the difference is attributed to platform arithmetic under the solver's loose default tolerance. That attribution is inferred and has not been reproduced under Orange's own Python.</p>
<p>E1 found three defects in the version under test, and each was fixed before the counts above were produced. Regression metrics were rounded to three decimals before the decimal-places setting applied. Descriptive statistics were rounded to four decimals, so the centered diabetes means printed as -0.0. For nineteen classifiers, a hyperparameter typed on the Advanced form did not reach the estimator while the run record still recorded the request, so KNeighbors with <code>n_neighbors</code> 3 was fitted with 5. <code>tests/unit/test_decimal_setting.py</code> and <code>test_knn_classifier_form_values_reach_the_estimator</code> cover the fixes, and the pre-fix outputs are kept in <code>digiterra_outputs_pre_fix_99dea69/</code>.</p>

<h2 id="sweep">Estimator sweep and artifact round trip</h2>
<p>The registry names 69 entries. <code>scripts/train_every_model.py</code> trains each entry on a matching example file through the function that the training route calls, with the hyperparameters that an empty form submits. The record of 2026-09-24 (scikit-learn 1.9.0, Python 3.11) shows 67 entries trained: 32 regression, 23 classification, and 12 clustering. <code>Polynomial</code> and <code>Perceptron</code> are registered without a train function and were skipped. The recorded training times sum to 48.7 minutes.</p>
<p><code>scripts/check_artifact_roundtrip.py</code> writes each of the 67 fitted estimators with skops 0.14.0 under the type allowlist, reads the artifact back, and compares the predictions for the first eight rows exactly. Sixty-six estimators returned equal predictions. <code>birch</code> cannot be written under the allowlist, and the application reports that at export instead of writing an artifact that needs an unlisted type to load. The record stores the dataset hash, seed, artifact hash, and both prediction vectors, and <code>--only &lt;model&gt;</code> reruns one estimator against the stored numbers.</p>
{figure("fig/fig_sweep_roundtrip.png", "Bar charts of estimators that train with an empty form by task, and estimators whose predictions are equal after the round trip", "Left, estimators that train with an empty form by task. Right, estimators whose predictions for the first eight rows are equal after a write to the bundle format and a read back. Sources: <code>every_model_sweep_2026-09-24.json</code>, <code>artifact_roundtrip_2026-09-24.json</code>.", "75%")}
<p>Together the two records bound the claim about the registry. The application trains 67 estimators from an empty form and writes and reloads 66 of the 67. The records do not show that every hyperparameter of every estimator has been exercised, and the nineteen-classifier finding of E1 is the reason that claim is not made.</p>

<h2 id="adversarial">Adversarial check</h2>
<p><code>scripts/adversarial_check.py</code> replays the probes of the adversarial review of 2026-09-24 and prints one line per probe: a crafted artifact that names <code>builtins.eval</code>, path traversal on the download routes, a foreign session, formula cells in the workbook export, a CSV with markup in the headers through both pages, cleared fields, a model-change race, Escape on a focused tooltip, figures in a 720-pixel window, and the review's pipeline tests. The record <code>docs/adversarial_check_2026-09-25.json</code> shows 10 of 10 probes passing, and the pipeline probe reports 37 tests passed. The script is run before a release and after any change to the routes, the pipeline, or either page. <code>CHANGELOG.md</code> lists the defects that the review found and the fix for each.</p>
</section>
"""

REFS = """
<section class="main-content" id="pane-refs" role="tabpanel" aria-labelledby="tab-refs" hidden>
<h2 id="references">References</h2>
<ul class="reference-list">
<li>Efron, B., Hastie, T., Johnstone, I., &amp; Tibshirani, R. (2004). Least angle regression. <em>The Annals of Statistics, 32</em>(2), 407&ndash;499. <a href="https://doi.org/10.1214/009053604000000067">https://doi.org/10.1214/009053604000000067</a></li>
<li>Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. <em>Annals of Eugenics, 7</em>(2), 179&ndash;188. <a href="https://doi.org/10.1111/j.1469-1809.1936.tb02137.x">https://doi.org/10.1111/j.1469-1809.1936.tb02137.x</a></li>
<li>Gorman, K. B., Williams, T. D., &amp; Fraser, W. R. (2014). Ecological sexual dimorphism and environmental variability within a community of Antarctic penguins (genus <em>Pygoscelis</em>). <em>PLoS ONE, 9</em>(3), e90081. <a href="https://doi.org/10.1371/journal.pone.0090081">https://doi.org/10.1371/journal.pone.0090081</a></li>
<li>Lei, J., G&rsquo;Sell, M., Rinaldo, A., Tibshirani, R. J., &amp; Wasserman, L. (2018). Distribution-free predictive inference for regression. <em>Journal of the American Statistical Association, 113</em>(523), 1094&ndash;1111. <a href="https://doi.org/10.1080/01621459.2017.1307116">https://doi.org/10.1080/01621459.2017.1307116</a></li>
<li>Liu, F. T., Ting, K. M., &amp; Zhou, Z.-H. (2008). Isolation forest. In <em>Proceedings of the 2008 Eighth IEEE International Conference on Data Mining</em> (pp. 413&ndash;422). IEEE. <a href="https://doi.org/10.1109/ICDM.2008.17">https://doi.org/10.1109/ICDM.2008.17</a></li>
<li>Pallets. (2026). <em>Flask</em> (Version 3.1.3) [Computer software]. <a href="https://flask.palletsprojects.com">https://flask.palletsprojects.com</a></li>
<li>Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., &amp; Duchesnay, &Eacute;. (2011). Scikit-learn: Machine learning in Python. <em>Journal of Machine Learning Research, 12</em>, 2825&ndash;2830.</li>
<li>skops developers. (n.d.). <em>skops</em> (Version 0.14.0) [Computer software]. <a href="https://github.com/skops-dev/skops">https://github.com/skops-dev/skops</a></li>
<li>World Wide Web Consortium. (2024). <em>Web Content Accessibility Guidelines (WCAG) 2.2</em> (W3C Recommendation, 12 December 2024). <a href="https://www.w3.org/TR/WCAG22/">https://www.w3.org/TR/WCAG22/</a></li>
</ul>
</section>
"""

SCRIPT = """
<script>
(function () {
  var order = ['tab-guide', 'tab-verify', 'tab-refs'];
  var tabs = { 'tab-guide': 'pane-guide', 'tab-verify': 'pane-verify', 'tab-refs': 'pane-refs' };
  function show(id, focus) {
    order.forEach(function (t) {
      var on = t === id;
      var tab = document.getElementById(t);
      tab.setAttribute('aria-selected', on);
      tab.tabIndex = on ? 0 : -1;
      document.getElementById(tabs[t]).hidden = !on;
    });
    if (focus) { document.getElementById(id).focus(); }
    window.scrollTo(0, 0);
  }
  order.forEach(function (t, i) {
    var tab = document.getElementById(t);
    tab.addEventListener('click', function () { show(t); });
    tab.addEventListener('keydown', function (e) {
      var next = null;
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { next = order[(i + 1) % order.length]; }
      else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { next = order[(i - 1 + order.length) % order.length]; }
      else if (e.key === 'Home') { next = order[0]; }
      else if (e.key === 'End') { next = order[order.length - 1]; }
      if (next) { e.preventDefault(); show(next, true); }
    });
  });
  // a link to an anchor in another pane opens that pane first
  document.addEventListener('click', function (e) {
    var a = e.target.closest('a[href^="#"]');
    if (!a) { return; }
    var el = document.getElementById(a.getAttribute('href').slice(1));
    if (!el) { return; }
    var pane = el.closest('[role=tabpanel]');
    if (pane && pane.hidden) {
      e.preventDefault();
      show(pane.getAttribute('aria-labelledby'));
      el.scrollIntoView();
    }
  });
  show(order[0]);
}());
</script>
</body>
</html>
"""

out = HERE.parent / "digiterra_guide_2026-09-25.html"
out.write_text(HEAD + GUIDE + VERIFY + REFS + SCRIPT)
print(out, out.stat().st_size)
