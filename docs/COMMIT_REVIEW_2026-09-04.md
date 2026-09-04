# Review 2: the fifteen commits after 16cbcc7

Review only. No file outside this one was changed.

## Scope and baseline

The request named HEAD as `ac9c15c`. HEAD is **`29532c2`**; `ac9c15c` is its parent.
`git log 16cbcc7..HEAD` is **15** commits, not 14. This review covers all fifteen.

    pytest tests/unit    184 passed, 34 warnings, 122.52s
    make lint            exit 0 (ruff check, ruff format --check on 193 files,
                         check_language both standards: nothing to report)

Working tree clean at the time of review.

---

## F1. A stale conformal record turns a classification result into a broken regression one

**Severity: high. Confirmed by running it.**

`python_scripts/preprocessing/uncertainty.py:44` declares the ContextVar.
`python_scripts/preprocessing/run_regression_pipeline.py:660` is the only caller of
`record_conformal`. `python_scripts/app_model_training.py:1071` reads it for every run
of every task.

Nothing clears the variable. A classification or clustering run never sets it, so in a
process where a regression run has already completed **in the same thread**, the
classification run inherits the regression run's half-width.

Confirmed:

    regression run  -> conformal: {'target': 92.51169476614038}   (diabetes)
    classification  -> conformal: {'target': 92.51169476614038}   (penguins)

The harm is not cosmetic. `python_scripts/helpers.py:1012` tests `conformal` **before**
the probability branch at `helpers.py:1030`, so the classifier takes the regression
path:

    interval columns on a CLASSIFICATION prediction: ['lower_90', 'upper_90']
    probability columns: []
    sample: [{'Predicted_Target': 'Gentoo', 'lower_90': nan, 'upper_90': nan}, ...]

The predicted class is a string, so `pd.to_numeric` gives NaN and both interval columns
are NaN, while `probability_<class>` and `max_probability` are suppressed entirely. The
manifest for that run records `"method": "split conformal"` for a classifier.

Scope of exposure. A new thread does **not** inherit a ContextVar; this was tested:

    main thread sees: True
    new thread sees the parent's value: False

`routes/modeling.py:59` and `:120` start each run in `threading.Thread`, so the served
path gets a fresh context per run and is not affected today. The exposure is every
in-process caller: `scripts/train_every_model.py`, `scripts/check_artifact_roundtrip.py`,
`tests/unit/test_pipelines.py`, and any future queue worker that runs jobs in a loop
rather than a thread per job. It is also the exposure if training ever moves into
`core/worker.py`, where the spawn child sets the variable in a different process and the
parent would read `None` and silently record nothing.

**Smallest fix.** Set both records unconditionally at the start of each run rather than
only on the success path. In `app_model_training.py`, immediately before the pipeline is
dispatched, call `set_applicability(None)` and `set_conformal(None)`. Two lines, and the
read at `:1069`/`:1071` then reflects this run alone.

**Test that would have caught it.** None asserts that a classification run carries no
interval. `tests/unit/test_applicability_uncertainty.py` gets a fresh thread per route
call, so it cannot see this.

---

## F2. The training ranges are z-scores, and are published as if they were data units

**Severity: high. Confirmed by running it.**

`python_scripts/preprocessing/applicability.py:79-80` records `feature_min`/`feature_max`
from the matrix passed in. The three pipelines pass the **scaled** matrix
(`run_classification_pipeline.py:234`, `run_clustering_pipeline.py:262`,
`run_regression_pipeline.py:212` all pass `X_train_s`).

For penguins with the standard scaler and the transformer on:

    feature_order: ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
                    'body_mass_g', 'island_Biscoe', 'island_Dream',
                    'island_Torgersen', 'sex_Female', 'sex_Male']
    feature_min:   {'bill_length_mm': -1.954, ...}
    feature_max:   {'bill_length_mm': 2.883, ...}

Two problems follow.

**The units are wrong in published text.** `python_scripts/reporting/bundle.py:264`
prints these verbatim into the model card as "Those ranges are: bill_length_mm -1.954 to
2.883". `docs/documentation.md:82` tells a reader `out_of_range_columns` lists "the
numeric columns whose value for that row falls outside the range seen in training". A
reader looking at a bill length in millimetres is shown a range in standard deviations
with a millimetre name attached. Nothing says the numbers are scaled.

**One-hot columns are named as features.** `island_Biscoe`, `sex_Male` and the rest are
in `feature_order`, so `out_of_range_columns` can name them. A reader is told the value
of `island_Biscoe` is outside its training range, which means nothing to them, and
`docs/documentation.md:82` says the list is "numeric columns", which these are not in any
sense the reader shares.

**Smallest fix.** Keep the detector on the scaled matrix, and record the ranges from the
raw training frame restricted to the columns a person supplied, before the preprocessor
expands them. That is one extra argument to `fit_applicability` and a second dict; the
detector is unaffected. Failing that, the smallest correct change is to state the units
in the card and drop the encoded columns from the range list.

---

## F3. A row of blanks is reported as ordinary

**Severity: medium. Confirmed by running it.**

`applicability.py:69` fills NaN with the column mean at fit time.
`applicability.py:101` fills NaN with `0.0` at score time. In scaled space the mean is
approximately 0, so the asymmetry the request asks about is nearly invisible; that is the
lesser half of the problem.

The substantive half is that filling at all hides the blank. A new row whose every value
is missing:

    all-blank row -> {'novelty_score': -0.512, 'novel_row': False, 'out_of_range_columns': ''}

The threshold for that model was -0.597, so a row with no data at all is reported as
**more** typical than the first percentile of the training rows, and names no
out-of-range column, because `applicability.py:108` skips NaN when comparing to the
range. A prediction made from nothing is reported as resting on familiar data.

**Which fill is right: neither.** A blank in a new row is not a value near the mean and it
is not zero. It is the absence of the measurement the model needs.

**Smallest fix.** In `score_rows`, compute a per-row missing count before filling, and set
`novel_row` true when any required feature is missing, independently of the score. One
boolean OR at `applicability.py:103`. Naming the missing columns in
`out_of_range_columns` would be better still but is a larger change.

---

## F4. The bundle format string did not change when the manifest gained two keys

**Severity: low. Confirmed by inspection and by loading a bundle.**

`python_scripts/reporting/bundle.py:47` still declares `BUNDLE_FORMAT =
"digiterra-bundle/2"`. `bundle.py:293` and `:294` added `applicability` and
`uncertainty` as top-level manifest keys in `c8272ae` and `4b39424`. A bundle written
before those commits and one written after both claim format 2 and have different
schemas.

Nothing raises. `routes/assets.py:146-147` applies the artifact with
`store[key] = objects.get(key)`, so an older artifact sets both to `None`;
`_applicability_record` and `_uncertainty_record` return `None` or the no-method form,
and the card renders its "recorded no applicability measure" branch. A run with no
applicability record imports cleanly. Confirmed: a fresh bundle lists
`['manifest.json', 'model_card.json', 'model_card.html', 'model.skops',
'visualizations.pdf']` with `format: digiterra-bundle/2`.

**Smallest fix.** Either bump to `digiterra-bundle/3` and accept 2 on read, or record a
`manifest_schema` integer beside `format`. The second is smaller and does not touch the
signature or the artifact.

---

## F5. Stratify guards: four is the whole set. No fifth found

**Refuted, with one caveat.**

The guards are `static/js/preprocess.js:78`, `static/js/modeling.js:201`,
`routes/preprocess.py:99`, and `python_scripts/app_model_training.py:479`. All four make
the classification exemption. Cluster is exempt in all four by the pre-existing
`cluster_models` and `outputType !== 'Cluster'` tests.

Caveat on the AutoML path. `app_model_training.py:470` derives the task from
`MODEL_REGISTRY.get(modelName)`. For a simple or advanced run `modelName` is a registry
key and this is right. The AutoML path was not exercised in this review, and if it ever
reaches that line with a `modelName` that is not a registry key, `registry_entry` is
`None`, `is_classification` is `False`, and an AutoML classification stratified on its
target would be refused with a message about leakage. Worth one test rather than a code
change.

---

## F6. A person's edit is overwritten by the next detection

**Severity: medium. Confirmed by driving the page.**

`static/js/preprocess.js` `runPreprocessDetection` calls `setPreprocessDefault` for
`missScopeAll`, `imputeStrategy`, `useTransformer` and `transformerColumn` on every run,
and `initPreprocessPage` binds it to `change` on `indicators`, `predictors` and
`outputType`.

Driven on penguins: open the cleaning controls, set the action for blanks to `mean`, then
edit the indicator columns.

    imputeStrategy edited to 'mean'; after editing indicators it is 'none'

The person's decision is discarded without a word. The tutorial's step 4 invites exactly
this edit.

**Smallest fix.** Record which ids the prefill and the detection have written, and in
`setPreprocessDefault` return early when the current value differs from the last value
this code wrote. About six lines, no change to the markup.

The other half of item 6 is sound: the prefill never fires for a person's own file. The
listener is registered inside the Try an Example handler only, and
`tests/integration/test_browser_advanced.py::test_a_persons_own_file_is_not_prefilled`
asserts it.

---

## F7. Model Preprocessing and the inference results carry leftovers

**Severity: low, all confirmed by driving the page.**

Headings visible in the form, in DOM order:

    ['Columns and task', 'Transformer', 'More settings', 'Scaling', 'Seed',
     'Train/Test Split', 'Data cleaning', 'Split, scaling and seed', 'Stratify']

- **Orphan `Transformer` heading**, `templates/index.html` inside the first
  `.preprocess-card`. `c82b244` moved `transformerOptionWrapper` and `transformerYes`
  into the hidden `#preprocessControls`, leaving the heading behind.
- **Two shell panels.** Their entire visible text is a heading and a stranded fragment:
  `'Data cleaning Check selected columns for blanks or zeros:'` and
  `'Split, scaling and seed Stratify'`. Both are cards whose contents moved into
  `#preprocessControls` and `#preprocessMoreSettings`.
- **`More settings` summary is 46px tall** because its `h3` keeps `display: block` and a
  14.4px margin. `static/style.css` sets `margin: 0` under `.doc-section > .doc-summary
  h3`, which does not match `.preprocess-more > .doc-summary`.

**Smallest fix.** Move the three stray headings into the hidden container with the
controls they label, and add `.preprocess-more > .doc-summary h3 { margin: 0; }`.

**The example button never recovers.** After a successful prediction:

    button label after completion: ['Loading the rows...', disabled=True, visible=True]

`static/js/core.js` `initUsePenguinRows` restores the label and the disabled state only in
the `catch`. On success the button stays disabled reading "Loading the rows..." for the
rest of the session, so a second run of the example rows is impossible.
**Smallest fix.** Restore both in a `finally`.

**Two residual sentences** in the inference results, both from before this work:

    'Descriptive statistics for inferred (predicted) values (same style as Data Exploration).'
    "No true labels for new data, so we only show predicted class counts below and the model's test-..."

The first describes the page to itself by comparison with another page. The second uses
first person plural.

---

## F8. What the two test files actually assert

**`tests/unit/test_applicability_uncertainty.py` asserts behaviour, not execution.** It
checks dtypes, that re-presenting the training rows flags at most two percent, that a
bill of 90 mm is called novel and names `bill_length_mm` while the row beside it is not,
that the interval covers at least 85 percent, and that a bundle imported into a session
that has trained nothing reproduces the scores to 1e-9 and the same width. Setting either
record to `None` turns all four red, which was verified.

Its gap is F1: every test gets a fresh thread through the route, so none can observe a
stale record, and none asserts that a classification run carries no interval.

**`tests/unit/test_stratify_categorical.py` asserts behaviour.** The third test captures
the `stratify` argument `train_test_split` actually receives and compares it to the target
column, which is the observable; the first two check the route accepts classification and
still refuses regression. Removing the pipeline branch turns it red, which was verified.

Its gap: nothing exercises AutoML or advanced mode, which is the caveat in F5.

---

## Standard B violations passed over, not fixed

`make lint` reports nothing because these sit outside the files
`scripts/check_language.py` scans, or inside constructs it skips.

1. `static/js/inference.js`, the inference summary block: "so **we** only show predicted
   class counts" is first person plural in user-facing text. The checker scans JS only for
   the STE standard, not for standard B, so it is not reported.
2. `static/js/inference.js`: "Descriptive statistics for inferred (predicted) values (same
   style as Data Exploration)" explains the page by reference to another page rather than
   saying what the table is.
3. `templates/index.html`, tutorial step 5: "Quantiles and bins are not offered, because
   they cut a continuous column into groups and the species column is text." The
   participle chain is long for procedural text; two sentences would read better.
4. `python_scripts/preprocessing/applicability.py:12` and `uncertainty.py:11`, module
   docstrings: "the novelty score ... is what checks it" and "That proviso is the whole of
   the method's weakness" are closer to argument than description. Correct, but the
   docstring standard elsewhere in this tree is plainer.

---

## Order I would fix these in

1. **F1**, because it silently produces wrong output columns and suppresses right ones.
2. **F2**, because the model card publishes numbers in units it does not name.
3. **F3**, because a prediction from an empty row is reported as trustworthy.
4. **F6**, because it discards a decision the tutorial tells a reader to make.
5. **F7** button restore, then the headings and the summary margin.
6. **F4** and the **F5** AutoML test, which are both small and neither is urgent.
