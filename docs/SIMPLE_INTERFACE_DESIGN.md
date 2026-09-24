# The guided page: design note, 2026-09-24

## What was surveyed

Eleven interfaces were read from their current documentation on 2026-09-24: Orange, KNIME Analytics Platform, RapidMiner AI Studio, H2O-3 Flow, Amazon SageMaker Canvas, Azure Machine Learning (Automated ML and designer), Google Vertex AI AutoML tabular, Weka Explorer, JASP, jamovi and DataRobot. The full notes with URLs are in `docs/competitor_survey_2026-09-24.txt`. The points that decide the design:

- Steps from a CSV to a fitted model run from four (Weka, JASP, DataRobot) to nine (Azure designer). Every node-graph tool (Orange, KNIME, Azure designer) makes the user assemble the pipeline before anything is fitted.
- A real novice and expert split exists in two products: RapidMiner Auto Model with "Open Process", and DataRobot's modeling modes with "Show advanced options". H2O Flow groups parameters as standard, Advanced and Expert on one form. The rest offer collapsible option groups or a syntax mode.
- Hyperparameters are editable per model in the desktop tools (Orange, KNIME, Weka, JASP, DataRobot's Advanced Tuning) and not in the cloud AutoML products (Canvas, Azure AutoML, Vertex).
- Data checks before training: Canvas (leakage, imbalance, missing, duplicates, anomalies), DataRobot (leakage, outliers, disguised missing), Azure guardrails (imbalance, missing, cardinality), RapidMiner (correlation with target, ID-ness, stability, missing). The free desktop tools show missing-value counts and nothing about leakage or duplicates.
- Run records: seeds are exposed by Orange, H2O, Weka, JASP and DataRobot; versions and model cards by the cloud vendors and DataRobot. No product read states that it records a hash of the input data or the versions of the installed packages.
- Accessibility: jamovi is the one free tool with a keyboard and screen reader commitment (NVDA and JAWS) and a VPAT; JASP's VPAT says screen readers are not supported and keyboard coverage is partial; the cloud vendors point to company-wide conformance reports; Orange, KNIME, Weka, RapidMiner, H2O and DataRobot state nothing.
- Cost and place: Canvas $1.90 per hour, Vertex $21.25 per node hour, Azure free of service charge but billed for compute; all three cloud only. The free tools are local desktop applications; H2O Flow and jamovi Cloud run in a browser.

## Where DiGiTerra already differs

DiGiTerra is free, runs in a browser against a local server or a hosted one, and has no node graph. It writes a run record that names the SHA-256 of the uploaded bytes, the seed, the request as sent, the estimator parameters, the metrics, the data-check findings and the version of every installed package; none of the eleven products read states that it records the data hash or the package versions. Its data checks (target duplicate, one-to-one categorical, constant and identifier columns, missingness, duplicate rows, class imbalance, small row count) match the cloud vendors' lists and are absent from the free desktop tools. Every hyperparameter is editable, which the cloud AutoML products do not offer. What DiGiTerra lacked, against the survey, is a real novice and expert split, an accessibility commitment, and a page whose steps a screen reader can follow.

## The design

One page, `/guided`, one column, one direction. A two-state control at the top, `Guided` and `Full`, implemented as a radio group so it is announced as a choice and not as an on/off switch; the choice persists in the browser. Guided shows five steps and hides everything else: upload, choose the target, choose the model family (or let the page pick from the target's type), run, read the result. Full shows the same five steps with every setting the training route accepts opened in place: split method and column, test size, stratification, scaler, missing and zero handling, feature selection, outlier handling, cross-validation, hyperparameter search, seed, decimal places, and a hyperparameter table for the chosen model. The hyperparameter table is filled from the estimator itself through a new route, `/model-defaults/<name>`, so the names and defaults cannot drift from scikit-learn; for the models whose trainers use their own argument names, the page says so and links to the full page rather than offering fields that would not reach the estimator.

Accessibility targets WCAG 2.2 AA. Landmarks (`header`, `main`, `nav`, `section` with headings), every control labelled by a visible `label`, errors announced through `aria-live="assertive"` and progress through `aria-live="polite"`, a visible focus ring, focus moved to each step's heading when it becomes available, no colour-only meaning (each data-check severity is also a word), tables with `scope` on headers, a skip link, and no time limit. Everything works from the keyboard alone: tab order follows the steps, Enter runs the current step, and the model list is a native `select`. The page is plain HTML and one script; it uses no framework, so the accessibility tree is the DOM.

The existing page stays as it is and is reachable from the new one; nothing in the routes changes for it.

## What the page does not do

It does not add estimators or checks; it exposes what the routes already do. It does not replace the AutoML mode. It does not claim conformance: an accessibility statement in the documentation says what was tested (keyboard walk, VoiceOver on macOS, axe-core in the browser test), and the VPAT is left for when someone with assistive technology has used it.
