---
title: 'DiGiTerra: An interactive interface for machine learning model development'
tags:
  - Python
  - machine learning
  - graphical user interface
  - reproducibility
  - tabular data
authors:
  - name: Paige E. DiGirolamo
    orcid: 0009-0007-9028-3401
    equal-contrib: true
    affiliation: 1
  - name: Rowan R. Terra
    orcid: 0009-0001-3890-705X
    equal-contrib: true
    corresponding: true
    affiliation: "2, 3, 4"
  - name: Djuna M. Gulliver
    orcid: 0000-0003-4219-5849
    affiliation: 3
  - name: Nancy J. Trun
    orcid: 0000-0003-0149-3874
    affiliation: 2
affiliations:
  - name: Stevens Institute of Technology, Hoboken, NJ, United States
    index: 1
  - name: Duquesne University, Pittsburgh, PA, United States
    index: 2
  - name: United States Department of Energy, National Energy Technology Laboratory, Pittsburgh, PA, United States
    index: 3
  - name: Oak Ridge Institute for Science and Education, Oak Ridge, TN, United States
    index: 4
date: 3 October 2026
bibliography: paper.bib
---

# Summary

DiGiTerra is a graphical interface for data exploration and machine learning on tabular data, written for researchers who need a model but do not write code. A user uploads a CSV, Excel, or Parquet file, examines it through descriptive statistics and figures, selects columns, applies preprocessing, and trains a regression, classification, or clustering model. The application registers 69 estimators from scikit-learn, 33 for regression, 24 for classification, and 12 for clustering, and every hyperparameter an estimator exposes remains editable. A fitted model can be applied to new rows, and each run writes a record carrying the input hash, random seed, requested settings, metrics, and the versions of Python and the core numerical packages, so a result can be reproduced or audited after the session ends. DiGiTerra runs in a browser, build scripts package it as a desktop application for macOS, Windows, and Linux, and the source is released under the MIT license.

![One DiGiTerra run. The left lane is what the user does, the right lane is what the application does, and the band behind each group of steps marks the page it belongs to. A blocking data check stops the run before a model is fitted.](DiGiTerra_Figure1.png)

# Statement of need

Machine learning methods are widely adopted in scientific research [@kapoor2024reforms], yet a research group rarely holds both the data and the expertise to apply the right method, and whether experimentalists can use these algorithms independently is still an open question [@greener2022guide]. Interfaces that require no code can lower that barrier, but the existing options force a trade between cost, approachability, and control. A second gap is provenance: reporting guidance for machine-learning-based science asks authors to describe the source of the data, the train-test split or cross-validation folds, and every model trained [@kapoor2024reforms]. An interface that trains a model without recording those facts leaves its user unable to meet that standard.

DiGiTerra is a free, open-source interface that needs no node graph and no command line, explains each setting in plain language, keeps every hyperparameter editable, and writes a machine-readable record of each run.

# State of the field

Eleven no-code and low-code interfaces were read from their documentation on 24 September 2026 (`docs/competitor_survey_2026-09-24.txt`). Amazon SageMaker Canvas runs in the AWS cloud and bills for each hour a workspace is in use [@aws2025canvas], and Azure Machine Learning bills for the compute a run uses [@microsoft2025azureml]. RapidMiner AI Studio is now a Siemens product [@siemens2026rapidminer]. H2O-3 offers a no-code web interface, Flow, which its documentation marks as deprecated in favor of the Python and R clients [@h2o2026h2o3]. KNIME composes an analysis from processing nodes that pass data to each other [@fillbrunn2017knime], and Orange pairs a scripting library with a visual programming platform [@demsar2013orange]; in both, the analysis is a workflow assembled from components. User interface design is fundamental to the success of interactive machine learning for non-experts [@dudley2018review], and the first Weka was already built for the domain specialist [@holmes1994weka].

Three gaps recur. Only RapidMiner and DataRobot offer a distinct novice path and expert path. SageMaker Canvas and Azure Machine Learning offer no per-model hyperparameters in their no-code interfaces, and the desktop tools that do (Orange, KNIME, Weka, JASP) document no leakage, duplicate-row or class-imbalance check before training. No product read states that its run record includes a hash of the input data or the installed package versions, and among the free tools jamovi alone documents screen reader support.

# Software design

Preprocessing covers missing values and zeros, one-hot encoding of text columns, stratified, time-ordered and grouped holdouts, and scaling. When no model is named on the AutoML page, a regression or classification candidate set is scored by five-fold cross-validation on the same folds over three seeds (one above 5,000 rows) and the best is trained; the fold scores are drawn as one box per candidate, so a winner separated from the runner-up by less than the fold-to-fold spread is shown as such. On the inference page a figure sets the spread of every numeric predictor in the training rows against the rows being predicted.

Before a run starts, data checks look for a feature that duplicates the target, a categorical column that maps one to one onto the class, constant and identifier columns, heavy missingness, duplicate rows, class imbalance, and too few rows for the split. A blocking finding stops the run; a user who overrides one must write a justification, and the run is then labeled as not valid for decision use. Each fitted regression or classification model is scored against a baseline, the training mean or the most common class, on the same validation rows.

The interface ships in two forms on the same routes. The classic page holds Simple, Advanced and AutoML modes. The guided page (`/guided`) is four steps, one on screen at a time, with a plain-language explanation on every control and metric and on the common hyperparameters, and a Full choice that opens a hyperparameter table filled from the estimator's own signature, so names and defaults cannot drift from scikit-learn. The page is built to WCAG 2.2 AA, and a Chromium test runs axe-core under the A and AA rule sets with no serious violation allowed; `docs/ACCESSIBILITY.md` states what is checked and what is not claimed.

Every run writes a manifest and a model card. The manifest records the SHA-256 of the uploaded bytes, the request as submitted, the seed, the estimator and its parameters, preprocessing and split, metrics, baseline, data-check findings, and the versions of Python, DiGiTerra and the numerical packages. Models are stored with skops rather than pickle, so loading builds only allowlisted types, and each installation signs its bundles with an HMAC-SHA256 key that import verifies. Command-line scripts score and rerun a bundle without the interface.

The application is written in Python 3.11 and 3.12 [@psf2026python], with HTML [@whatwg2026html] and JavaScript [@ecma2024ecmascript] on the front end, and Flask 3.1.3 [@pallets2026flask] with Werkzeug 3.1.8 [@pallets2026werkzeug] serves it. Scikit-learn 1.9.0 [@pedregosa2011sklearn] supplies the estimators, preprocessing steps, and metrics; NumPy 1.26.4 [@harris2020numpy], pandas 2.3.3 [@mckinney2010pandas], and SciPy 1.17.1 [@virtanen2020scipy] handle arrays, frames, and statistics; matplotlib 3.11.1 [@hunter2007matplotlib] and seaborn 0.13.2 [@waskom2021seaborn] draw the figures; SHAP 0.49.1 [@lundberg2017shap] produces the model explanations; skops 0.14.0 [@skops2026] writes the artifacts. A Flask application was chosen over a desktop toolkit so that one code base serves a researcher's own machine and a hosted instance behind an institutional proxy. When DiGiTerra runs as a hosted server, uploaded files are parsed in a separate process under a memory ceiling and a wall clock.

# Verification

Three records in the repository state what the software does with numbers. A sweep trains every registered estimator on a matching example dataset through the function the web route calls, with the defaults an empty form submits (16 models also need a value for a required field): 69 of 69 entries train. A second script writes and reloads every estimator through the bundle format and compares predictions: 69 of 69 round-trip identically, Birch through a centroid artifact because skops refuses its fitted tree.

The third record is a black-box differential test. Shuffled copies of the iris and diabetes example files were split once into fixed training and test sets (120 and 30 rows; 353 and 89 rows) and run with the same predictors and settings in DiGiTerra, Orange 3.39.0, RapidMiner AI Studio 2026.0.1 and R 4.5.1, the reference programs twice each, with a plain scikit-learn script as the arbiter. For ordinary least squares the ten coefficients and all 89 test predictions agree across the four programs at six significant figures, and the intercept agrees in the three programs that export it; for k-nearest neighbors all 30 predicted classes agree; for logistic regression DiGiTerra reproduces scikit-learn's class probabilities to within 3e-16, 71 of 90 bit for bit. The plan, case files, every program's saved output and the comparison script ship in `docs/differential_test_E1/`. The test found three defects in the version under test, among them a hyperparameter typed on the Advanced form that never reached nineteen classifiers. An adversarial review of the routes, the pipeline and the browser side found 25 further defects, each reproduced before it was fixed and now guarded by a regression test or a replayed probe; 17 of the 29 faults they itemize changed a reported number, and the changelog lists them.

A unit and integration suite runs on Linux and macOS across both supported Python versions. Ten example datasets ship in the repository, among them the iris measurements of @fisher1936iris and the Palmer Archipelago penguin data of @gorman2014penguins.

# Research impact statement

The Department of Energy's National Energy Technology Laboratory hosts DiGiTerra on its internal Energy Data eXchange platform (EDX) as a laboratory-approved executable for staff who model geochemical and materials data without writing code. EDX is not reachable from outside the laboratory, so the deployment cannot be linked here; the Phase 2 application under which it was accepted into the EDX Incubator ships in the repository (`docs/EDX_Incubator_Phase2_Application_Answers.pdf`) and records the requested hosting arrangement and the initial user group. The hosted copy is built from the same repository as the public release and writes a run record with the same fields.

# AI usage disclosure

Claude Opus 5 [@anthropic2026claudeopus5] and Cursor [@anysphere2026cursor] were used to audit and assist with the user interface code. Their changes were checked by the test suites and the verification records above.

# Author contributions

Paige E. DiGirolamo and Rowan R. Terra contributed to this project equally.

# Acknowledgements

Appreciation is extended to the Department of Energy, National Energy Technology Laboratory for agreeing to host DiGiTerra on their internal Energy Data eXchange platform (EDX). This research was supported in part by an appointment at the National Energy Technology Laboratory (NETL) through the Professional Internship Program (PIP), sponsored by the U.S. Department of Energy and administered by the U.S. Department of Energy Oak Ridge Institute for Science and Education (ORISE).

This article was prepared as an account of research sponsored by an agency of the United States Government. Neither the United States Government nor any agency thereof, nor any of their employees, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights. Reference therein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof. The views and opinions of authors expressed therein do not necessarily state or reflect those of the United States Government or any agency thereof.

# References
