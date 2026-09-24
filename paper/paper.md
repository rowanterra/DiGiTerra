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
date: 24 September 2026
bibliography: paper.bib
---

# Summary

DiGiTerra is a graphical interface for data exploration and machine learning on tabular data, written for researchers who need a model but do not write code. A user uploads a CSV, Excel, or Parquet file, examines it through descriptive statistics and figures, selects predictor and target columns, applies preprocessing, and trains a regression, classification, or clustering model. The application registers 67 trainable estimators from scikit-learn, 32 for regression, 23 for classification, and 12 for clustering, and every hyperparameter an estimator exposes remains editable. A fitted model can be applied to new rows, and each run writes a record carrying the input hash, random seed, requested settings, metrics, and installed package versions, so a result can be reproduced or audited after the session ends. DiGiTerra runs in a browser or as a desktop application on macOS, Windows, and Linux, and the source is released under the MIT license.

![One DiGiTerra run. The left lane is what the user does and the right lane is what the application does; the band behind each group of steps marks the page of the interface it belongs to. Rectangles are steps and diamonds are decisions. A blocking data check stops the run before a model is fitted, a run with no free training slot is queued rather than refused, and a prediction file whose columns do not match the training columns is refused.](DiGiTerra_Figure1.png)

# Statement of need

Machine learning methods are being widely adopted for scientific research [@kapoor2024reforms], yet a research group rarely holds both the data and the expertise to apply the right method to it, and whether experimentalists will be able to use these algorithms independently is still an open question [@greener2022guide]. Interfaces that require no code can lower that barrier, but the existing options force a trade between cost, approachability, and control, and a second gap is provenance. Reporting guidance for machine-learning-based science asks authors to describe the source of the data, the train-test split or cross-validation folds, and every model trained, and to justify that comparisons are against appropriate baselines [@kapoor2024reforms]. An interface that trains a model without recording those facts leaves its user unable to meet that standard.

DiGiTerra provides a free, open-source interface that needs no node graph, no command line, and no prior machine learning knowledge, keeps every hyperparameter editable, and writes a machine-readable record of each run. The workflow moves in one direction, from upload through evaluation, which narrows the ways a run can go wrong.

# State of the field

Amazon SageMaker Canvas bills by the hour its workspace runs, with further charges for data processing, training and prediction [@aws2025canvas], and Azure Machine Learning, which carries no service charge of its own, bills the compute a run consumes by the second [@microsoft2025azureml]. RapidMiner AI Studio is now a Siemens product [@siemens2026rapidminer]. Open-source alternatives carry a different cost. H2O-3 reaches its models through the Flow notebook and web interface [@h2o2026h2o3], which assumes more comfort with code than a novice has. KNIME composes an analysis from processing nodes that pass data to one another [@fillbrunn2017knime], and Orange pairs a Python scripting library with a visual programming platform [@demsar2013orange]; in both, the user designs the graph before a model is fitted. Interface design decides whether such a system works for a non-expert at all, and getting it right is not trivial [@dudley2018review]. A workbench built for the domain specialist rather than the machine learning expert is not a new idea, and goes back to the first Weka [@holmes1994weka]. Several existing interfaces also restrict hyperparameter access, which limits how far a user can take a model once the defaults stop serving the question.

# Software design

DiGiTerra reads tabular data from CSV, Excel, and Parquet files. After upload, a user can inspect descriptive statistics, correlation matrices, and pair plots before committing to a model. Preprocessing covers missing values and zeros, one-hot encoding of text columns, stratified train and test splits, and scaling. Time-ordered and grouped holdouts are available for data where a random split would leak information across the boundary.

Modeling is offered at three levels of control. AutoML fits a set of estimators with defaults, Simple exposes the settings most runs need, and Advanced opens hyperparameters, cross-validation, feature selection, and outlier handling. Before a run starts, a set of data checks looks for a feature that duplicates the target, a categorical column that maps one to one onto the class, constant and identifier columns, heavy missingness, duplicate rows, class imbalance, and a row count too small for the requested split. A blocking finding stops the run and states the reason, and a user who overrides one must supply a written justification, after which the run is labeled as not valid for decision use in the result, manifest and model card. Each fitted model is scored against a baseline on the same validation rows, the training mean for regression and the most common class for classification, and the result reports whether the model beat it. A fitted model can be applied to a new dataset when the column names match those used in training.

Every training run writes a manifest and a model card into the session directory. The manifest records the SHA-256 hash of the uploaded bytes, the request as submitted, the random seed, the estimator and its parameters, preprocessing and split, metrics, baseline comparison, data-check findings, and the version of every installed package. Runs also export as a bundle holding the record together with the fitted objects and a hash of the model artifact. Models are stored with skops rather than pickle, so loading a bundle builds only types on an allowlist instead of executing whatever the file describes, and each installation signs its bundles with an HMAC-SHA256 key that import verifies before reading the model. Command-line scripts score a bundle and rerun it, which lets a reviewer check a reported number without opening the interface.

The application is written in Python [@psf2026python], with HTML [@whatwg2026html] and JavaScript [@ecma2024ecmascript] composing the front end, and runs on Python 3.11 and 3.12. Flask 3.1.3 [@pallets2026flask] and Werkzeug 3.1.8 [@pallets2026werkzeug] serve the application. Scikit-learn 1.9.0 [@pedregosa2011sklearn] supplies the estimators, preprocessing steps, and metrics. NumPy 1.26.4 [@harris2020numpy], pandas 2.3.3 [@mckinney2010pandas], and SciPy 1.17.1 [@virtanen2020scipy] handle arrays, frames, and statistics, while matplotlib 3.11.1 [@hunter2007matplotlib] and seaborn 0.13.2 [@waskom2021seaborn] draw the figures and SHAP 0.49.1 [@lundberg2017shap] produces the model explanations. Model artifacts are written with skops 0.14.0 [@skops2026]. A Flask application was chosen over a desktop toolkit so that one code base serves both a researcher's own machine and a hosted instance behind an institutional proxy; in the hosted case the application reads an identity supplied by the proxy rather than implementing its own login. Parsing an uploaded file runs in a separate process under an address-space ceiling and a wall clock. That ceiling holds only where the kernel applies RLIMIT_AS, which at present means Linux and the BSDs; on a platform that will not enforce it, the application refuses the isolated path rather than running without a bound.

# Verification

Three records in the repository state what the software does with numbers. A sweep script trains every registered estimator on a matching example dataset through the same function the web route calls, using the defaults an empty form submits: 67 of 69 registry entries train, and the two skipped entries are registered without a trainer. A second script writes and reloads every estimator through the bundle format and compares predictions: 66 of 67 round-trip identically, and the one that cannot be written safely reports that rather than failing quietly.

The third record is a black-box differential test against independent software. Shuffled copies of the iris and diabetes example files were split once into fixed train and test halves (120 and 30 rows; 353 and 89 rows), and the same halves, predictors and settings were run in DiGiTerra, in Orange 3.39.0, in RapidMiner AI Studio 2026.0.1 and in R 4.5.1, each twice, with a plain scikit-learn script as the arbiter. For ordinary least squares the intercept, the ten coefficients and all 89 test predictions agree across the four programs at six significant figures; for k-nearest neighbours all 30 predicted classes agree; for logistic regression DiGiTerra reproduces scikit-learn's class probabilities exactly. The plan, the case files, every program's saved output and the comparison script ship in `docs/differential_test_E1/`. The test also found three defects in the version under test, all corrected before this submission: regression metrics were rounded to three decimals before the decimal-places setting was applied, descriptive statistics were rounded to four decimals, and for nineteen classifiers a hyperparameter typed on the Advanced form never reached the estimator while the run record still recorded the request. A regression test now covers each.

The repository carries a unit and integration test suite that runs on Linux and macOS across both supported Python versions, including a browser test that drives the page through Chromium and confirms values typed into Advanced controls reach the training request. Ten example datasets ship with the application, four for classification, three for regression, and three feature-only copies for clustering, alongside a two-file pair for the inference page. Eight are static copies of scikit-learn datasets, among them the iris measurements of @fisher1936iris; the penguin files carry the Palmer Archipelago data of @gorman2014penguins, distributed under CC0 through the palmerpenguins package. A step-by-step usage guide with annotated screenshots is in the repository documentation.

# Research impact statement

The Department of Energy's National Energy Technology Laboratory has agreed to host DiGiTerra on its internal Energy Data eXchange platform (EDX), where it is intended for laboratory staff who model geochemical and materials data without writing code. [AUTHORS: add any external users, publications that used DiGiTerra, or integrations before submission; nothing in the repository records one as of 24 September 2026.]


# Author contributions

Paige E. DiGirolamo and Rowan R. Terra contributed to this project equally.

# Acknowledgements

Appreciation is extended to the Department of Energy, National Energy Technology Laboratory for agreeing to host DiGiTerra on their internal Energy Data eXchange platform (EDX). This research was supported in part by an appointment at the National Energy Technology Laboratory (NETL) through the Professional Internship Program (PIP), sponsored by the U.S. Department of Energy and administered by the U.S. Department of Energy Oak Ridge Institute for Science and Education (ORISE).

This article was prepared as an account of research sponsored by an agency of the United States Government. Neither the United States Government nor any agency thereof, nor any of their employees, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights. Reference therein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof. The views and opinions of authors expressed therein do not necessarily state or reflect those of the United States Government or any agency thereof.

# References
