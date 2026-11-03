from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

import python_scripts.plotting.plot_style  # noqa: F401 - apply soft seaborn-like style
from python_scripts.config import VIS_DIR
from python_scripts.estimator_params import apply_estimator_params
from python_scripts.plotting.bundles import export_plots
from python_scripts.plotting.plot_predictor_spread import write_training_spread
from python_scripts.preprocessing.applicability import record_applicability, set_applicability
from python_scripts.preprocessing.baseline import classification_baseline
from python_scripts.preprocessing.feature_selection import apply_feature_selection
from python_scripts.preprocessing.hyperparameter_search import apply_hyperparameter_search
from python_scripts.preprocessing.outlier_handling import apply_outlier_handling
from python_scripts.preprocessing.split_fill import apply_fill_plan_to_split, get_fill_plan
from python_scripts.preprocessing.splitting import get_split_plan, split_indices
from python_scripts.preprocessing.uncertainty import set_conformal
from python_scripts.preprocessing.utilities import (
    _scale_pairs,
    get_feature_names,
    make_preprocessor,
    make_strat_labels_robust,
    sort_class_labels_numeric_bins,
)

logger = logging.getLogger(__name__)


def _encode_ordinal_target(y_part, label_to_int: dict[str, int]) -> pd.Series:
    """
    Map raw class labels to integers. label_to_int keys are strings (from str(v) over train/test);
    pandas .map(dict) uses raw values as keys, so float 1.0 would miss key '1.0' and become NaN.
    """
    ys = y_part if isinstance(y_part, pd.Series) else pd.Series(np.asarray(y_part).ravel())

    def _one(v):
        if pd.isna(v):
            return np.nan
        k = str(v)
        if k not in label_to_int:
            raise ValueError(
                f"The target label {k!r} has no ordinal mapping. "
                f"The known labels are {sorted(label_to_int.keys())!r}. "
                "Check the labels for a typo, and for a mix of integer and float types."
            )
        return label_to_int[k]

    return ys.map(_one)


def _normalize_classes(model):
    """Return a 1D array of class labels. Unwrap MultiOutputClassifier single-output list-of-one-array."""
    try:
        c = model.classes_
    except AttributeError:
        return None
    if c is None:
        return None
    if isinstance(c, (list, tuple)) and len(c) == 1:
        single = c[0]
        if hasattr(single, "shape") and getattr(single, "shape", None):
            return np.asarray(single).ravel()
    out = np.asarray(c)
    return out.ravel() if out.ndim > 1 else out


def run_classification(
    model,
    model_name,
    train_data,
    target_variables,
    use_stratified_split,
    X,
    y,
    stratifyColumn,
    units,
    X_scaler_type,
    seed,
    sigfig,
    quantileBinDict,
    useTransformer,
    transformer_cols,
    testSize,
    feature_selection_method="none",
    feature_selection_k=None,
    outlier_method="none",
    outlier_action="remove",
    hyperparameter_search="none",
    search_cv_folds=5,
    search_n_iter=50,
    modeling_mode="simple",
    **kwargs,
) -> dict[str, Any]:
    # Parameters typed on the guided page, by their scikit-learn names.
    model = apply_estimator_params(model)
    # A record must never outlive the run that made it. Both are cleared at the top of
    # the run, so a task that writes neither leaves nothing behind for the caller to
    # read: a classification run following a regression one in the same thread used
    # to inherit its interval.
    set_applicability(None)
    set_conformal(None)
    # Extract modeling_mode from kwargs if provided there (for backward compatibility)
    if "modeling_mode" in kwargs:
        modeling_mode = kwargs["modeling_mode"]

    if target_variables is None:
        raise ValueError("Classification requires an explicit target_col.")

    nump = X.select_dtypes(include=np.number).columns.tolist()
    catp = []
    if useTransformer == "Yes":
        catp = list(transformer_cols)
        # A numeric column chosen for encoding is a category, not a number, so it
        # leaves the numeric list; listing it twice raised a duplicate-column error.
        nump = [c for c in nump if c not in catp]
    textp = []

    _split_source = train_data  # frame with every column, for a split column that is not a feature
    use_cols = nump + catp + textp + list(target_variables)
    # A column the fill plan will impute keeps its NaN rows; every other NaN drops the row.
    fill_plan = get_fill_plan()
    ddf = train_data[use_cols].dropna(subset=fill_plan.dropna_subset(use_cols) if fill_plan else None).copy()

    # target_variables=target_variables[0]
    # Rowan 10/13
    # Strat labels
    if not quantileBinDict["quantile"] == 0:
        logger.debug("Using quantiles for stratification")
        quantiles = quantileBinDict["quantile"]
        train_data, use_strat, counts = make_strat_labels_robust(
            ddf, stratifyColumn, "classification", testSize, quantiles
        )

    elif not quantileBinDict["bin"] == 0:
        logger.debug("Using bins for stratification")

        def parseIntString(s):
            if s.strip() == "inf":
                return np.inf
            else:
                return int(s)

        bins = list(map(parseIntString, quantileBinDict["bin"].split(",")))
        train_data, use_strat, counts = make_strat_labels_robust(ddf, stratifyColumn, "classification", testSize, bins)

    elif stratifyColumn:
        # Neither bins nor quantiles: stratify on the column's own values. The
        # candidate loop below accepts the column only when it has at least two
        # values and at least two rows in each, which is what a split needs.
        train_data, use_strat, _counts = ddf.copy(), True, {}

    else:
        train_data, use_strat, _counts = ddf.copy(), False, {}

    # Split
    strat = None
    if use_strat:
        # Prefer a generic label created by make_strat_labels_robust (commonly "STRAT_LABEL"),
        candidate_cols = []
        for c in ["STRAT_LABEL", "STRAT", "BIN_LABEL", "QUANTILE_STRATIFY"]:
            if c in train_data.columns:
                candidate_cols.append(c)
        if stratifyColumn and stratifyColumn in train_data.columns:
            candidate_cols.append(stratifyColumn)

        for cand in candidate_cols:
            s = train_data[cand]
            vc = s.value_counts(dropna=True)
            if len(vc) >= 2 and (vc.min() >= 2):
                strat = s
                break

    X = ddf[nump + catp + textp].copy()
    # Ensure y is properly formatted - convert DataFrame to Series if single column
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 1:
            # ddf dropped rows with missing values; y must follow or the split
            # rejects the length mismatch.
            y = y.iloc[:, 0].loc[ddf.index]
        # For multi-column DataFrame, keep as is but ensure index alignment
        else:
            y = y.loc[ddf.index]
    elif isinstance(y, pd.Series):
        y = y.loc[ddf.index]
    else:
        # Convert to Series if it's an array-like
        y = pd.Series(y, index=ddf.index) if len(ddf) == len(y) else y

    # The active split plan selects random, time-ordered, or grouped rows. The
    # split column is read from the frame the pipeline received when the column
    # is not a feature. The default plan is the former train_test_split call.
    split_plan = get_split_plan()
    train_idx, test_idx = split_indices(
        X, testSize, seed, strat=strat, plan=split_plan, source=_split_source, feature_columns=X.columns
    )
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    # Statistical fills are fitted on the training rows only and applied to both halves.
    ddf, _plan = apply_fill_plan_to_split(ddf, X_train.index)
    if _plan is not None:
        feat_cols = nump + catp + textp
        X_train = ddf.loc[X_train.index, feat_cols]
        X_test = ddf.loc[X_test.index, feat_cols]

    # Ordinal encoding for pre-made bins: map labels to integers in numeric bin order
    # so the classifier fits ordered classes (<50, 50-150, 150-500, >500).
    ordinal_classes = None
    label_to_int = None
    int_to_label = None
    y_train_int = None
    is_single_target = isinstance(y_train, pd.Series) or (
        hasattr(y_train, "shape") and len(y_train.shape) >= 2 and y_train.shape[1] == 1
    )
    if is_single_target:
        all_labels = list(set(str(v) for v in np.ravel(y_train)) | set(str(v) for v in np.ravel(y_test)))
        ordinal_classes = sort_class_labels_numeric_bins(all_labels)
        label_to_int = {c: i for i, c in enumerate(ordinal_classes)}
        int_to_label = {i: c for c, i in label_to_int.items()}
        y_train_int = _encode_ordinal_target(y_train, label_to_int)
        _encode_ordinal_target(y_test, label_to_int)

    # RAW splits for transformer
    X_train_raw = ddf.loc[X_train.index, nump + catp + textp].copy()
    # The spread of each raw predictor in the training rows, for the inference page.
    write_training_spread(X_train_raw[nump], VIS_DIR)
    X_test_raw = ddf.loc[X_test.index, nump + catp + textp].copy()

    # Transform
    preproc = make_preprocessor(numeric_cols=nump, categorical_cols=catp, cat_mode="onehot")
    X_train_t = pd.DataFrame(preproc.fit_transform(X_train_raw), index=X_train_raw.index)
    X_test_t = pd.DataFrame(preproc.transform(X_test_raw), index=X_test_raw.index)
    feat_names = get_feature_names(preproc, X_train_raw)
    if len(feat_names) == X_train_t.shape[1]:
        X_train_t.columns = feat_names
        X_test_t.columns = feat_names

    # Scale X only
    X_train_s, X_test_s, _, _, X_scaler, _ = _scale_pairs(X_train_t, X_test_t, None, None, X_scaler_type, "none")
    # What the training rows looked like, for measuring a new row against them later.
    record_applicability(X_train_s, seed)

    # Apply outlier handling (before feature selection)
    outlier_info = None
    if outlier_method != "none":
        train_index_before = X_train_s.index  # kept to report which rows were removed
        X_train_s, X_test_s, outlier_mask = apply_outlier_handling(
            X_train_s, X_test_s, y_train, outlier_method, outlier_action, seed=seed
        )
        # Update y_train if outliers were removed
        if outlier_action == "remove":
            y_train = y_train[outlier_mask]
            if y_train_int is not None:
                y_train_int = y_train_int.loc[outlier_mask]
        n_outliers = int((~outlier_mask).sum())
        outlier_info = {
            "method": outlier_method,
            "action": outlier_action,
            "n_outliers": n_outliers,
            "original_samples": int(len(outlier_mask)),
            "remaining_samples": int(outlier_mask.sum()),
        }
        if outlier_action == "remove" and n_outliers > 0:
            outlier_info["removed_row_indices"] = train_index_before[~outlier_mask].tolist()

    # Apply feature selection
    feature_selection_info = None
    original_feature_count = X_train_s.shape[1]
    X_train_s.columns.tolist() if isinstance(X_train_s, pd.DataFrame) else []
    if feature_selection_method != "none" and feature_selection_k:
        X_train_s, X_test_s, feature_selector = apply_feature_selection(
            X_train_s,
            X_test_s,
            y_train_int if y_train_int is not None else y_train,
            feature_selection_method,
            feature_selection_k,
            "classification",
            seed=seed,
        )
        if feature_selector is not None and isinstance(X_train_s, pd.DataFrame):
            selected_features = X_train_s.columns.tolist()
            feature_selection_info = {
                "method": feature_selection_method,
                "k_requested": feature_selection_k,
                "original_count": int(original_feature_count),
                "selected_count": len(selected_features),
                "selected_features": selected_features,
            }

    # fit on ordinal integers when bins are pre-made so the model sees ordered classes (0,1,2,...)
    fit_y = y_train_int if y_train_int is not None else y_train
    fit_y_series = fit_y if isinstance(fit_y, pd.Series) else pd.Series(np.asarray(fit_y).ravel())
    if fit_y_series.isna().any():
        n_bad = int(fit_y_series.isna().sum())
        raise ValueError(
            f"The target has {n_bad} missing value(s) after preprocessing. "
            "In Data Cleaning, add the target to the missing-value handling. "
            "You can also drop the rows that have a missing target. "
            "If the target still has blanks, run autodetect again with Targets selected."
        )
    if hyperparameter_search != "none":
        model = apply_hyperparameter_search(
            model,
            X_train_s,
            fit_y,
            hyperparameter_search,
            param_grid=None,
            cv_folds=search_cv_folds,
            n_iter=search_n_iter,
            problem_type="classification",
            model_name=model_name,
            seed=seed,
        )
    else:
        model.fit(X_train_s, fit_y)

    # Evaluate
    y_pred = model.predict(X_test_s)
    if ordinal_classes is not None and int_to_label is not None:
        # Map integer predictions back to original bin labels for report and display
        y_pred_1d = np.array([int_to_label[int(p)] for p in np.ravel(y_pred)])
        classes = ordinal_classes
        # int_to_label records the str(v) forms the vocabulary was built from, so a
        # numeric target would leave y_test as ints against string predictions and
        # classification_report would reject the mixed input.
        y_test_1d = np.array([str(v) for v in np.ravel(y_test)])
    else:
        _norm = _normalize_classes(model)
        classes = _norm if _norm is not None else model.classes_
        classes = sort_class_labels_numeric_bins(classes)
        y_test_1d = np.asarray(y_test).ravel()
        y_pred_1d = np.asarray(y_pred).ravel()
    report = classification_report(y_test_1d, y_pred_1d, output_dict=True)
    cm = confusion_matrix(y_test_1d, y_pred_1d, labels=classes)
    # The most common training class, scored on the same validation rows.
    try:
        baseline_info = classification_baseline(y_train, y_test_1d, report.get("accuracy"))
    except Exception as e:  # a baseline failure must not stop a run
        logger.debug("Could not score the majority baseline: %s", e)
        baseline_info = None

    # compute additional metrics
    from python_scripts.preprocessing.calculate_all_metrics import calculate_classification_metrics

    additional_metrics = calculate_classification_metrics(
        y_test_1d,
        y_pred_1d,
        y_score=None,  # Will be calculated from model
        classes=classes,
        model=model,
        X_test=X_test_s,
    )

    logger.debug(f"Classification report:\n{report}")
    logger.debug(f"Confusion matrix:\n{cm}")
    logger.debug(f"Model classes: {classes}")

    # Determine mode label for graphics
    if modeling_mode == "automl":
        mode_label = "DiGiTerra AutoML"
    elif modeling_mode == "advanced":
        mode_label = "DiGiTerra Advanced Modeling"
    else:
        mode_label = "DiGiTerra Simple Modeling"

    art = {
        "type": "classification",
        "seed": seed,
        "preprocessor": preproc,
        "X_scaler": X_scaler,
        "model": model,
        "model_name": f"{model_name} ({mode_label})",
        # X_train_s is what survives feature selection and is what the model saw.
        "feature_names": list(X_train_s.columns) if isinstance(X_train_s, pd.DataFrame) else list(X_train_t.columns),
        "splits": {"X_train": X_train_s, "X_test": X_test_s, "y_train": y_train, "y_test": y_test},
        "predictions": {"y_test_pred": y_pred_1d},
        "metrics": {"classification_report": report, "confusion_matrix": cm, "classes": classes},
        "additional_metrics": additional_metrics,
    }

    pdf_pages = PdfPages(VIS_DIR / "visualizations.pdf")
    svctrue = False
    if model_name == "SVC":
        svctrue = True

    export_plots(art, pdf_pages, units=units, svctrue=svctrue)

    pdf_pages.close()

    logger.info("Classification pipeline completed")
    quantileBinResults = ""

    # Extract parameters - use best_estimator params from search if available
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    if isinstance(model, (GridSearchCV, RandomizedSearchCV)):
        # Use best_estimator's params which includes all parameters with best values from search
        params_to_return = model.best_estimator_.get_params()
        logger.info(f"Using best parameters from hyperparameter search: {model.best_params_}")
    else:
        params_to_return = model.get_params()

    # Use processed data shapes (after outlier removal and feature selection) for accurate sample counts
    # X_train_s and X_test_s are the final processed data used for training
    processed_X_train_shape = X_train_s.shape if hasattr(X_train_s, "shape") else (len(X_train_s), 0)
    processed_X_test_shape = X_test_s.shape if hasattr(X_test_s, "shape") else (len(X_test_s), 0)
    # y_train may have been filtered if outliers were removed
    processed_y_train_shape = y_train.shape if hasattr(y_train, "shape") else (len(y_train),)
    processed_y_test_shape = y_test.shape if hasattr(y_test, "shape") else (len(y_test),)

    # Store fitted inference artifacts on the model so prediction can replay train-time transforms.
    try:
        model._digiterra_problem_type = "classification"
        model._digiterra_preprocessor = preproc
        model._digiterra_raw_features = X_train.columns.tolist()
        model._digiterra_feature_names = X_train_t.columns.tolist()
        model._digiterra_model_features = X_train_s.columns.tolist()
        if int_to_label is not None:
            model._digiterra_int_to_label = int_to_label
    except Exception:
        logger.debug("Could not attach inference artifacts to classification model", exc_info=True)

    # Write training target summary for inference page comparison (target distribution the model was built on)
    try:
        y_flat = np.asarray(y_train).ravel()
        vc = pd.Series(y_flat).value_counts()
        col_name = "Target"
        if target_variables:
            col_name = target_variables[0] if isinstance(target_variables, (list, tuple)) else str(target_variables)
        elif hasattr(y_train, "name") and y_train.name is not None:
            col_name = str(y_train.name)
        elif hasattr(y_train, "columns") and len(y_train.columns):
            col_name = str(y_train.columns[0])
        training_summary = [
            {
                "column": col_name,
                "n": int(len(y_flat)),
                "value_counts": [{"value": str(k), "count": int(v)} for k, v in vc.items()],
            }
        ]
        with open(VIS_DIR / "training_target_summary.json", "w") as f:
            json.dump({"model_type": "classification", "summary": training_summary}, f, indent=2)
    except Exception as e:
        logger.debug("Could not write training target summary: %s", e)

    return (
        report,
        cm,
        params_to_return,
        {
            "X_train": processed_X_train_shape,
            "X_test": processed_X_test_shape,
            "y_train": processed_y_train_shape,
            "y_test": processed_y_test_shape,
        },
        model,
        X_scaler,
        quantileBinResults,
        X_train.columns.tolist(),
        additional_metrics,
        feature_selection_info,
        outlier_info,
        baseline_info,
    )
