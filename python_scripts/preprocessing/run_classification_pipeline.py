from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    precision_recall_curve, average_precision_score, roc_curve, auc, roc_auc_score
)

from matplotlib.backends.backend_pdf import PdfPages
from python_scripts.preprocessing.utilites import make_strat_labels_robust
from python_scripts.preprocessing.utilites import make_preprocessor
from python_scripts.preprocessing.utilites import get_feature_names
from python_scripts.preprocessing.utilites import _scale_pairs
from python_scripts.preprocessing.utilites import export_plots
from python_scripts.preprocessing.feature_selection import apply_feature_selection
from python_scripts.preprocessing.outlier_handling import apply_outlier_handling
from python_scripts.preprocessing.hyperparameter_search import apply_hyperparameter_search

from python_scripts.config import VIS_DIR


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


def run_classification(model, model_name,
                    train_data, target_variables, use_stratified_split,
                    X, y, stratifyColumn,
                    units, X_scaler_type,
                    seed, sigfig, quantileBinDict, useTransformer, transformer_cols, testSize,
                    feature_selection_method='none', feature_selection_k=None,
                    outlier_method='none', outlier_action='remove',
                    hyperparameter_search='none', search_cv_folds=5, search_n_iter=50,
                    modeling_mode='simple', **kwargs) -> Dict[str, Any]:
    # Extract modeling_mode from kwargs if provided there (for backward compatibility)
    if 'modeling_mode' in kwargs:
        modeling_mode = kwargs['modeling_mode']

    if target_variables is None:
        raise ValueError("Classification requires an explicit target_col.")
    
    nump = X.select_dtypes(include=np.number).columns.tolist()
    catp = []
    if useTransformer == 'Yes':
        catp=transformer_cols.tolist()
    textp=[]

    use_cols = nump + catp + textp + list(target_variables)
    ddf = train_data[use_cols].dropna().copy()

    #target_variables=target_variables[0]
    #Rowan 10/13
    # Strat labels
    if not quantileBinDict['quantile']==0:
        logger.debug('Using quantiles for stratification')
        quantiles = quantileBinDict['quantile']
        train_data, use_strat, counts = make_strat_labels_robust(ddf, stratifyColumn, "classification", testSize, quantiles)
        
    elif not quantileBinDict['bin']==0:
        logger.debug("Using bins for stratification")
        def parseIntString(s):
            if s.strip()=='inf':
                return np.inf
            else:
                return int(s)

        bins = list(map(parseIntString, quantileBinDict['bin'].split(',')))
        binLabels = quantileBinDict['binsLabel'].split(',')     
        train_data, use_strat, counts = make_strat_labels_robust(ddf, stratifyColumn, "classification", testSize, bins)

    else:
        train_data, use_strat, counts = ddf.copy(), False, {}

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
            y = y.iloc[:, 0]
        # For multi-column DataFrame, keep as is but ensure index alignment
        else:
            y = y.loc[ddf.index]
    elif isinstance(y, pd.Series):
        y = y.loc[ddf.index]
    else:
        # Convert to Series if it's an array-like
        y = pd.Series(y, index=ddf.index) if len(ddf) == len(y) else y
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testSize, stratify=strat, random_state=seed
    )

    # RAW splits for transformer
    X_train_raw = ddf.loc[X_train.index, nump + catp + textp].copy()
    X_test_raw  = ddf.loc[X_test.index,  nump + catp + textp].copy()

    # Transform
    preproc = make_preprocessor(numeric_cols=nump, categorical_cols=catp, cat_mode="onehot")
    X_train_t = pd.DataFrame(preproc.fit_transform(X_train_raw), index=X_train_raw.index)
    X_test_t  = pd.DataFrame(preproc.transform(X_test_raw),     index=X_test_raw.index)
    feat_names = get_feature_names(preproc, X_train_raw)
    if len(feat_names) == X_train_t.shape[1]:
        X_train_t.columns = feat_names; X_test_t.columns = feat_names

    # Scale X only
    X_train_s, X_test_s, _, _, X_scaler, _ = _scale_pairs(X_train_t, X_test_t, None, None, X_scaler_type, "none")

    # Apply outlier handling (before feature selection)
    train_indices_kept = X_train_s.index  # Track which training samples are kept
    if outlier_method != 'none':
        X_train_s, X_test_s, outlier_mask = apply_outlier_handling(
            X_train_s, X_test_s, y_train, outlier_method, outlier_action
        )
        # Update y_train if outliers were removed
        if outlier_action == 'remove':
            y_train = y_train[outlier_mask]
            train_indices_kept = X_train_s.index  # Update indices to match remaining samples

    # Apply feature selection
    if feature_selection_method != 'none' and feature_selection_k:
        X_train_s, X_test_s, _ = apply_feature_selection(
            X_train_s, X_test_s, y_train, feature_selection_method, feature_selection_k, 'classification'
        )

    # Apply hyperparameter search
    if hyperparameter_search != 'none':
        model = apply_hyperparameter_search(
            model, X_train_s, y_train, hyperparameter_search,
            param_grid=None, cv_folds=search_cv_folds, n_iter=search_n_iter, 
            problem_type='classification', model_name=model_name
        )
    else:
        # Fit
        model.fit(X_train_s, y_train)

    # Evaluate
    y_pred = model.predict(X_test_s)
    _norm = _normalize_classes(model)
    classes = _norm if _norm is not None else model.classes_
    # Ensure 1D targets for metrics (MultiOutputClassifier returns 2D predict for single output)
    y_test_1d = np.asarray(y_test).ravel()
    y_pred_1d = np.asarray(y_pred).ravel()
    report = classification_report(y_test_1d, y_pred_1d, output_dict=True)
    cm = confusion_matrix(y_test_1d, y_pred_1d, labels=classes)

    # Calculate comprehensive additional metrics
    from python_scripts.preprocessing.calculate_all_metrics import calculate_classification_metrics
    additional_metrics = calculate_classification_metrics(
        y_test_1d, y_pred_1d,
        y_score=None,  # Will be calculated from model
        classes=classes,
        model=model,
        X_test=X_test_s
    )

    logger.debug(f"Classification report:\n{report}")
    logger.debug(f"Confusion matrix:\n{cm}")
    logger.debug(f"Model classes: {classes}")

    # Determine mode label for graphics
    if modeling_mode == 'automl':
        mode_label = 'DiGiTerra AutoML'
    elif modeling_mode == 'advanced':
        mode_label = 'DiGiTerra Advanced Modeling'
    else:
        mode_label = 'DiGiTerra Simple Modeling'
    
    art = {
        "type": "classification",
        "preprocessor": preproc,
        "X_scaler": X_scaler,
        "model": model,
        "model_name": f"{model_name} ({mode_label})",
        "feature_names": list(X_train_t.columns),
        "splits": {
            "X_train": X_train_s, "X_test": X_test_s,
            "y_train": y_train,   "y_test": y_test
        },
        "predictions": {"y_test_pred": y_pred_1d},
        "metrics": {"classification_report": report, "confusion_matrix": cm, "classes": classes},
        "additional_metrics": additional_metrics
    }

    pdf_pages = PdfPages(VIS_DIR / "visualizations.pdf")
    svctrue=False
    if model_name=='SVC':
        svctrue=True

    export_plots(
       art, pdf_pages, units=units, svctrue=svctrue)
    
    pdf_pages.close()

    logger.info("Classification pipeline completed")
    quantileBinResults = ''
    
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
    processed_X_train_shape = X_train_s.shape if hasattr(X_train_s, 'shape') else (len(X_train_s), 0)
    processed_X_test_shape = X_test_s.shape if hasattr(X_test_s, 'shape') else (len(X_test_s), 0)
    # y_train may have been filtered if outliers were removed
    processed_y_train_shape = y_train.shape if hasattr(y_train, 'shape') else (len(y_train),)
    processed_y_test_shape = y_test.shape if hasattr(y_test, 'shape') else (len(y_test),)
    
    # Return in the expected order: report, cm, params, shapes, model, X_scaler, quantileBin_results, feature_order, additional_metrics
    # additional_metrics is added at the end for backward compatibility
    return report, cm, params_to_return, {
                'X_train': processed_X_train_shape,
                'X_test': processed_X_test_shape,
                'y_train': processed_y_train_shape,
                'y_test': processed_y_test_shape
            }, model, X_scaler, quantileBinResults, X_train.columns.tolist(), additional_metrics
