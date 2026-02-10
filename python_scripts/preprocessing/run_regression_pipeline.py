from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
import re
import math
import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

logger = logging.getLogger(__name__)

from sklearn.model_selection import train_test_split

from matplotlib.backends.backend_pdf import PdfPages
from python_scripts.preprocessing.visualize_predictions import visualize_predictions
from python_scripts.plotting.plot_shap_summary_graphic import plot_shap_summary
from python_scripts.preprocessing.utilites import make_strat_labels_robust
from python_scripts.preprocessing.utilites import make_preprocessor
from python_scripts.preprocessing.utilites import get_feature_names
from python_scripts.preprocessing.utilites import _scale_pairs
from python_scripts.preprocessing.utilites import regression_report
from python_scripts.preprocessing.utilites import export_plots
from python_scripts.preprocessing.evaluate_by_quantile import evaluate_by_quantile
from python_scripts.preprocessing.feature_selection import apply_feature_selection
from python_scripts.preprocessing.outlier_handling import apply_outlier_handling
from python_scripts.preprocessing.hyperparameter_search import apply_hyperparameter_search, _estimate_param_combinations

from python_scripts.config import VIS_DIR

def run_regression(model, model_name,
                    train_data, target_variables, use_stratified_split,
                    X, y, stratifyColumn,
                    units, X_scaler_type, y_scaler_type,
                    seed, sigfig, quantileBinDict, useTransformer, transformer_cols, testSize,
                    feature_selection_method='none', feature_selection_k=None,
                    outlier_method='none', outlier_action='remove',
                    hyperparameter_search='none', search_cv_folds=5, search_n_iter=50,
                    progress_tracker=None, modeling_mode='simple', **kwargs) -> Dict[str, Any]:
    # Extract modeling_mode from kwargs if provided there (for backward compatibility with train functions)
    if 'modeling_mode' in kwargs:
        modeling_mode = kwargs['modeling_mode']
    

    #assert cfg.problem_type.lower() == "regression", "Config.problem_type must be 'regression'."
    if target_variables is None:
        raise ValueError("Regression requires an explicit target_col.")
    

    # Build modeling frame

    if progress_tracker:
        progress_tracker.update_stage('model_training', 'running', 5, f'Preparing data for {model_name}...')
    
    nump = X.select_dtypes(include=np.number).columns.tolist()
    catp = []
    if useTransformer == 'Yes':
        catp=transformer_cols
    textp=[]
    
    use_cols = nump + catp + textp + list(target_variables)
    ddf = train_data[use_cols].dropna().copy()
    
    if progress_tracker:
        progress_tracker.update_stage('model_training', 'running', 10, f'Splitting data into train/test sets...')

    target_variable=target_variables[0]
    # Ensure numeric target
    # if not np.issubdtype(ddf[target_variable].dtype, np.number):
    #     ddf[target_variable] = pd.to_numeric(ddf[target_variable], errors="raise")
    y = y.apply(pd.to_numeric, errors="raise")

    #Rowan 10/13
    # Strat labels
    if not quantileBinDict['quantile']==0:
        logger.debug('Using quantiles for stratification')
        quantiles = quantileBinDict['quantile']
        train_data, use_strat, counts = make_strat_labels_robust(ddf, stratifyColumn, "regression", testSize, quantiles)
        
    elif not quantileBinDict['bin']==0:
        logger.debug("Using bins for stratification")
        def parseIntString(s):
            if s.strip()=='inf':
                return np.inf
            else:
                return int(s)

        bins = list(map(parseIntString, quantileBinDict['bin'].split(',')))
        binLabels = quantileBinDict['binsLabel'].split(',')     
        train_data, use_strat, counts = make_strat_labels_robust(ddf, stratifyColumn, "regression", testSize, bins)

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
    #y = ddf[[target_variable]].copy()
    #Rowan 10/13
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testSize, stratify=strat, random_state=seed
    )
    #if useTransformer == 'Yes':
    # RAW splits for transformer
    X_train_raw = ddf.loc[X_train.index, nump + catp + textp].copy()
    X_test_raw  = ddf.loc[X_test.index,  nump + catp + textp].copy()

    if progress_tracker:
        progress_tracker.update_stage('model_training', 'running', 15, 'Transforming features...')
    
    # Transform
    preproc = make_preprocessor(numeric_cols=nump, categorical_cols=catp, cat_mode="onehot") #categorical_cols are from transformer user input?
    X_train_t = pd.DataFrame(preproc.fit_transform(X_train_raw), index=X_train_raw.index)
    X_test_t  = pd.DataFrame(preproc.transform(X_test_raw),     index=X_test_raw.index)
    feat_names = get_feature_names(preproc, X_train_raw)
    if len(feat_names) == X_train_t.shape[1]:
        X_train_t.columns = feat_names; X_test_t.columns = feat_names

    if progress_tracker:
        progress_tracker.update_stage('model_training', 'running', 20, f'Scaling data ({X_scaler_type})...')
    
    # Scale
    X_train_s, X_test_s, y_train_s, y_test_s, X_scaler, y_scaler = _scale_pairs(
        X_train_t, X_test_t, y_train, y_test, X_scaler_type, y_scaler_type
    )

    # Save baseline state (before advanced options) for baseline graphics
    has_advanced_options = (feature_selection_method != 'none' and feature_selection_k) or outlier_method != 'none'
    
    # Generate baseline graphics if advanced options will be applied
    if has_advanced_options:
        # Train baseline model (no advanced options)
        baseline_y_train_fit = y_train_s.values.ravel() if y_train_s.shape[1] == 1 else y_train_s.values
        
        # Prepare baseline data for training
        if isinstance(X_train_s, pd.DataFrame):
            baseline_X_train_array = X_train_s.values
        else:
            baseline_X_train_array = X_train_s
        
        # Create a copy of the model for baseline training
        baseline_model = copy.deepcopy(model)
        
        # Validate baseline X and y shapes match before training
        baseline_y_samples = len(baseline_y_train_fit) if baseline_y_train_fit.ndim == 1 else baseline_y_train_fit.shape[0]
        if baseline_X_train_array.shape[0] != baseline_y_samples:
            raise ValueError(f"Size mismatch in baseline model training: baseline_X_train_array has {baseline_X_train_array.shape[0]} samples, "
                           f"but baseline_y_train_fit has {baseline_y_samples} samples (shape: {baseline_y_train_fit.shape}). "
                           f"X_train_s shape: {X_train_s.shape}, y_train_s shape: {y_train_s.shape}")
        
        # Apply hyperparameter search to baseline if enabled (but no feature selection/outlier handling)
        if hyperparameter_search != 'none':
            if progress_tracker:
                total_combinations = _estimate_param_combinations(model_name, search_n_iter, hyperparameter_search)
                progress_tracker.update_stage('baseline_training', 'running', 0, 
                                           f'Training baseline model ({total_combinations} combinations)...')
            baseline_model = apply_hyperparameter_search(
                baseline_model, baseline_X_train_array, baseline_y_train_fit, hyperparameter_search,
                param_grid=None, cv_folds=search_cv_folds, n_iter=search_n_iter, 
                problem_type='regression', model_name=model_name, progress_tracker=None  # Don't track baseline training
            )
        else:
            baseline_model.fit(baseline_X_train_array, baseline_y_train_fit)
        
        # Generate baseline predictions
        if isinstance(X_test_s, pd.DataFrame):
            baseline_X_test_array = X_test_s.values
        else:
            baseline_X_test_array = X_test_s
        
        baseline_y_tr_pred_s = baseline_model.predict(baseline_X_train_array)
        baseline_y_te_pred_s = baseline_model.predict(baseline_X_test_array)
        
        # Inverse transform predictions
        if y_scaler is not None:
            baseline_y_tr_pred = pd.DataFrame(y_scaler.inverse_transform(baseline_y_tr_pred_s.reshape(-1,1) if baseline_y_tr_pred_s.ndim==1 else baseline_y_tr_pred_s),
                                             columns=y.columns, index=X_train_s.index)
            baseline_y_te_pred = pd.DataFrame(y_scaler.inverse_transform(baseline_y_te_pred_s.reshape(-1,1) if baseline_y_te_pred_s.ndim==1 else baseline_y_te_pred_s),
                                             columns=y.columns, index=y_test.index)
        else:
            baseline_y_tr_pred = pd.DataFrame(baseline_y_tr_pred_s, columns=y.columns, index=X_train_s.index)
            baseline_y_te_pred = pd.DataFrame(baseline_y_te_pred_s, columns=y.columns, index=y_test.index)
        
        # Generate baseline graphics
        if progress_tracker:
            progress_tracker.update_stage('visualization', 'running', 10, 'Generating baseline visualizations...')
        baseline_pdf_pages = PdfPages(VIS_DIR / "visualizations_baseline.pdf")
        
        # Determine mode label for baseline graphics
        mode_label = 'DiGiTerra Simple Modeling' if modeling_mode == 'simple' else (
            'DiGiTerra Advanced Modeling' if modeling_mode == 'advanced' else 'DiGiTerra AutoML'
        )
        visualize_predictions(
            model_name, y_train, baseline_y_tr_pred, y_test, baseline_y_te_pred, 
            target_variables, units, sigfig, baseline_pdf_pages,
            file_suffix='', label_suffix=f'({mode_label})'
        )
        
        # Generate baseline SHAP if single target
        if np.asarray(target_variables).size == 1:
            try:
                if isinstance(X_train_s, pd.DataFrame):
                    baseline_shap_features = X_train_s.copy()
                else:
                    baseline_shap_features = pd.DataFrame(baseline_X_train_array, columns=X_train_t.columns[:baseline_X_train_array.shape[1]])
                
                # Handle NaN values
                if baseline_shap_features.isna().any().any():
                    numeric_cols = baseline_shap_features.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        baseline_shap_features[numeric_cols] = baseline_shap_features[numeric_cols].fillna(baseline_shap_features[numeric_cols].mean())
                    if baseline_shap_features.isna().any().any():
                        baseline_shap_features = baseline_shap_features.dropna(axis=1)
                
                baseline_shap_array = baseline_shap_features.values
                baseline_feature_names = baseline_shap_features.columns.tolist()
                
                if np.isinf(baseline_shap_array).any():
                    baseline_shap_array = np.nan_to_num(baseline_shap_array, nan=0.0, posinf=0.0, neginf=0.0)
                
                plot_shap_summary(baseline_model, baseline_shap_array, baseline_feature_names, model_name, 
                                baseline_pdf_pages, file_suffix='', label_suffix=f'({mode_label})')
            except Exception as error:
                logger.warning(f"Baseline SHAP summary failed: {error}", exc_info=True)
        
        baseline_pdf_pages.close()

    # Apply outlier handling (before feature selection)
    # Convert y_train_s to numpy array, handling both single and multi-target cases
    if y_train_s.shape[1] == 1:
        y_train_fit = y_train_s.values.ravel()  # 1D array for single target
    else:
        y_train_fit = y_train_s.values  # 2D array for multi-target
    train_indices_kept = X_train_s.index  # Track which training samples are kept
    
    # Validate X and y have matching sample counts before outlier handling
    if X_train_s.shape[0] != y_train_s.shape[0]:
        raise ValueError(f"Size mismatch before outlier handling: X_train_s has {X_train_s.shape[0]} samples, "
                       f"but y_train_s has {y_train_s.shape[0]} samples. "
                       f"X_train_s shape: {X_train_s.shape}, y_train_s shape: {y_train_s.shape}")
    
    outlier_info = None
    if outlier_method != 'none':
        if progress_tracker:
            progress_tracker.update_stage('outlier_handling', 'running', 0, f'Detecting outliers using {outlier_method}...')
        # For outlier detection, we don't need y_train, but pass None to be safe
        # (y_train is not used in apply_outlier_handling for detection)
        X_train_s, X_test_s, outlier_mask = apply_outlier_handling(
            X_train_s, X_test_s, None, outlier_method, outlier_action
        )
        # Update y_train if outliers were removed
        if outlier_action == 'remove':
            # Ensure outlier_mask length matches original y_train_fit length
            original_y_length = len(y_train_fit) if y_train_fit.ndim == 1 else y_train_fit.shape[0]
            if len(outlier_mask) != original_y_length:
                raise ValueError(f"Size mismatch after outlier removal: outlier_mask length ({len(outlier_mask)}) != y_train_fit length ({original_y_length}). "
                               f"X_train_s shape before: {X_train_s.shape if hasattr(X_train_s, 'shape') else 'N/A'}, "
                               f"X_train_s shape after: {X_train_s.shape}, y_train_s shape: {y_train_s.shape}")
            # Filter y_train_fit using outlier_mask (works for both 1D and 2D arrays)
            y_train_fit = y_train_fit[outlier_mask]
            train_indices_kept = X_train_s.index  # Update indices to match remaining samples
        n_outliers = (~outlier_mask).sum() if hasattr(outlier_mask, 'sum') else len(outlier_mask) - outlier_mask.sum()
        original_sample_count = len(outlier_mask)
        remaining_sample_count = outlier_mask.sum() if hasattr(outlier_mask, 'sum') else sum(outlier_mask)
        outlier_info = {
            'method': outlier_method,
            'action': outlier_action,
            'n_outliers': int(n_outliers),
            'original_samples': int(original_sample_count),
            'remaining_samples': int(remaining_sample_count)
        }
        if progress_tracker:
            progress_tracker.update_stage('outlier_handling', 'completed', 100, 
                                        f'Outlier handling complete ({n_outliers} outliers {"removed" if outlier_action == "remove" else "capped"})')

    # Apply feature selection
    feature_selector = None
    original_feature_count = X_train_s.shape[1]
    original_feature_names = X_train_s.columns.tolist() if isinstance(X_train_s, pd.DataFrame) else [f'Feature_{i}' for i in range(original_feature_count)]
    selected_features = None
    if feature_selection_method != 'none' and feature_selection_k:
        if progress_tracker:
            progress_tracker.update_stage('feature_selection', 'running', 0, f'Selecting {feature_selection_k} features using {feature_selection_method}...')
        # For feature selection, y_train needs to be a Series
        # Handle both 1D and 2D y_train_fit cases
        if y_train_fit.ndim == 1:
            y_train_for_selection = pd.Series(y_train_fit, index=X_train_s.index)
        else:
            # For multi-target, use first target for feature selection (or could use all targets)
            # Most feature selection methods expect 1D target
            y_train_for_selection = pd.Series(y_train_fit[:, 0], index=X_train_s.index)
        
        X_train_s, X_test_s, feature_selector = apply_feature_selection(
            X_train_s, X_test_s, y_train_for_selection, feature_selection_method, feature_selection_k, 'regression'
        )
        
        # Validate shapes after feature selection
        y_train_fit_samples_after_fs = len(y_train_fit) if y_train_fit.ndim == 1 else y_train_fit.shape[0]
        if X_train_s.shape[0] != y_train_fit_samples_after_fs:
            raise ValueError(f"Size mismatch after feature selection: X_train_s has {X_train_s.shape[0]} samples, "
                           f"but y_train_fit has {y_train_fit_samples_after_fs} samples.")
        if feature_selector is not None:
            # Get selected feature names from the selector
            if hasattr(feature_selector, 'get_support'):
                selected_indices = feature_selector.get_support()
                if isinstance(X_train_s, pd.DataFrame):
                    # After selection, X_train_s.columns contains the selected features
                    selected_features = X_train_s.columns.tolist()
                else:
                    # If we lost column names, reconstruct from original names
                    selected_features = [original_feature_names[i] for i, selected in enumerate(selected_indices) if selected]
        if progress_tracker:
            progress_tracker.update_stage('feature_selection', 'completed', 100, 
                                        f'Feature selection complete ({X_train_s.shape[1]} features selected)')

    # Validate data before training - check for NaN values
    # First, try to handle NaN values gracefully before raising an error
    if isinstance(X_train_s, pd.DataFrame):
        if X_train_s.isna().any().any():
            nan_cols = X_train_s.columns[X_train_s.isna().any()].tolist()
            nan_count = X_train_s.isna().sum().sum()
            
            # Try to fill NaN values in numeric columns with column means
            numeric_cols = X_train_s.select_dtypes(include=[np.number]).columns
            nan_numeric_cols = [col for col in nan_cols if col in numeric_cols]
            
            if nan_numeric_cols:
                logger.warning(f"NaN values detected in numeric columns {nan_numeric_cols}. Attempting to fill with column means...")
                for col in nan_numeric_cols:
                    col_mean = X_train_s[col].mean()
                    if pd.isna(col_mean):
                        # If column mean is also NaN (all values are NaN), fill with 0
                        X_train_s[col] = X_train_s[col].fillna(0.0)
                        logger.warning(f"Column {col} had all NaN values, filled with 0.0")
                    else:
                        X_train_s[col] = X_train_s[col].fillna(col_mean)
            
            # Drop any remaining columns with NaN (non-numeric columns)
            remaining_nan_cols = X_train_s.columns[X_train_s.isna().any()].tolist()
            if remaining_nan_cols:
                logger.warning(f"Dropping columns with remaining NaN values: {remaining_nan_cols}")
                X_train_s = X_train_s.drop(columns=remaining_nan_cols)
            
            # Final check - if NaN still exists, raise error with detailed info
            if X_train_s.isna().any().any():
                nan_cols_final = X_train_s.columns[X_train_s.isna().any()].tolist()
                nan_count_final = X_train_s.isna().sum().sum()
                raise ValueError(f"NaN values could not be removed from X_train_s. {nan_count_final} NaN values remain in columns: {nan_cols_final}. "
                               f"Data shape: {X_train_s.shape}, Columns: {list(X_train_s.columns)}. "
                               f"This may be caused by column name mismatches after feature selection or scaling.")
        
        # Convert to numpy array for model training (scikit-learn expects arrays)
        X_train_array = X_train_s.values
        # Double-check the array doesn't have NaN
        if np.isnan(X_train_array).any():
            # Try to replace NaN with 0 as last resort
            nan_positions = np.isnan(X_train_array)
            nan_count = nan_positions.sum()
            logger.warning(f"NaN values detected in X_train_array after conversion. Replacing {nan_count} NaN values with 0.0")
            X_train_array = np.nan_to_num(X_train_array, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        X_train_array = X_train_s
        if np.isnan(X_train_array).any():
            # Try to replace NaN with 0 as last resort
            nan_positions = np.isnan(X_train_array)
            nan_count = nan_positions.sum()
            logger.warning(f"NaN values detected in X_train_s (numpy array). Replacing {nan_count} NaN values with 0.0")
            X_train_array = np.nan_to_num(X_train_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check y_train_fit for NaN
    if np.isnan(y_train_fit).any():
        nan_count = np.isnan(y_train_fit).sum()
        logger.warning(f"NaN values detected in y_train_fit ({nan_count} values). Replacing with 0.0")
        y_train_fit = np.nan_to_num(y_train_fit, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Validate X and y have matching sample counts before training
    y_train_fit_samples = len(y_train_fit) if y_train_fit.ndim == 1 else y_train_fit.shape[0]
    if X_train_array.shape[0] != y_train_fit_samples:
        raise ValueError(f"Size mismatch before model training: X_train_array has {X_train_array.shape[0]} samples, "
                       f"but y_train_fit has {y_train_fit_samples} samples. "
                       f"X_train_s shape: {X_train_s.shape if isinstance(X_train_s, pd.DataFrame) else 'N/A'}, "
                       f"y_train_s shape: {y_train_s.shape}, "
                       f"y_train_fit shape: {y_train_fit.shape}, "
                       f"train_indices_kept length: {len(train_indices_kept)}, "
                       f"outlier_method: {outlier_method}, outlier_action: {outlier_action}")
    
    # Ensure X_train_s remains a DataFrame for SHAP (with correct columns)
    # This is important because SHAP needs column names for feature names
    if not isinstance(X_train_s, pd.DataFrame):
        # Reconstruct DataFrame from array using the column names we should have
        # This shouldn't happen if the pipeline is correct, but handle it gracefully
        if hasattr(X_train_t, 'columns') and X_train_array.shape[1] <= len(X_train_t.columns):
            X_train_s = pd.DataFrame(X_train_array, 
                                     index=range(len(X_train_array)) if not hasattr(X_train_s, 'index') else X_train_s.index, 
                                     columns=X_train_t.columns[:X_train_array.shape[1]])
        else:
            X_train_s = pd.DataFrame(X_train_array, columns=[f'feature_{i}' for i in range(X_train_array.shape[1])])
    
    # Apply hyperparameter search
    if hyperparameter_search != 'none':
        if progress_tracker:
            total_combinations = _estimate_param_combinations(model_name, search_n_iter, hyperparameter_search)
            progress_tracker.update_stage('hyperparameter_search', 'running', 0, 
                                       f'Starting {hyperparameter_search} search ({total_combinations} combinations, {search_cv_folds} CV folds)...')
        model = apply_hyperparameter_search(
            model, X_train_array, y_train_fit, hyperparameter_search,
            param_grid=None, cv_folds=search_cv_folds, n_iter=search_n_iter, 
            problem_type='regression', model_name=model_name, progress_tracker=progress_tracker
        )
        if progress_tracker:
            progress_tracker.update_stage('hyperparameter_search', 'completed', 100, 'Hyperparameter search complete')
            progress_tracker.update_stage('model_training', 'running', 80, f'Training {model_name} with best parameters...')
        model.fit(X_train_array, y_train_fit)
        if progress_tracker:
            progress_tracker.update_stage('model_training', 'running', 90, 'Generating predictions...')
    else:
        # Fit
        if progress_tracker:
            progress_tracker.update_stage('model_training', 'running', 30, f'Training {model_name} model...')
        model.fit(X_train_array, y_train_fit)
        if progress_tracker:
            progress_tracker.update_stage('model_training', 'running', 70, 'Generating predictions...')

    # Predict (+ inverse)
    # Convert test data to array if needed
    if isinstance(X_test_s, pd.DataFrame):
        X_test_array = X_test_s.values
    else:
        X_test_array = X_test_s
    
    y_tr_pred_s = model.predict(X_train_array)
    y_te_pred_s = model.predict(X_test_array)
    
    # Get the actual y_train and y_test values that match the predictions
    # (accounting for outlier removal)
    y_train_actual = y_train.loc[train_indices_kept] if len(train_indices_kept) < len(y_train) else y_train
    
    if y_scaler is not None:
        y_tr_pred = pd.DataFrame(y_scaler.inverse_transform(y_tr_pred_s.reshape(-1,1) if y_tr_pred_s.ndim==1 else y_tr_pred_s),
                                 columns=y.columns, index=train_indices_kept)
        y_te_pred = pd.DataFrame(y_scaler.inverse_transform(y_te_pred_s.reshape(-1,1) if y_te_pred_s.ndim==1 else y_te_pred_s),
                                 columns=y.columns, index=y_test.index)
    else:
        y_tr_pred = pd.DataFrame(y_tr_pred_s, columns=y.columns, index=train_indices_kept)
        y_te_pred = pd.DataFrame(y_te_pred_s, columns=y.columns, index=y_test.index)

    if progress_tracker:
        progress_tracker.update_stage('model_training', 'running', 85, 'Calculating performance metrics...')
    
    # Metrics
    metrics_train = regression_report(y_train_actual, y_tr_pred, target_names=list(y.columns))
    metrics_test  = regression_report(y_test,  y_te_pred, target_names=list(y.columns))

    if progress_tracker:
        progress_tracker.update_stage('model_training', 'completed', 100, 'Model training complete')
        progress_tracker.update_stage('visualization', 'running', 0, 'Generating visualizations...')
    pdf_pages = PdfPages(VIS_DIR / "visualizations.pdf")
    quantileBin_results = ''
    if use_strat:
        test_metadata = train_data.loc[X_test.index][["QUANTILE_STRATIFY"]]
        logger.debug("Evaluating by quantile")
        quantileBin_results = evaluate_by_quantile(
            model_name, y_test[target_variable], y_te_pred[target_variable],
            test_metadata, target_variables, sigfig, pdf_pages, "QUANTILE_STRATIFY"
        )


    logger.debug(f"Training metrics:\n{metrics_train}")
    logger.debug(f"Test metrics:\n{metrics_test}")


    # Determine mode label for graphics
    if modeling_mode == 'automl':
        mode_label = 'DiGiTerra AutoML'
    elif modeling_mode == 'advanced':
        mode_label = 'DiGiTerra Advanced Modeling'
    else:
        mode_label = 'DiGiTerra Simple Modeling'
    
    art =  { "type": 'regression',
        "preprocessor": preproc,
        "X_scaler": X_scaler,
        "y_scaler": y_scaler,
        "model": model,
        "model_name": f"{model_name} ({mode_label})",
        "feature_names": list(X_train_t.columns),
        "splits": {
            "X_train": X_train_s, "X_test": X_test_s,
            "y_train": y_train,   "y_test": y_test
        },
        "predictions": {
            "y_train_pred": y_tr_pred, "y_test_pred": y_te_pred
        },
        "metrics": {
            "train": metrics_train, "test": metrics_test
        }
    }

    
    

    graphics_suffix = '_advanced' if has_advanced_options else ''
    
    # Use y_train_actual instead of y_train to match the filtered predictions after outlier removal
    visualize_predictions(
        model_name, y_train_actual, y_tr_pred, y_test, y_te_pred, target_variables, units, sigfig, pdf_pages,
        file_suffix=graphics_suffix, label_suffix=f'({mode_label})'
    )

    export_plots(
       art, pdf_pages, units=units)
    
    # Construct shap_path using graphics_suffix to match what plot_shap_summary actually creates
    shap_filename = f"shap_summary{graphics_suffix}.png"
    shap_path = VIS_DIR / shap_filename
    if np.asarray(target_variables).size == 1:
        try:
            # X_train_s should be a DataFrame with correct column names
            # After scaling, it has columns from X_train_t
            # After feature selection, it has only the selected columns
            if isinstance(X_train_s, pd.DataFrame):
                # Use X_train_s directly - it should already have the correct columns
                shap_features = X_train_s.copy()
            else:
                # X_train_s is a numpy array - convert to DataFrame
                # This shouldn't happen, but handle it gracefully
                num_cols = X_train_s.shape[1] if hasattr(X_train_s, 'shape') else len(X_train_s[0]) if hasattr(X_train_s, '__len__') and len(X_train_s) > 0 else 0
                if num_cols > 0:
                    # Try to get column names from X_train_t if available
                    if num_cols <= len(X_train_t.columns):
                        shap_features = pd.DataFrame(
                            X_train_s,
                            index=range(len(X_train_s)) if not hasattr(X_train_s, 'index') else X_train_s.index,
                            columns=X_train_t.columns[:num_cols]
                        )
                    else:
                        # Fallback: use integer column names
                        shap_features = pd.DataFrame(
                            X_train_s,
                            columns=[f'feature_{i}' for i in range(num_cols)]
                        )
                else:
                    raise ValueError("X_train_s has no columns for SHAP")
            
            # Validate: ensure no NaN values in the data itself (not from column mismatch)
            if shap_features.isna().any().any():
                nan_cols = shap_features.columns[shap_features.isna().any()].tolist()
                nan_count = shap_features.isna().sum().sum()
                logger.warning(f"{nan_count} NaN values detected in data columns {nan_cols} before SHAP.")
                logger.debug(f"Data shape: {shap_features.shape}, Column names: {list(shap_features.columns)}")
                
                # Try to fill NaN with column means (for numeric columns only)
                numeric_cols = shap_features.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    shap_features[numeric_cols] = shap_features[numeric_cols].fillna(shap_features[numeric_cols].mean())
                
                # If still NaN, drop those columns
                if shap_features.isna().any().any():
                    remaining_nan_cols = shap_features.columns[shap_features.isna().any()].tolist()
                    shap_features = shap_features.dropna(axis=1)
                    logger.warning(f"Dropped columns with NaN: {remaining_nan_cols}")
            
            # Final check - ensure no NaN values remain
            if shap_features.isna().any().any():
                remaining_nan_cols = shap_features.columns[shap_features.isna().any()].tolist()
                raise ValueError(f"Input X contains NaN values that could not be removed. NaN columns: {remaining_nan_cols}")
            
            # Ensure we have valid data
            if shap_features.empty or shap_features.shape[1] == 0:
                raise ValueError("SHAP input is empty after processing")
            
            # Convert to numpy array for SHAP
            shap_array = shap_features.values
            feature_names = shap_features.columns.tolist()
            
            # Additional validation: check for inf values
            if np.isinf(shap_array).any():
                logger.warning("Infinite values detected in SHAP input. Replacing with finite values...")
                shap_array = np.nan_to_num(shap_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Final validation: ensure no NaN in the array
            if np.isnan(shap_array).any():
                raise ValueError("NaN values still present in SHAP input array after all processing")
            
            plot_shap_summary(model, shap_array, feature_names, model_name, pdf_pages,
                            file_suffix=graphics_suffix, label_suffix=f'({mode_label})')
        except Exception as error:
            logger.error(f"SHAP summary failed: {error}", exc_info=True)
            if shap_path.exists():
                shap_path.unlink()
    else:
        if shap_path.exists():
            shap_path.unlink()

    pdf_pages.close()
    if progress_tracker:
        progress_tracker.update_stage('visualization', 'completed', 100, 'Visualizations generated')
    quantileBinResults = ''
    
    # Extract parameters - use best_estimator params from search if available
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    if isinstance(model, (GridSearchCV, RandomizedSearchCV)):
        # Use best_estimator's params which includes all parameters with best values from search
        params_to_return = model.best_estimator_.get_params()
        logger.info(f"Using best parameters from hyperparameter search: {model.best_params_}")
    else:
        params_to_return = model.get_params()
    
    # Prepare feature selection info
    feature_selection_info = None
    if feature_selector is not None and selected_features is not None:
        feature_selection_info = {
            'method': feature_selection_method,
            'k_requested': feature_selection_k,
            'original_count': int(original_feature_count),
            'selected_count': len(selected_features),
            'selected_features': selected_features
        }
    
    # Use processed data shapes (after outlier removal and feature selection) for accurate sample counts
    # X_train_s and X_test_s are the final processed data used for training
    processed_X_train_shape = X_train_s.shape if hasattr(X_train_s, 'shape') else (len(X_train_s), 0)
    processed_X_test_shape = X_test_s.shape if hasattr(X_test_s, 'shape') else (len(X_test_s), 0)
    processed_y_train_shape = y_train_actual.shape if hasattr(y_train_actual, 'shape') else (len(y_train_actual),)
    processed_y_test_shape = y_test.shape if hasattr(y_test, 'shape') else (len(y_test),)
    
    return metrics_train, metrics_test, params_to_return, {
                'X_train': processed_X_train_shape,
                'X_test': processed_X_test_shape,
                'y_train': processed_y_train_shape,
                'y_test': processed_y_test_shape
            }, model, y_scaler, X_scaler, quantileBinResults, X_train.columns.tolist(), feature_selection_info, outlier_info
