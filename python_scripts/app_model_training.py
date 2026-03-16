"""Model training orchestration for DiGiTerra.

Extracted from app.py so route logic stays thin. get_storage is injected
to avoid circular imports (app.py passes _get_session_storage).
"""

import logging
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd

from python_scripts import config as _config
from python_scripts.preprocessing.progress_tracker import get_tracker, remove_tracker, set_result
from python_scripts.helpers import (
    preprocess_data,
    run_cross_validation,
    unpack_classification_result,
    write_to_excel,
    write_to_excelClassifier,
    write_to_excelCluster,
    write_to_excelRegression,
)
from python_scripts.models.regression_models.train_linear import train_linear
from python_scripts.models.regression_models.train_lasso import train_lasso
from python_scripts.models.regression_models.train_elasticnet import train_elasticnet
from python_scripts.models.regression_models.train_gb import train_gb
from python_scripts.models.regression_models.train_knn import train_knn
from python_scripts.models.regression_models.train_mlp import train_mlp
from python_scripts.models.regression_models.train_perceptron import train_perceptron
from python_scripts.models.regression_models.train_rf import train_rf
from python_scripts.models.regression_models.train_ridge import train_ridge
from python_scripts.models.regression_models.train_svr import train_svr
from python_scripts.models.regression_models.train_bayesian_ridge import train_bayesian_ridge
from python_scripts.models.regression_models.train_ard_regression import train_ard_regression
from python_scripts.models.regression_models.train_extra_trees import train_extra_trees
from python_scripts.models.regression_models.train_adaboost_regressor import train_adaboost_regressor
from python_scripts.models.regression_models.train_bagging_regressor import train_bagging_regressor
from python_scripts.models.regression_models.train_decision_tree_regressor import train_decision_tree_regressor
from python_scripts.models.regression_models.train_elasticnet_cv import train_elasticnet_cv
from python_scripts.models.regression_models.train_hist_gradient_boosting_regressor import train_hist_gradient_boosting_regressor
from python_scripts.models.regression_models.train_huber_regressor import train_huber_regressor
from python_scripts.models.regression_models.train_lars import train_lars
from python_scripts.models.regression_models.train_lars_cv import train_lars_cv
from python_scripts.models.regression_models.train_lasso_cv import train_lasso_cv
from python_scripts.models.regression_models.train_lassolars import train_lassolars
from python_scripts.models.regression_models.train_lassolars_cv import train_lassolars_cv
from python_scripts.models.regression_models.train_linearsvr import train_linearsvr
from python_scripts.models.regression_models.train_nusvr import train_nusvr
from python_scripts.models.regression_models.train_orthogonal_matching_pursuit import train_orthogonal_matching_pursuit
from python_scripts.models.regression_models.train_passive_aggressive_regressor import train_passive_aggressive_regressor
from python_scripts.models.regression_models.train_quantile_regressor import train_quantile_regressor
from python_scripts.models.regression_models.train_radius_neighbors_regressor import train_radius_neighbors_regressor
from python_scripts.models.regression_models.train_ransac_regressor import train_ransac_regressor
from python_scripts.models.regression_models.train_ridge_cv import train_ridge_cv
from python_scripts.models.regression_models.train_sgd_regressor import train_sgd_regressor
from python_scripts.models.regression_models.train_theilsen_regressor import train_theilsen_regressor
from python_scripts.models.classify_models.train_logistic_classifier import train_logistic_classifier
from python_scripts.models.classify_models.train_mlp_classifier import train_mlp_classifier
from python_scripts.models.classify_models.train_rf_classifier import train_rf_classifier
from python_scripts.models.classify_models.train_svc import train_svc
from python_scripts.models.classify_models.train_extra_trees_classifier import train_extra_trees_classifier
from python_scripts.models.classify_models.train_gaussian_nb import train_gaussian_nb
from python_scripts.models.classify_models.train_sgd_classifier import train_sgd_classifier
from python_scripts.models.classify_models.train_adaboost_classifier import train_adaboost_classifier
from python_scripts.models.classify_models.train_bagging_classifier import train_bagging_classifier
from python_scripts.models.classify_models.train_bernoulli_nb import train_bernoulli_nb
from python_scripts.models.classify_models.train_categorical_nb import train_categorical_nb
from python_scripts.models.classify_models.train_complement_nb import train_complement_nb
from python_scripts.models.classify_models.train_decision_tree_classifier import train_decision_tree_classifier
from python_scripts.models.classify_models.train_gradient_boosting_classifier import train_gradient_boosting_classifier
from python_scripts.models.classify_models.train_hist_gradient_boosting_classifier import train_hist_gradient_boosting_classifier
from python_scripts.models.classify_models.train_kneighbors_classifier import train_kneighbors_classifier
from python_scripts.models.classify_models.train_linear_discriminant_analysis import train_linear_discriminant_analysis
from python_scripts.models.classify_models.train_linearsvc import train_linearsvc
from python_scripts.models.classify_models.train_multinomial_nb import train_multinomial_nb
from python_scripts.models.classify_models.train_nusvc import train_nusvc
from python_scripts.models.classify_models.train_passive_aggressive_classifier import train_passive_aggressive_classifier
from python_scripts.models.classify_models.train_quadratic_discriminant_analysis import train_quadratic_discriminant_analysis
from python_scripts.models.classify_models.train_ridge_classifier import train_ridge_classifier
from python_scripts.models.cluster_models.train_agglomerative import train_agglomerative
from python_scripts.models.cluster_models.train_gmm import train_gmm
from python_scripts.models.cluster_models.train_kmeans import train_kmeans
from python_scripts.models.cluster_models.train_dbscan import train_dbscan
from python_scripts.models.cluster_models.train_birch import train_birch
from python_scripts.models.cluster_models.train_spectral import train_spectral
from python_scripts.models.cluster_models.train_affinity_propagation import train_affinity_propagation
from python_scripts.models.cluster_models.train_bisecting_kmeans import train_bisecting_kmeans
from python_scripts.models.cluster_models.train_hdbscan import train_hdbscan
from python_scripts.models.cluster_models.train_meanshift import train_meanshift
from python_scripts.models.cluster_models.train_minibatch_kmeans import train_minibatch_kmeans
from python_scripts.models.cluster_models.train_optics import train_optics

DEFAULT_CV_FOLDS = 5
DEFAULT_SEARCH_ITERATIONS = 50
RANDOM_SEED_MAX = 1000

logger = logging.getLogger(__name__)

def _safe_select_columns(df, column_names_or_indices):
    """Safely select columns from a dataframe, handling duplicate column names.
    
    If column_names_or_indices contains duplicate names and the dataframe has
    duplicate columns, selects by position using the original indices.
    
    Args:
        df: DataFrame to select from
        column_names_or_indices: List of column names or indices
    
    Returns:
        DataFrame with selected columns
    """
    # Check if we have duplicate column names in the selection
    if isinstance(column_names_or_indices, list) and len(column_names_or_indices) != len(set(column_names_or_indices)):
        # Has duplicates in selection - need to use position-based selection
        # Convert names to indices
        indices = []
        for name in column_names_or_indices:
            # Find all positions where this name appears
            matches = [i for i, col in enumerate(df.columns) if col == name]
            if matches:
                indices.append(matches[0])  # Take first occurrence
            else:
                raise ValueError(f"Column '{name}' not found in dataframe")
        return df.iloc[:, indices]
    else:
        # No duplicates or single column - use normal selection
        try:
            return df[column_names_or_indices]
        except KeyError as e:
            # If selection fails due to duplicates, try position-based
            if "not unique" in str(e).lower():
                # Convert to indices and use iloc
                indices = [df.columns.get_loc(name) if isinstance(name, str) else name 
                          for name in column_names_or_indices]
                # Handle duplicate names by taking first occurrence
                seen = {}
                unique_indices = []
                for i, name in enumerate(column_names_or_indices):
                    if name not in seen:
                        seen[name] = True
                        if isinstance(name, str):
                            # Get first occurrence of this column name
                            pos = df.columns.tolist().index(name)
                            unique_indices.append(pos)
                        else:
                            unique_indices.append(name)
                return df.iloc[:, unique_indices]
            raise



def _maybe_fix_swapped_scalers(y_scaler, X_scaler, y_train_array):
    """Heuristic guard against swapped scalers.

    If the scaler stored as y_scaler looks like it was fit on X (or vice versa),
    inverse-transforming predictions will explode (e.g., target values predicted in the
    millions). This check swaps the two when that signature is detected.

    Parameters
    ----------
    y_scaler, X_scaler : sklearn-like scalers or None
    y_train_array : np.ndarray
        The training target values used to fit y_scaler.

    Returns
    -------
    (y_scaler, X_scaler)
    """
    if y_scaler is None or X_scaler is None:
        return y_scaler, X_scaler

    # Only scalers like StandardScaler/RobustScaler/MinMaxScaler expose mean_/scale_
    y_mean_attr = getattr(y_scaler, 'mean_', None)
    x_mean_attr = getattr(X_scaler, 'mean_', None)
    if y_mean_attr is None or x_mean_attr is None:
        return y_scaler, X_scaler

    try:
        y_train_mean = float(np.mean(y_train_array))
        y_train_std = float(np.std(y_train_array))
        ys_mean = float(np.mean(y_mean_attr))
        # If y_scaler's mean is wildly far from the target mean, it's likely swapped.
        if np.isfinite(ys_mean) and abs(ys_mean - y_train_mean) > 10 * max(1.0, y_train_std):
            logger.warning('y_scaler looks incorrect (likely swapped with X_scaler). Swapping them.')
            return X_scaler, y_scaler
    except Exception:
        # If anything goes wrong, do nothing rather than break the app.
        return y_scaler, X_scaler

    return y_scaler, X_scaler



def _json_safe_params(obj):
    """Recursively sanitize params from model.get_params() for JSON.
    Replaces estimators, ndarrays, and other non-serializable values with placeholders.
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe_params(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe_params(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
    return f"<{type(obj).__module__}.{type(obj).__name__}>"



def run_model_training(session_id: str, data: dict, storage_session_id: str, get_storage):
    """Run model training (session storage is obtained via get_storage(storage_session_id)).
    See app.py _run_model_training for full doc.
    """
    tracker = get_tracker(session_id)
    tracker.start()
    store = get_storage(storage_session_id)
    
    try:
        # Input validation
        required_fields = ['indicators', 'predictors', 'hyperparameters', 'models', 'nonreq', 
                          'scaler', 'units', 'sigfig', 'seedValue', 'testSize', 'stratifyBool',
                          'dropMissing', 'imputeStrategy', 'dropZero', 'quantileBinDict',
                          'useTransformer', 'transformerCols']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
        
        if 'data' not in store:
            raise ValueError("No data uploaded. Please upload a file first.")
        
        # getting all the parameters from the front end
        selected_indicators = data['indicators']
        selected_predictors = data['predictors']
        hyperparameters = data['hyperparameters']
        modelName = data['models']
        nonreq = data['nonreq']
        scaler = data['scaler']
        units = data['units']
        sigfig = data['sigfig']
        seed = data['seedValue']
        raw_test = data.get('testSize', '0.20')
        try:
            testSize = float(raw_test) if raw_test not in (None, '') else 0.2
        except (TypeError, ValueError):
            testSize = 0.2
        stratifyBool = data['stratifyBool']
        drop_missing = data['dropMissing']
        impute_strategy = data['imputeStrategy']
        drop_zero = data['dropZero']
        quantileBinDict = data['quantileBinDict']
        useTransformer = data['useTransformer']
        transformerCols = data['transformerCols']
        cross_validation_type = data.get('crossValidationType', 'None')
        cross_validation_folds = int(data.get('crossValidationFolds', DEFAULT_CV_FOLDS) or DEFAULT_CV_FOLDS)
        
        # Advanced features parameters
        hyperparameter_search = data.get('hyperparameterSearch', 'none')
        search_cv_folds = int(data.get('searchCVFolds', 5) or 5)
        search_n_iter = int(data.get('searchNIter', DEFAULT_SEARCH_ITERATIONS) or DEFAULT_SEARCH_ITERATIONS)
        feature_selection_method = data.get('featureSelectionMethod', 'none')
        feature_selection_k = data.get('featureSelectionK')
        outlier_method = data.get('outlierMethod', 'none')
        outlier_action = data.get('outlierAction', 'remove')
        modeling_mode = data.get('modelingMode', 'simple')  # 'simple', 'advanced', or 'automl'
        
        # Normalize cross_validation_type for comparison (handle both 'None' and 'none')
        cv_type_normalized = str(cross_validation_type).strip() if cross_validation_type else 'None'
        cv_enabled = cv_type_normalized.lower() not in ['none', '']
        
        # Set up progress tracking based on selected features
        tracker.set_stage_enabled('outlier_handling', outlier_method != 'none')
        tracker.set_stage_enabled('feature_selection', feature_selection_method != 'none')
        tracker.set_stage_enabled('hyperparameter_search', hyperparameter_search != 'none')
        tracker.set_stage_enabled('cross_validation', cv_enabled)
        
        # Advanced features are now implemented - log when used
        if hyperparameter_search != 'none':
            logger.info(f"Using hyperparameter search ({hyperparameter_search}) with {search_cv_folds} CV folds.")
            tracker.update_stage('hyperparameter_search', 'pending', 0, 
                               f'Will search {search_n_iter} iterations with {search_cv_folds} CV folds')
        if feature_selection_method != 'none':
            logger.info(f"Using feature selection ({feature_selection_method}) with k={feature_selection_k}.")
            tracker.update_stage('feature_selection', 'pending', 0, 
                               f'Selecting {feature_selection_k} features using {feature_selection_method}')
        if outlier_method != 'none':
            logger.info(f"Using outlier handling ({outlier_method}) with action: {outlier_action}.")
            tracker.update_stage('outlier_handling', 'pending', 0, 
                               f'Detecting outliers using {outlier_method} ({outlier_action})')
        if cv_enabled:
            logger.info(f"Using cross-validation ({cross_validation_type}) with {cross_validation_folds} folds.")
            tracker.update_stage('cross_validation', 'pending', 0, 
                               f'Will run {cross_validation_type} with {cross_validation_folds} folds')
        
        # Data preprocessing
        tracker.update_stage('data_preprocessing', 'running', 10, 'Loading data...')
        df = store['data']
        
        # Select columns by position to avoid issues with duplicate column names
        # Convert indices to list if single value, ensure they're integers
        if isinstance(selected_predictors, (int, np.integer)):
            selected_predictors = [selected_predictors]
        if isinstance(selected_indicators, (int, np.integer)):
            selected_indicators = [selected_indicators]
        
        try:
            # Use take() to select by position, then convert to list to avoid duplicate name issues
            predictor_names = df.columns.take(selected_predictors).tolist()
            indicator_names = df.columns.take(selected_indicators).tolist()
        except (KeyError, IndexError) as e:
            # Fallback to direct indexing if take() fails
            try:
                predictor_names = df.columns[selected_predictors].tolist()
                indicator_names = df.columns[selected_indicators].tolist()
            except Exception as e2:
                raise ValueError(f"Error selecting columns: {str(e2)}. This may be due to duplicate column names in your CSV file. Please ensure all column names are unique.")

        # Persist inference settings so /predict can apply the same preprocessing choices.
        store['inference_config'] = {
            'indicator_names': list(indicator_names),
            'predictor_names': list(predictor_names),
            'drop_missing': drop_missing,
            'impute_strategy': impute_strategy,
            'drop_zero': drop_zero,
            'useTransformer': useTransformer,
        }
        
        tracker.update_stage('data_preprocessing', 'running', 30, 'Preparing features and targets...')

        transformer_names = []
        if useTransformer == 'Yes':
            if isinstance(transformerCols, (int, np.integer)):
                transformerCols = [transformerCols]
            try:
                transformer_names = df.columns.take(transformerCols).tolist()
            except (KeyError, IndexError):
                transformer_names = df.columns[transformerCols].tolist()

        stratifyColumn = ''
        stratify_name = ''
        if stratifyBool:
            if 'stratifyColumn' not in data:
                raise ValueError("Missing required field: stratifyColumn (required when stratifyBool is True)")
            stratifyColumn = data['stratifyColumn']
            try:
                if isinstance(stratifyColumn, (int, np.integer)):
                    stratify_name = df.columns.take([stratifyColumn]).tolist()[0]
                else:
                    stratify_name = df.columns[stratifyColumn]
                    if isinstance(stratify_name, pd.Index):
                        stratify_name = stratify_name.tolist()[0] if len(stratify_name) > 0 else stratify_name
            except (KeyError, IndexError):
                stratify_name = df.columns[stratifyColumn]
                if isinstance(stratify_name, pd.Index):
                    stratify_name = stratify_name.tolist()[0] if len(stratify_name) > 0 else stratify_name
            if stratify_name and predictor_names and stratify_name in predictor_names:
                raise ValueError(
                    "Do not stratify by your target column. That would leak target information into the train/test split. Choose a different column to stratify by."
                )

        if seed:  # if user gives seed add to storage
            store['seed'] = seed
        else:  # no seed given
            # if 'seed' not in memStorage.keys(): #if seed not already saved then add random seed
            seed = random.randint(0, RANDOM_SEED_MAX)
            store['seed'] = seed
            #else: 
            #    seed = memStorage['seed']   #else use prior randomly generated seed or keep generating random seed? - ask rowan need way to generate new random seed
            #if given seed then given no seed, want new random

        tracker.update_stage('data_preprocessing', 'running', 50, 'Cleaning and preprocessing data...')
    ## Preprocessing
        df = preprocess_data(df=df, target_cols=predictor_names, indicator_cols=indicator_names, drop_missing=drop_missing, impute_strategy=impute_strategy, drop_zero=drop_zero)
        tracker.update_stage('data_preprocessing', 'completed', 100, 'Data preprocessing complete')
        
        # Update model training stage to indicate we're starting
        tracker.update_stage('model_training', 'running', 0, f'Initializing {modelName} model training...')
        quantileBin_results = ''

        if modelName == 'TerraFORMER':
            model

    ## Selecting Model 
    ## Sending targets, indicators, hyperparameters, scaler, stratify yes/no, quantiles/bins, transformers, sigfig, seed/test size to the train_model script
            ## Every train_model script calls the run_classification/clustering/regression_pipeline script
            
        #Regression models
        elif modelName == 'Linear': 
            #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order = train(modelName, predictor_names, df, stratifyBool, df[indicator_names], df[predictor_names], stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig)
            # Use safe column selection to handle potential duplicate column names
            X_data = _safe_select_columns(df, indicator_names)
            y_data = _safe_select_columns(df, predictor_names)
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_linear(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=X_data, y=y_data, stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'BayesianRidge':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_bayesian_ridge(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'ARDRegression':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_ard_regression(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'Ridge': 
            if nonreq:
                RidgeFitIntersept = True
                RidgeNormalize = True
                RidgeCopyX = True
                RidgePositive = True
                if hyperparameters['RidgeFitIntersept'] == 'false':
                    RidgeFitIntersept = False
                if hyperparameters['RidgeNormalize'] == 'false':
                    RidgeNormalize = False
                if hyperparameters['RidgeCopyX'] == 'false':
                    RidgeCopyX = False
                if hyperparameters['RidgePositive'] == 'false':
                    RidgePositive = False
                            
                #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name,  units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_ridge(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize,
                            alpha=hyperparameters['alpha'], 
                            solver=hyperparameters['solver'],
                            RidgeFitIntersept = RidgeFitIntersept,
                            RidgeNormalize = RidgeNormalize,
                            RidgeCopyX = RidgeCopyX,
                            RidgePositive = RidgePositive,
                            RidgeMaxIter = hyperparameters['RidgeMaxIter'],
                            RidgeTol = hyperparameters['RidgeTol'],
                            RidgeRandomState = seed,
                            feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
                            outlier_method=outlier_method, outlier_action=outlier_action,
                            hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
                            progress_tracker=tracker, modeling_mode=modeling_mode
                            )
            else: 
                #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_ridge(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize,
                            alpha=hyperparameters['alpha'],
                            RidgeRandomState = seed,
                            feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
                            outlier_method=outlier_method, outlier_action=outlier_action,
                            hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
                            progress_tracker=tracker)

        elif modelName == 'Lasso': 
            if nonreq: 
                LassoFitIntersept = True
                LassoPrecompute = True
                LassoCopyX = True
                LassoWarmStart = True
                LassoPositive = True
                if hyperparameters['LassoFitIntersept'] == 'false':
                    LassoFitIntersept = False
                if hyperparameters['LassoPrecompute'] == 'false':
                    LassoPrecompute = False
                if hyperparameters['LassoCopyX'] == 'false':
                    LassoCopyX = False
                if hyperparameters['LassoWarmStart'] == 'false':
                    LassoWarmStart = False
                if hyperparameters['LassoPositive'] == 'false':
                    LassoPositive = False

                #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_lasso(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize,
                            alpha = hyperparameters['alpha'], 
                            max_iter = hyperparameters['max_iter'],
                            fit_intercept = LassoFitIntersept,
                            precompute = LassoPrecompute,
                            copy_X = LassoCopyX,
                            tol = hyperparameters['LassoTol'],
                            warm_start = LassoWarmStart,
                            positive = LassoPositive,
                            random_state = seed,
                            selection = hyperparameters['LassoSelection'],
                            feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
                            outlier_method=outlier_method, outlier_action=outlier_action,
                            hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
                            progress_tracker=tracker, modeling_mode=modeling_mode
                            )
            else:
                #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_lasso(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize,
                            alpha=hyperparameters['alpha'],
                            random_state = seed,
                            feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
                            outlier_method=outlier_method, outlier_action=outlier_action,
                            hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
                            progress_tracker=tracker, modeling_mode=modeling_mode
                            )

        elif modelName == 'ElasticNet': 
            #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_elasticnet(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize,
                        alpha=hyperparameters['alpha'], 
                        l1_ratio=hyperparameters['l1_ratio'],
                        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
                        outlier_method=outlier_method, outlier_action=outlier_action,
                        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
                        progress_tracker=tracker)

        elif modelName == 'SVM': #precomputed kernal doesn't work - weird with multiregressor
            if nonreq:
                SVMshrinking = True
                SVMprobability = True
                SVMBreakTies = True
                SVMverbose = True
                if hyperparameters['SVMshrinking'] == 'false':
                    SVMshrinking = False
                if hyperparameters['SVMprobability'] == 'false':
                    SVMprobability = False
                if hyperparameters['SVMBreakTies'] == 'false':
                    SVMBreakTies = False
                if hyperparameters['SVMverbose'] == 'false':
                    SVMverbose = False

                kernel = hyperparameters['kernel']
                if kernel =='rbf':
                    #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                    train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_svr(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker,
                                C=hyperparameters['C'], 
                                kernel=kernel,  
                                gamma=hyperparameters['gamma'],
                                
                                    coef0=hyperparameters['SVMcoef0'],
                                    shrinking=SVMshrinking,
                                    probability=SVMprobability,
                                    tol=hyperparameters['SVMtol'],
                                    cache_size=hyperparameters['SVMCacheSize'],
                                    class_weight=hyperparameters['SVMClassWeight'],
                                    verbose=SVMverbose,
                                    max_iter=hyperparameters['SVMmaxIter'],
                                    decision_function_shape=hyperparameters['SVMdecisionFunctionShape'],
                                    break_ties=SVMBreakTies,
                                    random_state=seed,
                                    )
                    
                elif kernel =='poly':
                    #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                    train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_svr(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker,
                                C=hyperparameters['C'], 
                                kernel=kernel, 
                                degree=hyperparameters['degree'], 
                                gamma=hyperparameters['gamma'],
                                coef0=hyperparameters['SVMcoef0'],
                                    shrinking=SVMshrinking,
                                    probability=SVMprobability,
                                    tol=hyperparameters['SVMtol'],
                                    cache_size=hyperparameters['SVMCacheSize'],
                                    class_weight=hyperparameters['SVMClassWeight'],
                                    verbose=SVMverbose,
                                    max_iter=hyperparameters['SVMmaxIter'],
                                    decision_function_shape=hyperparameters['SVMdecisionFunctionShape'],
                                    break_ties=SVMBreakTies,
                                    random_state=seed,)
                else:
                    #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                    train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_svr(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker,
                                C=hyperparameters['C'], 
                                kernel=kernel,
                                coef0=hyperparameters['SVMcoef0'],
                                    shrinking=SVMshrinking,
                                    probability=SVMprobability,
                                    tol=hyperparameters['SVMtol'],
                                    cache_size=hyperparameters['SVMCacheSize'],
                                    class_weight=hyperparameters['SVMClassWeight'],
                                    verbose=SVMverbose,
                                    max_iter=hyperparameters['SVMmaxIter'],
                                    decision_function_shape=hyperparameters['SVMdecisionFunctionShape'],
                                    break_ties=SVMBreakTies,
                                    random_state=seed,)

            else:
                kernel = hyperparameters['kernel']
                if kernel =='rbf':
                    #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, 
                    train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_svr(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker,
                        C=hyperparameters['C'], kernel=kernel,  gamma=hyperparameters['gamma'],
                            random_state = seed)
                elif kernel =='poly':
                    #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, 
                    train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_svr(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker,
                        C=hyperparameters['C'], kernel=kernel, degree=hyperparameters['degree'], gamma=hyperparameters['gamma'],
                            random_state = seed)
                else:
                    #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, 
                    train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_svr(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker,
                        C=hyperparameters['C'], kernel=kernel,
                            random_state = seed)

        elif modelName == 'RF': 
            if nonreq:
                RFBoostrap = True
                RFoobScore = True
                RFWarmStart = True
                if hyperparameters['RFBoostrap'] == 'false':
                    RFBoostrap = True
                if hyperparameters['RFoobScore'] == 'false':
                    RFoobScore = True
                if hyperparameters['RFWarmStart'] == 'false':
                    RFWarmStart = True

                val = None
                if 'max_depth' in hyperparameters.keys():
                    val = hyperparameters['max_depth']
                #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_rf(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize,
                            n_estimators=hyperparameters['n_estimators'], 
                            max_depth=val, 
                            min_samples_split=hyperparameters['min_samples_split'], 
                            min_samples_leaf=hyperparameters['min_samples_leaf'], 
                            random_state = seed,
                            min_weight_fraction_leaf=hyperparameters['RFmin_weight_fraction_leaf'],
                            max_leaf_nodes=hyperparameters['RFMaxLeafNodes'],
                            min_impurity_decrease=hyperparameters['RFMinImpurityDecrease'],
                            bootstrap=RFBoostrap,
                            oob_score=RFoobScore,
                            n_jobs=hyperparameters['RFNJobs'],
                            verbose=hyperparameters['RFVerbose'],
                            warm_start=RFWarmStart,
                            feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
                            outlier_method=outlier_method, outlier_action=outlier_action,
                            hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
                            progress_tracker=tracker, modeling_mode=modeling_mode
                            )
            else:
                val = None
                if 'max_depth' in hyperparameters.keys():
                    val = hyperparameters['max_depth']
                #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_rf(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize,
                        n_estimators=hyperparameters['n_estimators'],
                        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
                        outlier_method=outlier_method, outlier_action=outlier_action,
                        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
                        progress_tracker=tracker
                        )

        elif modelName == 'ExtraTrees':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_extra_trees(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        # NOTE: LogisticRegression is a classifier, not a regressor. It has been removed from regression section.
        # Use Logistic_classifier instead for classification tasks.

        elif modelName == 'MLP': 
            if nonreq:
                    MLPShuffle = True
                    MLPVerbose = True
                    MLPWarmStart = True
                    MLPNesterovsMomentum = True
                    MLPEarlyStopping = True
                    if hyperparameters['MLPShuffle']=='false':
                        MLPShuffle= False
                    if hyperparameters['MLPVerbose']=='false':
                        MLPVerbose= False
                    if hyperparameters['MLPWarmStart']=='false':
                        MLPWarmStart= False
                    if hyperparameters['MLPNesterovsMomentum']=='false':
                        MLPNesterovsMomentum= False
                    if hyperparameters['MLPEarlyStopping']=='false':
                        MLPEarlyStopping= False


                    hiddenlayersizeString = '(' + hyperparameters['hidden_layer_sizes1'] + ',' + hyperparameters['hidden_layer_sizes2']
                    if hyperparameters['hidden_layer_sizes3']:
                        hiddenlayersizeString += ',' + hyperparameters['hidden_layer_sizes3'] + ')'
                    else:
                        hiddenlayersizeString += ')'
                    # if not hyperparameters['hidden_layer_sizes1']:
                    #     hiddenlayersizeString = None
                    
                    #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler, quantileBin_results = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler,  seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                    train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_mlp(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize,
                            feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
                            outlier_method=outlier_method, outlier_action=outlier_action,
                            hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
                                hidden_layer_sizes=ast.literal_eval(hiddenlayersizeString), 
                                activation=hyperparameters['activation'], 
                                solver=hyperparameters['solver'], 
                                alpha=hyperparameters['alpha'], 
                                learning_rate=hyperparameters['learning_rate'], 
                                max_iter=hyperparameters['MLPMaxIter'],
                                batch_size=hyperparameters['MLPBatchSize'],
                                beta_1=hyperparameters['MLPBeta1'],
                                beta_2=hyperparameters['MLPBeta2'],
                                early_stopping=MLPEarlyStopping,
                                epsilon=hyperparameters['MLPEpsilon'],
                                learning_rate_init=hyperparameters['MLPLearningRateInit'],
                                momentum=hyperparameters['MLPMomentum'],
                                nesterovs_momentum=MLPNesterovsMomentum,
                                power_t=hyperparameters['MLPPowerT'],
                                random_state=seed,
                                shuffle=MLPShuffle,
                                tol=hyperparameters['MLPTol'],
                                validation_fraction=hyperparameters['MLPValidationFraction'],
                                verbose=MLPVerbose,
                                warm_start=MLPWarmStart
                    )
                    #self.model=MLPRegressor()
            
            else:
                hiddenlayersizeString = '(' + hyperparameters['hidden_layer_sizes1'] + ',' + hyperparameters['hidden_layer_sizes2']
                if hyperparameters['hidden_layer_sizes3']:
                    hiddenlayersizeString += ',' + hyperparameters['hidden_layer_sizes3'] + ')'
                else:
                    hiddenlayersizeString += ')'
                # if not hyperparameters['hidden_layer_sizes1']:
                #     hiddenlayersizeString = None
                
                #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler, quantileBin_results = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_mlp(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
                            progress_tracker=tracker,
                            hidden_layer_sizes=ast.literal_eval(hiddenlayersizeString), 
                            activation=hyperparameters['activation'], 
                            solver=hyperparameters['solver'],
                            random_state = seed
                            )
                #self.model=MLPRegressor()

        # NOTE: Perceptron is a classifier, not a regressor. It has been removed from regression section.
        # Use Perceptron_classifier instead for classification tasks.

        elif modelName == 'K-Nearest': 
            if nonreq:
                metricParams = None
                if hyperparameters['KNearestMetricParams'] != '':
                    metricParams = hyperparameters['KNearestMetricParams']

                #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler,  seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_knn(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker,
                            n_neighbors=hyperparameters['n_neighbors'], 
                            metric=hyperparameters['metric'],
                            algorithm=hyperparameters['KNearestAlgorithm'],
                            leaf_size=hyperparameters['KNearestLeafSize'],
                            metric_params=metricParams,
                            n_jobs=hyperparameters['KNearestNJobs'],
                            p=hyperparameters['KNearestP'],
                            weights=hyperparameters['KNearestWeights'],
                        )

            else: 
                #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_knn(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker,
                        n_neighbors=hyperparameters['n_neighbors'], 
                        )

        elif modelName == 'gradient_boosting': #weird with multiregressor
            if nonreq:
                GBWarmStart = True
                if hyperparameters['GBWarmStart']:
                    GBWarmStart=False

                init=None
                if hyperparameters['GBInit']!='':
                    init=hyperparameters['GBInit']

                max_features=None
                if hyperparameters['GBMaxFeatrues']!='':
                    max_features=hyperparameters['GBMaxFeatrues']

                #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_gb(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker,
                        n_estimators=hyperparameters['n_estimators'], 
                        learning_rate=hyperparameters['learning_rate'], 
                        max_depth=hyperparameters['max_depth'], 
                        loss=hyperparameters['GBLoss'], 
                        subsample=hyperparameters['GBSubsample'], 
                        criterion=hyperparameters['GBCriterion'], 
                        min_samples_split=hyperparameters['GBMinSamplesSplit'], 
                        min_samples_leaf=hyperparameters['GBMinSamplesLeaf'], 
                        min_weight_fraction_leaf=hyperparameters['GBMinWeightFractionLeaf'], 
                        min_impurity_decrease=hyperparameters['GBMinImpurityDecrease'], 
                        init=init,
                        random_state=seed, 
                        max_features=max_features, 
                        alpha=hyperparameters['GBAlpha'], 
                        verbose=hyperparameters['GBVerbose'], 
                        max_leaf_nodes=hyperparameters['GBMaxLeafNodes'], 
                        warm_start=GBWarmStart
                        
                        )
            else:
                #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_gb(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker,
                        n_estimators=hyperparameters['n_estimators'], 
                        learning_rate=hyperparameters['learning_rate'],
                        random_state = seed
                        )

        # Additional Regression Models
        elif modelName == 'AdaBoost':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_adaboost_regressor(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'Bagging':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_bagging_regressor(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'DecisionTree':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_decision_tree_regressor(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'ElasticNetCV':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_elasticnet_cv(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'HistGradientBoosting':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_hist_gradient_boosting_regressor(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'Huber':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_huber_regressor(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'LARS':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_lars(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'LARSCV':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_lars_cv(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'LassoCV':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_lasso_cv(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'LassoLars':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_lassolars(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'LinearSVR':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_linearsvr(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'NuSVR':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_nusvr(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'OMP':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_orthogonal_matching_pursuit(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'PassiveAggressive':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_passive_aggressive_regressor(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'Quantile':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_quantile_regressor(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'RadiusNeighbors':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_radius_neighbors_regressor(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'RANSAC':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_ransac_regressor(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'RidgeCV':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_ridge_cv(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'SGD':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_sgd_regressor(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'TheilSen':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_theilsen_regressor(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        #Special Model
        elif modelName == 'Polynomial': #set up
            model = Model(model = modelName, 
                        degree=hyperparameters['degree_specificity'])

        #Classifier Models


        elif modelName == 'Perceptron': #done with nonreq but cant test
            if nonreq: 
                PerceptronFitIntercept = True
                PerceptronShuffle = True
                PerceptronEarlyStopping = True
                PerceptronWarmStart = True
                if hyperparameters['PerceptronFitIntercept'] == 'false':
                    PerceptronFitIntercept=False
                if hyperparameters['PerceptronShuffle'] == 'false':
                    PerceptronShuffle=False
                if hyperparameters['PerceptronEarlyStopping'] == 'false':
                    PerceptronEarlyStopping=False
                if hyperparameters['PerceptronWarmStart'] == 'false':
                    PerceptronWarmStart=False
                    
                model =  model = Model(model = modelName, 
                        max_iter=hyperparameters['max_iter'], 
                        eta0=hyperparameters['eta0'], 
                            penalty=hyperparameters['PerceptronPenalty'],
                            alpha=hyperparameters['PerceptronAlpha'],
                            fit_intercept=PerceptronFitIntercept,
                            tol=hyperparameters['PerceptronTol'],
                            shuffle=PerceptronShuffle,
                            verbose=hyperparameters['PerceptronVerbose'],
                            n_jobs=hyperparameters['PerceptronNJobs'],
                            random_state=seed,
                            early_stopping=PerceptronEarlyStopping,
                            validation_fraction=hyperparameters['PerceptronValidationFraction'],
                            n_iter_no_change=hyperparameters['PerceptronNIterNoChange'],
                            class_weight=hyperparameters['PerceptronClassWeight'],
                            warm_start=PerceptronWarmStart
                        )

            else:
                model = Model(model = modelName, 
                            max_iter=hyperparameters['max_iter'], 
                            eta0=hyperparameters['eta0'], 
                            )

        elif modelName == 'Logistic_classifier':
            if nonreq:
                Class_LogisticDual = True
                Class_LogisticFitIntercept = True
                Class_LogisticWarmStart = True
                if hyperparameters['Class_LogisticDual']=='false':
                    Class_LogisticDual = False
                if hyperparameters['Class_LogisticFitIntercept']=='false':
                    Class_LogisticFitIntercept = False
                if hyperparameters['Class_LogisticWarmStart']=='false':
                    Class_LogisticWarmStart = False

                Class_LogisticClassWeight = None
                Class_LogisticNJobs = None
                Class_Logisticl1Ratio = None
                if hyperparameters['Class_LogisticClassWeight']!='':
                    Class_LogisticClassWeight = hyperparameters['Class_LogisticClassWeight']
                if hyperparameters['Class_LogisticNJobs']!='':
                    Class_LogisticNJobs = hyperparameters['Class_LogisticNJobs']
                if hyperparameters['Class_Logisticl1Ratio']!='':
                    Class_Logisticl1Ratio = hyperparameters['Class_Logisticl1Ratio']

                result_tuple = train_logistic_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
                        Class_LogisticDual = Class_LogisticDual,
                        Class_LogisticFitIntercept = Class_LogisticFitIntercept,
                        Class_LogisticWarmStart = Class_LogisticWarmStart,
                        Class_LogisticSolver = hyperparameters['Class_LogisticSolver'],
                        Class_LogisticMultiClass = hyperparameters['Class_LogisticMultiClass'],
                        Class_CLogistic = hyperparameters['Class_CLogistic'],
                        Class_Logistic_penalty = hyperparameters['Class_Logistic_penalty'],
                        Class_LogisticTol = hyperparameters['Class_LogisticTol'],
                        Class_Logisticintercept_scaling = hyperparameters['Class_Logisticintercept_scaling'],
                        Class_LogisticClassWeight = Class_LogisticClassWeight,
                        Class_LogisticMaxIterations = hyperparameters['Class_LogisticMaxIterations'],
                        Class_LogisticVerbose = hyperparameters['Class_LogisticVerbose'],
                        Class_LogisticNJobs = Class_LogisticNJobs,
                        Class_Logisticl1Ratio = Class_Logisticl1Ratio, )                                                                                                                                                                                                                                                                                        

            else:
                result_tuple = train_logistic_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
                if len(result_tuple) >= 9:
                    report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = result_tuple
                else:
                    report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order = result_tuple[:8]
                    additional_metrics = None
        
        elif modelName == 'ExtraTrees_classifier':
            result_tuple = train_extra_trees_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)

        elif modelName == 'GaussianNB_classifier':
            result_tuple = train_gaussian_nb(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)

        elif modelName == 'SGD_classifier':
            result_tuple = train_sgd_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)

        elif modelName == 'MLP_classifier':  
            if nonreq:
                    MLPShuffle = True
                    MLPVerbose = True
                    MLPWarmStart = True
                    MLPNesterovsMomentum = True
                    MLPEarlyStopping = True
                    if hyperparameters['MLPShuffle']=='false':
                        MLPShuffle= False
                    if hyperparameters['MLPVerbose']=='false':
                        MLPVerbose= False
                    if hyperparameters['MLPWarmStart']=='false':
                        MLPWarmStart= False
                    if hyperparameters['MLPNesterovsMomentum']=='false':
                        MLPNesterovsMomentum= False
                    if hyperparameters['MLPEarlyStopping']=='false':
                        MLPEarlyStopping= False


                    hiddenlayersizeString = '(' + hyperparameters['hidden_layer_sizes1'] + ',' + hyperparameters['hidden_layer_sizes2']
                    if hyperparameters['hidden_layer_sizes3']:
                        hiddenlayersizeString += ',' + hyperparameters['hidden_layer_sizes3'] + ')'
                    else:
                        hiddenlayersizeString += ')'

                    result_tuple = train_mlp_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize,
                                hidden_layer_sizes=ast.literal_eval(hiddenlayersizeString), 
                                activation=hyperparameters['activation'], 
                                solver=hyperparameters['solver'], 
                                alpha=hyperparameters['alpha'], 
                                learning_rate=hyperparameters['learning_rate'], 
                                max_iter=hyperparameters['MLPMaxIter'],
                                batch_size=hyperparameters['MLPBatchSize'],
                                beta_1=hyperparameters['MLPBeta1'],
                                beta_2=hyperparameters['MLPBeta2'],
                                early_stopping=MLPEarlyStopping,
                                epsilon=hyperparameters['MLPEpsilon'],
                                learning_rate_init=hyperparameters['MLPLearningRateInit'],
                                momentum=hyperparameters['MLPMomentum'],
                                nesterovs_momentum=MLPNesterovsMomentum,
                                power_t=hyperparameters['MLPPowerT'],
                                random_state=seed,
                                shuffle=MLPShuffle,
                                tol=hyperparameters['MLPTol'],
                                validation_fraction=hyperparameters['MLPValidationFraction'],
                                verbose=MLPVerbose,
                                warm_start=MLPWarmStart)
                    report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)
            else:
                hiddenlayersizeString = '(' + hyperparameters['hidden_layer_sizes1'] + ',' + hyperparameters['hidden_layer_sizes2']
                if hyperparameters['hidden_layer_sizes3']:
                    hiddenlayersizeString += ',' + hyperparameters['hidden_layer_sizes3'] + ')'
                else:
                        hiddenlayersizeString += ')'
                result_tuple = train_mlp_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode,
                            hidden_layer_sizes=ast.literal_eval(hiddenlayersizeString), 
                            activation=hyperparameters['activation'], 
                            solver=hyperparameters['solver'],
                            random_state = seed    )
                report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)

        elif modelName == 'RF_classifier':
            if nonreq:
                RFBoostrap = True
                RFoobScore = True
                RFWarmStart = True
                if hyperparameters['RFBoostrap'] == 'false':
                    RFBoostrap = True
                if hyperparameters['RFoobScore'] == 'false':
                    RFoobScore = True
                if hyperparameters['RFWarmStart'] == 'false':
                    RFWarmStart = True

                val = None
                if 'max_depth' in hyperparameters.keys():
                    val = hyperparameters['max_depth']

                result_tuple = train_rf_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
                            n_estimators=hyperparameters['n_estimators'], 
                            max_depth=val, 
                            min_samples_split=hyperparameters['min_samples_split'], 
                            min_samples_leaf=hyperparameters['min_samples_leaf'], 
                            random_state = seed,
                            min_weight_fraction_leaf=hyperparameters['RFmin_weight_fraction_leaf'],
                            max_leaf_nodes=hyperparameters['RFMaxLeafNodes'],
                            min_impurity_decrease=hyperparameters['RFMinImpurityDecrease'],
                            bootstrap=RFBoostrap,
                            oob_score=RFoobScore,
                            n_jobs=hyperparameters['RFNJobs'],
                            verbose=hyperparameters['RFVerbose'],
                            warm_start=RFWarmStart,
                            )
                report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)
            else:
                result_tuple = train_rf_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
                            n_estimators=hyperparameters['n_estimators'], )
                report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)
       

        elif modelName == 'SVC_classifier':
            if nonreq:
                SVMshrinking = True
                SVMprobability = True
                SVMBreakTies = True
                SVMverbose = True
                if hyperparameters['SVCshrinking'] == 'false':
                    SVMshrinking = False
                if hyperparameters['SVCprobability'] == 'false':
                    SVMprobability = False
                if hyperparameters['SVCBreakTies'] == 'false':
                    SVMBreakTies = False
                if hyperparameters['SVCverbose'] == 'false':
                    SVMverbose = False
                svc_class_weight = _parse_class_weight(hyperparameters.get('SVCClassWeight'))

                kernel = hyperparameters['kernel']
                if kernel =='rbf':
                    result_tuple = train_svc(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize,
                                C=hyperparameters['C'], 
                                kernel=kernel,  
                                gamma=hyperparameters['gamma'],
                                coef0=hyperparameters['SVCcoef0'],
                                shrinking=SVMshrinking,
                                probability=SVMprobability,
                                tol=hyperparameters['SVCtol'],
                                cache_size=hyperparameters['SVCCacheSize'],
                                class_weight=svc_class_weight,
                                verbose=SVMverbose,
                                max_iter=hyperparameters['SVCmaxIter'],
                                decision_function_shape=hyperparameters['SVCdecisionFunctionShape'],
                                break_ties=SVMBreakTies,
                                random_state=seed,
                                )
                    report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)

                    
                elif kernel =='poly':
                    result_tuple = train_svc(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize,
                            C=hyperparameters['C'], 
                            kernel=kernel, 
                            degree=hyperparameters['degree'], 
                            gamma=hyperparameters['gamma'],
                            coef0=hyperparameters['SVCcoef0'],
                            shrinking=SVMshrinking,
                            probability=SVMprobability,
                            tol=hyperparameters['SVCtol'],
                            cache_size=hyperparameters['SVCCacheSize'],
                            class_weight=svc_class_weight,
                            verbose=SVMverbose,
                            max_iter=hyperparameters['SVCmaxIter'],
                            decision_function_shape=hyperparameters['SVCdecisionFunctionShape'],
                            break_ties=SVMBreakTies,
                            random_state=seed,)
                    report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)

                else:
                    result_tuple = train_svc(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize,
                                C=hyperparameters['C'], 
                                kernel=kernel, 
                                degree=hyperparameters['degree'], 
                                gamma=hyperparameters['gamma'],
                                coef0=hyperparameters['SVCcoef0'],
                                shrinking=SVMshrinking,
                                probability=SVMprobability,
                                tol=hyperparameters['SVCtol'],
                                cache_size=hyperparameters['SVCCacheSize'],
                                class_weight=svc_class_weight,
                                verbose=SVMverbose,
                                max_iter=hyperparameters['SVCmaxIter'],
                                decision_function_shape=hyperparameters['SVCdecisionFunctionShape'],
                                break_ties=SVMBreakTies,
                                random_state=seed,)
                    report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)


            else:
                kernel = hyperparameters['kernel']
                if kernel =='rbf':
                    result_tuple = train_svc(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode,
                            C=hyperparameters['C'], kernel=kernel,  gamma=hyperparameters['gamma'],
                            random_state = seed)
                    report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)

                elif kernel =='poly':
                    result_tuple = train_svc(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode,
                            C=hyperparameters['C'], kernel=kernel, degree=hyperparameters['degree'], gamma=hyperparameters['gamma'],
                            random_state = seed)
                    report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)
                else:
                    result_tuple = train_svc(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode,
                            C=hyperparameters['C'], kernel=kernel,
                            random_state = seed)
                    report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)

        # Additional Classification Models
        elif modelName == 'AdaBoost_classifier':
            result_tuple = train_adaboost_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)

        elif modelName == 'Bagging_classifier':
            result_tuple = train_bagging_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)

        elif modelName == 'BernoulliNB_classifier':
            result_tuple = train_bernoulli_nb(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)

        elif modelName == 'CategoricalNB_classifier':
            result_tuple = train_categorical_nb(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)

        elif modelName == 'ComplementNB_classifier':
            result_tuple = train_complement_nb(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)

        elif modelName == 'DecisionTree_classifier':
            result_tuple = train_decision_tree_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)

        elif modelName == 'GradientBoosting_classifier':
            result_tuple = train_gradient_boosting_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)

        elif modelName == 'HistGradientBoosting_classifier':
            result_tuple = train_hist_gradient_boosting_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)

        elif modelName == 'KNeighbors_classifier':
            result_tuple = train_kneighbors_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)

        elif modelName == 'LDA_classifier':
            result_tuple = train_linear_discriminant_analysis(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)

        elif modelName == 'LinearSVC_classifier':
            result_tuple = train_linearsvc(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)

        elif modelName == 'MultinomialNB_classifier':
            result_tuple = train_multinomial_nb(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)

        elif modelName == 'NuSVC_classifier':
            result_tuple = train_nusvc(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)

        elif modelName == 'PassiveAggressive_classifier':
            result_tuple = train_passive_aggressive_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)

        elif modelName == 'QDA_classifier':
            result_tuple = train_quadratic_discriminant_analysis(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)

        elif modelName == 'Ridge_classifier':
            result_tuple = train_ridge_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result_tuple)

        #Cluster Models 
        elif modelName == 'kmeans':
            if nonreq:
                copy_x = True
                if hyperparameters['copy_x']=="false":
                    copy_x=False

                if hyperparameters['n_init']=='auto':
                    n_init='auto'
                elif hyperparameters['n_init']=='warn':
                    n_init='warn'
                else:
                    n_init = int(hyperparameters['n_init'])

                train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid  =  train_kmeans(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize, k_min=2, k_max=8,
                                n_clusters=hyperparameters['n_clusters'],
                                init=hyperparameters['init'],
                                n_init=n_init,
                                max_iter=hyperparameters['max_iter'],
                                tol=hyperparameters['tol'],
                                verbose=hyperparameters['verbose'],
                                copy_x=copy_x,
                                algorithm=hyperparameters['algorithm'],)
        
            else:
                train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid  =  train_kmeans(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize, k_min=2, k_max=8,
                                n_clusters=hyperparameters['n_clusters'])
        
        elif modelName == 'gmm':
            if nonreq:
                warm_start = True
                if hyperparameters['warm_start']=='false':
                    warm_start=False

                if hyperparameters['weights_init']=='':
                    weights_init=None
                else:
                    weights_init=hyperparameters['weights_init']

                if hyperparameters['means_init']=='':
                    means_init=None
                else:
                    means_init=hyperparameters['means_init']

                if hyperparameters['precisions_init']=='':
                    precisions_init=None
                else:
                    precisions_init=hyperparameters['precisions_init']

                train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid = train_gmm(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize, k_min=2, k_max=8,
                                n_components=hyperparameters['n_components'],
                                covariance_type=hyperparameters['covariance_type'],
                                tol=hyperparameters['tol'],
                                reg_covar=hyperparameters['reg_covar'],
                                max_iter=hyperparameters['max_iter'],
                                n_init=hyperparameters['n_init'],
                                init_params=hyperparameters['init_params'],
                                weights_init=weights_init,
                                means_init=means_init,
                                precisions_init=precisions_init,
                                warm_start=warm_start,
                                verbose=hyperparameters['verbose'],
                                verbose_interval=hyperparameters['verbose_interval'],
                                )
                
            else:
                train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid = train_gmm(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize, k_min=2, k_max=8,
                                n_components=hyperparameters['n_components'])
        
        elif modelName == 'agglo':
            if nonreq:
                compute_distances=False
                if hyperparameters['distance_threshold']=='true':
                    compute_distances=True

                compute_full_tree = 'auto'
                if hyperparameters['compute_full_tree']=='true':
                    compute_full_tree=True
                elif hyperparameters['compute_full_tree']=='false':
                    compute_full_tree=False


                if hyperparameters['connectivity']=='':
                    connectivity=None
                else:
                    connectivity=hyperparameters['connectivity']

                if hyperparameters['memory']=='':
                    memory=None
                else:
                    memory=hyperparameters['memory']

                if hyperparameters['n_clusters']=='':
                    n_clusters=None
                else:
                    n_clusters=hyperparameters['n_clusters']

                if hyperparameters['distance_threshold']=='':
                    distance_threshold=None
                else:
                    distance_threshold=hyperparameters['distance_threshold']

                train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid = train_agglomerative(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize, k_min=2, k_max=8,
                                n_clusters=n_clusters,
                                metric=hyperparameters['metric'],
                                memory=memory,
                                connectivity=connectivity,
                                compute_full_tree=compute_full_tree,
                                linkage=hyperparameters['linkage'],
                                distance_threshold=distance_threshold,
                                compute_distances=compute_distances,)
                
                
            else:   
                if hyperparameters['n_clusters']=='':
                    n_clusters=None
                else:
                    n_clusters=hyperparameters['n_clusters']

                train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid = train_agglomerative(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize, k_min=2, k_max=8,
                                n_clusters=n_clusters)

        elif modelName == 'dbscan':
            train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid = train_dbscan(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize)

        elif modelName == 'birch':
            train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid = train_birch(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize)

        elif modelName == 'spectral':
            train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid = train_spectral(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize)

        # Additional Clustering Models
        elif modelName == 'affinity_propagation':
            train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid = train_affinity_propagation(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize)

        elif modelName == 'bisecting_kmeans':
            train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid = train_bisecting_kmeans(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize, k_min=2, k_max=8)

        elif modelName == 'hdbscan':
            train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid = train_hdbscan(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize)

        elif modelName == 'meanshift':
            train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid = train_meanshift(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize)

        elif modelName == 'minibatch_kmeans':
            train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid = train_minibatch_kmeans(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize, k_min=2, k_max=8)

        elif modelName == 'optics':
            train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid = train_optics(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize)

        else:
            raise ValueError('invalid model architecture specified')

        fmt = f".{sigfig}f" 

    #results go in result dictionary and get written to excel file
    
        # Initialize feature selection and outlier info only if not already set by regression pipeline
        # (Regression models set these values, classification/clustering models don't)
        if 'feature_selection_info' not in locals():
            feature_selection_info = None
        if 'outlier_info' not in locals():
            outlier_info = None
        
        cv_file = None
        # Use normalized check for cross validation
        cv_summary_data = None
        if cv_enabled:
            # Update cross-validation status to running
            tracker.update_stage('cross_validation', 'running', 10, 
                               f'Running {cross_validation_type} with {cross_validation_folds} folds...')
            if modelName.endswith('_classifier'):
                cv_result = run_cross_validation(
                    df=df,
                    indicator_names=indicator_names,
                    predictor_names=predictor_names,
                    model=storedModel,
                    scaler=scaler,
                    cv_type=cross_validation_type,
                    cv_folds=cross_validation_folds,
                    useTransformer=useTransformer,
                    transformer_cols=transformer_names,
                    seed=seed,
                    problem_type="classification",
                )
                if cv_result and isinstance(cv_result, tuple) and cv_result[0] is not None:
                    cv_file_path, cv_summary_df = cv_result
                    cv_file = cv_file_path.name
                    # Convert summary to dict for JSON serialization
                    cv_summary_data = cv_summary_df.to_dict('records')
                    tracker.update_stage('cross_validation', 'completed', 100, 
                                       f'Cross-validation complete ({cross_validation_type}, {cross_validation_folds} folds)')
                elif cv_result and not isinstance(cv_result, tuple):
                    # Backward compatibility - old return format
                    cv_file = cv_result.name
                    tracker.update_stage('cross_validation', 'completed', 100, 
                                       f'Cross-validation complete ({cross_validation_type}, {cross_validation_folds} folds)')
            elif modelName not in ['kmeans', 'gmm', 'agglo', 'dbscan', 'birch', 'spectral', 'affinity_propagation', 'bisecting_kmeans', 'hdbscan', 'meanshift', 'minibatch_kmeans', 'optics']:
                cv_result = run_cross_validation(
                    df=df,
                    indicator_names=indicator_names,
                    predictor_names=predictor_names,
                    model=storedModel,
                    scaler=scaler,
                    cv_type=cross_validation_type,
                    cv_folds=cross_validation_folds,
                    useTransformer=useTransformer,
                    transformer_cols=transformer_names,
                    seed=seed,
                    problem_type="regression",
                    y_scaler_type=scaler,
                )
                if cv_result and isinstance(cv_result, tuple) and cv_result[0] is not None:
                    cv_file_path, cv_summary_df = cv_result
                    cv_file = cv_file_path.name
                    # Convert summary to dict for JSON serialization
                    cv_summary_data = cv_summary_df.to_dict('records')
                    tracker.update_stage('cross_validation', 'completed', 100, 
                                       f'Cross-validation complete ({cross_validation_type}, {cross_validation_folds} folds)')
                elif cv_result and not isinstance(cv_result, tuple):
                    # Backward compatibility - old return format
                    cv_file = cv_result.name
                    tracker.update_stage('cross_validation', 'completed', 100, 
                                       f'Cross-validation complete ({cross_validation_type}, {cross_validation_folds} folds)')

    ## Classification results
        if modelName.endswith('_classifier'): 
            
            # Initialize additional_metrics - will be set by models that return it via unpack_classification_result
            if 'additional_metrics' not in locals():
                additional_metrics = None
            
            # Convert to list if needed (handle both lists and numpy arrays)
            indicator_list = indicator_names.tolist() if hasattr(indicator_names, 'tolist') else list(indicator_names)
            predictor_list = predictor_names.tolist() if hasattr(predictor_names, 'tolist') else list(predictor_names)
            
            # {'Adelie Penguin (Pygoscelis adeliae)': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 34.0}, 
            #  'Chinstrap penguin (Pygoscelis antarctica)': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 11.0}, 
            #  'Gentoo penguin (Pygoscelis papua)': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 22.0}, 
            #  'accuracy': 1.0, 'macro avg': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 67.0}, 
            #  'weighted avg': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 67.0}}

            result = {
                'accuracy': format(round(report['accuracy'], sigfig), fmt),
                'precision': format(round(report['weighted avg']['precision'], sigfig), fmt),
                'recall': format(round(report['weighted avg']['recall'], sigfig), fmt),
                'f1score': format(round(report['weighted avg']['f1-score'], sigfig), fmt),
                'support': format(round(report['weighted avg']['support'], sigfig), fmt),
                'macro_precision': format(round(report['macro avg']['precision'], sigfig), fmt),
                'macro_recall': format(round(report['macro avg']['recall'], sigfig), fmt),
                'macro_f1score': format(round(report['macro avg']['f1-score'], sigfig), fmt),
                'macro_support': format(round(report['macro avg']['support'], sigfig), fmt),
                'indicators': [str(i) for i in indicator_list],
                'predictors': [str(p) for p in predictor_list],
                'cross_validation_file': cv_file,
                'model_params': _json_safe_params(params),  # Sanitize: get_params() can include estimators
                'feature_selection_info': feature_selection_info,
                'outlier_info': outlier_info,
            }

            #write to excel for classifier with all comprehensive metrics
            write_to_excelClassifier(data, indicator_names, predictor_names, stratify_name, scaler, seed, modelName, params, units, report, cm, 
                                   additional_metrics=additional_metrics)

    ## Cluster results
        elif modelName in ['kmeans', 'gmm', 'agglo', 'dbscan', 'birch', 'spectral', 'affinity_propagation', 'bisecting_kmeans', 'hdbscan', 'meanshift', 'minibatch_kmeans', 'optics']:
            result = {
                'train_silhouette' : format(round(train_results['silhouette'],sigfig), fmt),
                'train_calinski_harabasz': format(round(train_results['calinski_harabasz'],sigfig), fmt),
                'train_davies_bouldin' : format(round(train_results['davies_bouldin'],sigfig), fmt),
                'test_silhouette' : format(round(test_results['silhouette'],sigfig), fmt),
                'test_calinski_harabasz': format(round(test_results['calinski_harabasz'],sigfig), fmt),
                'test_davies_bouldin' : format(round(test_results['davies_bouldin'],sigfig), fmt),
                'best_k': best_k,
                'model_params': _json_safe_params(params),  # Sanitize: get_params() can include estimators
            }
            #write to excel for cluster
            write_to_excelCluster(data, indicator_names, stratify_name, scaler, seed, modelName, params, units, train_results['silhouette'], train_results['calinski_harabasz'], train_results['davies_bouldin'], test_results['silhouette'], test_results['calinski_harabasz'], test_results['davies_bouldin'], best_k, centers, silhouette_grid)

    ## Regression results
        else: 
            trainOverall = train_results.iloc[-1]
            testOverall = test_results.iloc[-1]

            #std of RMSE = np.std(all the target rmse) and np.std(all the test rmse)
            train_rmse_values = train_results['RMSE'][:-1]
            test_rmse_values = test_results['RMSE'][:-1]
            train_mae_values = train_results['MAE'][:-1]
            test_mae_values = test_results['MAE'][:-1]

            train_rmse_std = np.std(train_rmse_values) if len(train_rmse_values) > 1 else None
            train_mae_std = np.std(train_mae_values) if len(train_mae_values) > 1 else None
            test_rmse_std = np.std(test_rmse_values) if len(test_rmse_values) > 1 else None
            test_mae_std = np.std(test_mae_values) if len(test_mae_values) > 1 else None

            # Extract sample counts from shapes dictionary (shapes contains tuples like (n_samples, n_features))
            train_n = shapes.get('X_train', (0,))[0] if isinstance(shapes, dict) and 'X_train' in shapes else 0
            test_n = shapes.get('X_test', (0,))[0] if isinstance(shapes, dict) and 'X_test' in shapes else 0
            
            result = {
                'trainscore': format(round(trainOverall['R²'],sigfig), fmt),
                'valscore': format(round(testOverall['R²'],sigfig), fmt),
                'trainrmse': format(round(trainOverall['RMSE'],sigfig), fmt),
                'trainrmsestd': format(round(train_rmse_std, sigfig), fmt) if train_rmse_std is not None else 'N/A',
                'trainmae': format(round(trainOverall['MAE'],sigfig), fmt),
                'trainmaestd': format(round(train_mae_std, sigfig), fmt) if train_mae_std is not None else 'N/A',
                'valrmse': format(round(testOverall['RMSE'],sigfig), fmt),
                'valrmsestd': format(round(test_rmse_std, sigfig), fmt) if test_rmse_std is not None else 'N/A',
                'valmae': format(round(testOverall['MAE'],sigfig), fmt),
                'valmaestd': format(round(test_mae_std, sigfig), fmt) if test_mae_std is not None else 'N/A',
                'train_n': int(train_n) if train_n else 0,
                'test_n': int(test_n) if test_n else 0,
                'indicators': [str(i) for i in (indicator_names.tolist() if hasattr(indicator_names, 'tolist') else list(indicator_names))],
                'predictors': [str(p) for p in (predictor_names.tolist() if hasattr(predictor_names, 'tolist') else list(predictor_names))],
                'cross_validation_file': cv_file,
                'cross_validation_summary': cv_summary_data,  # Add CV summary data
                'model_params': _json_safe_params(params),  # Sanitize: get_params() can include estimators
                'feature_selection_info': feature_selection_info,  # Feature selection details (None if not set by regression pipeline)
                'outlier_info': outlier_info,  # Outlier handling details (None if not set by regression pipeline)
            }
            regression_visuals = []
            
            # Check for baseline graphics (no advanced options)
            vis_dir = _config.VIS_DIR
            baseline_target_plot_exists = (vis_dir / "target_plot_1.png").exists()
            baseline_pred_actual_exists = (vis_dir / "target_plot_pred_actual_1.png").exists()
            baseline_residuals_exists = (vis_dir / "target_plot_residuals_1.png").exists()
            baseline_shap_exists = (vis_dir / "shap_summary.png").exists()
            
            # Check for advanced graphics (with advanced options)
            advanced_target_plot_exists = (vis_dir / "target_plot_1_advanced.png").exists()
            advanced_pred_actual_exists = (vis_dir / "target_plot_pred_actual_1_advanced.png").exists()
            advanced_residuals_exists = (vis_dir / "target_plot_residuals_1_advanced.png").exists()
            advanced_shap_exists = (vis_dir / "shap_summary_advanced.png").exists()
            
            # Determine mode label
            mode_label = 'DiGiTerra Simple Modeling' if modeling_mode == 'simple' else (
                'DiGiTerra Advanced Modeling' if modeling_mode == 'advanced' else 'DiGiTerra AutoML'
            )
            
            # Add separate Predicted vs Actual and Test Residuals first (defaults for the two panels)
            if baseline_pred_actual_exists:
                regression_visuals.append({'label': f'Predicted vs Actual (per target) - {mode_label}', 'file': 'target_plot_pred_actual', 'type': 'baseline'})
            if advanced_pred_actual_exists:
                regression_visuals.append({'label': f'Predicted vs Actual (per target) - {mode_label}', 'file': 'target_plot_pred_actual_advanced', 'type': 'advanced'})
            if baseline_residuals_exists:
                regression_visuals.append({'label': f'Test Residuals (per target) - {mode_label}', 'file': 'target_plot_residuals', 'type': 'baseline'})
            if advanced_residuals_exists:
                regression_visuals.append({'label': f'Test Residuals (per target) - {mode_label}', 'file': 'target_plot_residuals_advanced', 'type': 'advanced'})
            # Combined view (optional)
            if baseline_target_plot_exists:
                regression_visuals.append({'label': f'Predicted vs Actual + Residuals (combined, per target) - {mode_label}', 'file': 'target_plot', 'type': 'baseline'})
            if advanced_target_plot_exists:
                regression_visuals.append({'label': f'Predicted vs Actual + Residuals (combined, per target) - {mode_label}', 'file': 'target_plot_advanced', 'type': 'advanced'})
            if not baseline_target_plot_exists and not advanced_target_plot_exists:
                regression_visuals.append({'label': 'Predicted vs Actual + Residuals (per target)', 'file': 'target_plot', 'type': 'default'})
            
            regression_candidates = [
                ('regression_predicted_vs_actual.png', 'Predicted vs Actual (summary)'),
                ('regression_residuals_hist.png', 'Residuals Histogram'),
                ('regression_residuals_vs_fitted.png', 'Residuals vs Fitted'),
                ('regression_density.png', 'Actual vs Predicted Density'),
                ('regression_feature_importance.png', 'Feature Importance'),
                ('regression_permutation_importance.png', 'Permutation Importance'),
            ]
            
            # Add baseline SHAP if exists
            if baseline_shap_exists:
                regression_visuals.append({'label': f'SHAP Summary - {mode_label}', 'file': 'shap_summary', 'type': 'baseline'})
            
            # Add advanced SHAP if exists
            if advanced_shap_exists:
                regression_visuals.append({'label': f'SHAP Summary - {mode_label}', 'file': 'shap_summary_advanced', 'type': 'advanced'})
            
            # Add other regression candidates (these don't have baseline/advanced variants)
            for filename, label in regression_candidates:
                if (vis_dir / filename).exists():
                    regression_visuals.append({'label': label, 'file': filename, 'type': 'default'})
            
            result['regression_visuals'] = regression_visuals
            try:
                write_to_excelRegression(data, indicator_names, predictor_names, stratify_name, modelName, params, units, trainOverall, testOverall, train_results, test_results, scaler, seed, shapes, quantileBin_results, cross_validation_summary=cv_summary_data, feature_selection_info=feature_selection_info, outlier_info=outlier_info)
            except Exception as excel_err:
                logger.warning("Excel export failed (model results unchanged): %s", excel_err)

            # Guard against swapped scalers (common cause of wildly wrong predictions)
            try:
                y_train_array = df[predictor_names].to_numpy()
                y_scaler, X_scaler = _maybe_fix_swapped_scalers(y_scaler, X_scaler, y_train_array)
            except Exception:
                pass

            try:
                _y_train = df[predictor_names].to_numpy()
                store['y_train_mean'] = float(np.mean(_y_train))
                store['y_train_std'] = float(np.std(_y_train))
            except Exception:
                store['y_train_mean'] = None
                store['y_train_std'] = None

            store['y_scaler'] = y_scaler


        store['model'] = storedModel #10/1 Rowan this is how the model is stored, it is returned from the train functions
        store['X_scaler'] = X_scaler
        store['feature_order'] = feature_order
        # Store predictor names for use in prediction (handles multi-target regression)
        store['predictor_names'] = predictor_names
        # Store model type and training visualization for inference results page
        if modelName.endswith('_classifier'):
            store['model_type'] = 'classification'
            store['training_visualization'] = 'confusion_matrix.png'
            store['training_visualization_version'] = str(int(time.time() * 1000))
        elif modelName in ['kmeans', 'gmm', 'agglo', 'dbscan', 'birch', 'spectral', 'affinity_propagation', 'bisecting_kmeans', 'hdbscan', 'meanshift', 'minibatch_kmeans', 'optics']:
            store['model_type'] = 'cluster'
            store['training_visualization'] = 'cluster_pca_train.png'
            store['training_visualization_version'] = str(int(time.time() * 1000))
        else:
            store['model_type'] = 'regression'
            rv = result.get('regression_visuals') or []
            inference_filename = shapes.get('training_visualization_filename') if isinstance(shapes, dict) else None
            if inference_filename:
                store['training_visualization'] = inference_filename
            else:
                use_advanced = modeling_mode == 'advanced' and any(v.get('type') == 'advanced' for v in rv)
                store['training_visualization'] = 'target_plot_1_advanced.png' if use_advanced else 'target_plot_1.png'
            store['training_visualization_version'] = str(int(time.time() * 1000))

        # Mark all stages complete
        tracker.complete()
        result['session_id'] = session_id  # Include session ID in response
        
        # Store result for SSE endpoint
        set_result(session_id, result)
        
    except Exception as e:
        # Ensure tracker is cleaned up on error
        logger.error(f"Error in model training: {e}", exc_info=True)
        error_result = {'error': str(e), 'session_id': session_id}
        set_result(session_id, error_result)
        tracker.update_stage('model_training', 'completed', 0, f'Error: {str(e)}')
        remove_tracker(session_id)

