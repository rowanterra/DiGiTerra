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
from python_scripts.model_registry import MODEL_REGISTRY, get_model_kwargs, CLUSTER_MODELS

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
            # Only forbid stratify-by-target for regression/classification (not clustering; clustering has no supervised target)
            cluster_models = {'kmeans', 'gmm', 'agglo', 'dbscan', 'birch', 'spectral', 'affinity_propagation', 'bisecting_kmeans', 'hdbscan', 'meanshift', 'minibatch_kmeans', 'optics'}
            if modelName not in cluster_models and stratify_name and predictor_names and stratify_name in predictor_names:
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

        # Registry-based dispatch
        if modelName == 'TerraFORMER' or modelName not in MODEL_REGISTRY or MODEL_REGISTRY[modelName][0] is None:
            raise ValueError('invalid model architecture specified')
        else:
            train_fn, problem_type = MODEL_REGISTRY[modelName]
            X_data = _safe_select_columns(df, indicator_names)
            y_data = _safe_select_columns(df, predictor_names)
            base = dict(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=X_data, y=y_data, stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)
            if problem_type == 'regression':
                base['y_scaler_type'] = scaler
            kwargs = get_model_kwargs(modelName, hyperparameters, nonreq, seed)
            if problem_type == 'cluster':
                base_c = dict(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize, k_min=2, k_max=8)
                result = train_fn(**base_c, **kwargs)
            elif problem_type == 'regression':
                result = train_fn(modelName, **base, **kwargs)
            else:
                result = train_fn(**base, **kwargs)
            if problem_type == 'classification':
                report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics, feature_selection_info, outlier_info = unpack_classification_result(result)
            elif problem_type == 'cluster':
                train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid = result
            else:
                train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = result

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
            if problem_type == 'classification':
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
            elif modelName not in CLUSTER_MODELS:
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
        if problem_type == 'classification': 
            
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
                                       additional_metrics=additional_metrics, outlier_info=outlier_info)

    ## Cluster results
        elif modelName in CLUSTER_MODELS:
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
        if problem_type == 'classification':
            store['model_type'] = 'classification'
            store['training_visualization'] = 'confusion_matrix.png'
            store['training_visualization_version'] = str(int(time.time() * 1000))
        elif modelName in CLUSTER_MODELS:
            store['model_type'] = 'cluster'
            store['training_visualization'] = 'cluster_pca_train.png'
            store['training_visualization_version'] = str(int(time.time() * 1000))
        else:
            store['model_type'] = 'regression'
            rv = result.get('regression_visuals') or []
            use_advanced = modeling_mode == 'advanced' and any(v.get('type') == 'advanced' for v in rv)
            pred_actual_name = 'target_plot_pred_actual_1_advanced.png' if use_advanced else 'target_plot_pred_actual_1.png'
            if (vis_dir / pred_actual_name).exists():
                store['training_visualization'] = pred_actual_name
            else:
                inference_filename = shapes.get('training_visualization_filename') if isinstance(shapes, dict) else None
                if inference_filename:
                    store['training_visualization'] = inference_filename
                else:
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

