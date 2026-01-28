from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
from python_scripts.preprocessing.run_regression_pipeline import run_regression

def train_gb(model_type, train_data, target_variables, use_stratified_split,
                X, y, stratifyColumn, units,
                X_scaler_type, y_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols,testSize,
                feature_selection_method='none', feature_selection_k=None,
                outlier_method='none', outlier_action='remove',
                hyperparameter_search='none', search_cv_folds=5, search_n_iter=50,
                progress_tracker=None, modeling_mode='simple', **kwargs):

    base_model = GradientBoostingRegressor(n_estimators=kwargs.get('n_estimators', 100), 
                                           learning_rate=kwargs.get('learning_rate', 0.1), 
                                           max_depth=kwargs.get('max_depth', 3), 
                                           loss=kwargs.get('loss', 'absolute_error'),
                                           subsample=kwargs.get('subsample', 1),
                                           criterion=kwargs.get('criterion', 'friedman_mse'),
                                           min_samples_split=kwargs.get('min_samples_split', 2),
                                           min_samples_leaf=kwargs.get('min_samples_leaf', 1),
                                           min_weight_fraction_leaf=kwargs.get('min_weight_fraction_leaf', 0),
                                           min_impurity_decrease=kwargs.get('min_impurity_decrease', 0),
                                           init=kwargs.get('init', None),
                                           random_state=kwargs.get('random_state', None),
                                           max_features=kwargs.get('max_features', None),
                                           alpha=kwargs.get('alpha', .9),
                                           verbose=kwargs.get('verbose', 0),
                                           max_leaf_nodes=kwargs.get('max_leaf_nodes', None),
                                           warm_start=kwargs.get('warm_start', False))
    
    # Use MultiOutputRegressor only for multi-target regression
    if np.asarray(target_variables).size > 1:
        model = MultiOutputRegressor(base_model)
    else:
        model = base_model

    return run_regression(
        model, "GradientBoosting",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, X_scaler_type, y_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols,testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        progress_tracker=progress_tracker, modeling_mode=modeling_mode
    )
