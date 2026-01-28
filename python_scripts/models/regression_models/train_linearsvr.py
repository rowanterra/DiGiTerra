from sklearn.svm import LinearSVR
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
from python_scripts.preprocessing.run_regression_pipeline import run_regression

def train_linearsvr(model_type, train_data, target_variables, use_stratified_split,
                X, y, stratifyColumn, units,
                X_scaler_type, y_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols, testSize,
                feature_selection_method='none', feature_selection_k=None,
                outlier_method='none', outlier_action='remove',
                hyperparameter_search='none', search_cv_folds=5, search_n_iter=50,
                progress_tracker=None, **kwargs):

    base_model = LinearSVR(epsilon=kwargs.get('epsilon', 0.0),
                          tol=kwargs.get('tol', 1e-4),
                          C=kwargs.get('C', 1.0),
                          loss=kwargs.get('loss', 'epsilon_insensitive'),
                          fit_intercept=kwargs.get('fit_intercept', True),
                          intercept_scaling=kwargs.get('intercept_scaling', 1.0),
                          dual=kwargs.get('dual', True),
                          verbose=kwargs.get('verbose', 0),
                          random_state=kwargs.get('random_state', seed),
                          max_iter=kwargs.get('max_iter', 1000))
    
    # Use MultiOutputRegressor only for multi-target regression
    if np.asarray(target_variables).size > 1:
        model = MultiOutputRegressor(base_model)
    else:
        model = base_model

    return run_regression(
        model, "LinearSVR",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, X_scaler_type, y_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        progress_tracker=progress_tracker, modeling_mode=kwargs.get('modeling_mode', 'simple')
    )
