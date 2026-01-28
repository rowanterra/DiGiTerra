from sklearn.svm import NuSVR
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
from python_scripts.preprocessing.run_regression_pipeline import run_regression

def train_nusvr(model_type, train_data, target_variables, use_stratified_split,
                X, y, stratifyColumn, units,
                X_scaler_type, y_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols, testSize,
                feature_selection_method='none', feature_selection_k=None,
                outlier_method='none', outlier_action='remove',
                hyperparameter_search='none', search_cv_folds=5, search_n_iter=50,
                progress_tracker=None, **kwargs):

    base_model = NuSVR(nu=kwargs.get('nu', 0.5),
                      C=kwargs.get('C', 1.0),
                      kernel=kwargs.get('kernel', 'rbf'),
                      degree=kwargs.get('degree', 3),
                      gamma=kwargs.get('gamma', 'scale'),
                      coef0=kwargs.get('coef0', 0.0),
                      shrinking=kwargs.get('shrinking', True),
                      tol=kwargs.get('tol', 1e-3),
                      cache_size=kwargs.get('cache_size', 200),
                      verbose=kwargs.get('verbose', False),
                      max_iter=kwargs.get('max_iter', -1))
    
    # Use MultiOutputRegressor only for multi-target regression
    if np.asarray(target_variables).size > 1:
        model = MultiOutputRegressor(base_model)
    else:
        model = base_model

    return run_regression(
        model, "NuSVR",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, X_scaler_type, y_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        progress_tracker=progress_tracker
    )
