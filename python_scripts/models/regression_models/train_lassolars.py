from sklearn.linear_model import LassoLars
import numpy as np
from python_scripts.preprocessing.run_regression_pipeline import run_regression

def train_lassolars(model_type, train_data, target_variables, use_stratified_split,
                X, y, stratifyColumn, units,
                X_scaler_type, y_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols, testSize,
                feature_selection_method='none', feature_selection_k=None,
                outlier_method='none', outlier_action='remove',
                hyperparameter_search='none', search_cv_folds=5, search_n_iter=50,
                progress_tracker=None, **kwargs):

    model = LassoLars(alpha=kwargs.get('alpha', 1.0),
                     fit_intercept=kwargs.get('fit_intercept', True),
                     verbose=kwargs.get('verbose', False),
                     precompute=kwargs.get('precompute', 'auto'),
                     max_iter=kwargs.get('max_iter', 500),
                     eps=kwargs.get('eps', np.finfo(float).eps),
                     copy_X=kwargs.get('copy_X', True),
                     fit_path=kwargs.get('fit_path', True),
                     positive=kwargs.get('positive', False),
                     jitter=kwargs.get('jitter', None),
                     random_state=kwargs.get('random_state', seed),
                     )

    return run_regression(
        model, "LassoLars",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, X_scaler_type, y_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        progress_tracker=progress_tracker, modeling_mode=kwargs.get('modeling_mode', 'simple')
    )
