from sklearn.linear_model import Lars
import numpy as np
from python_scripts.preprocessing.run_regression_pipeline import run_regression

def train_lars(model_type, train_data, target_variables, use_stratified_split,
                X, y, stratifyColumn, units,
                X_scaler_type, y_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols, testSize,
                feature_selection_method='none', feature_selection_k=None,
                outlier_method='none', outlier_action='remove',
                hyperparameter_search='none', search_cv_folds=5, search_n_iter=50,
                progress_tracker=None, **kwargs):

    model = Lars(fit_intercept=kwargs.get('fit_intercept', True),
                verbose=kwargs.get('verbose', False),
                precompute=kwargs.get('precompute', 'auto'),
                n_nonzero_coefs=kwargs.get('n_nonzero_coefs', 500),
                eps=kwargs.get('eps', np.finfo(float).eps),
                copy_X=kwargs.get('copy_X', True),
                fit_path=kwargs.get('fit_path', True),
                jitter=kwargs.get('jitter', None),
                random_state=kwargs.get('random_state', seed),
                )

    return run_regression(
        model, "Lars",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, X_scaler_type, y_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        progress_tracker=progress_tracker
    )
