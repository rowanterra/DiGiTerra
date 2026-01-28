from sklearn.linear_model import RANSACRegressor, LinearRegression
import numpy as np
from python_scripts.preprocessing.run_regression_pipeline import run_regression

def train_ransac_regressor(model_type, train_data, target_variables, use_stratified_split,
                X, y, stratifyColumn, units,
                X_scaler_type, y_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols, testSize,
                feature_selection_method='none', feature_selection_k=None,
                outlier_method='none', outlier_action='remove',
                hyperparameter_search='none', search_cv_folds=5, search_n_iter=50,
                progress_tracker=None, **kwargs):
    # RANSACRegressor requires an estimator - use LinearRegression as default if none provided
    estimator = kwargs.get('estimator', None)
    if estimator is None:
        estimator = LinearRegression()

    model = RANSACRegressor(estimator=estimator,
                           min_samples=kwargs.get('min_samples', None),
                           residual_threshold=kwargs.get('residual_threshold', None),
                           is_data_valid=kwargs.get('is_data_valid', None),
                           is_model_valid=kwargs.get('is_model_valid', None),
                           max_trials=kwargs.get('max_trials', 100),
                           max_skips=kwargs.get('max_skips', np.inf),
                           stop_n_inliers=kwargs.get('stop_n_inliers', np.inf),
                           stop_score=kwargs.get('stop_score', np.inf),
                           stop_probability=kwargs.get('stop_probability', 0.99),
                           loss=kwargs.get('loss', 'absolute_error'),
                           random_state=kwargs.get('random_state', seed),
                           )

    return run_regression(
        model, "RANSACRegressor",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, X_scaler_type, y_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        progress_tracker=progress_tracker
    )
