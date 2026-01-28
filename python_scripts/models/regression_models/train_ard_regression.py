from sklearn.linear_model import ARDRegression

from python_scripts.preprocessing.run_regression_pipeline import run_regression


def train_ard_regression(model_type, train_data, target_variables, use_stratified_split,
                         X, y, stratifyColumn, units,
                         X_scaler_type, y_scaler_type,
                         seed, quantileBinDict, sigfig, useTransformer, categorical_cols, testSize,
                         feature_selection_method='none', feature_selection_k=None,
                         outlier_method='none', outlier_action='remove',
                         hyperparameter_search='none', search_cv_folds=5, search_n_iter=50,
                         progress_tracker=None, **kwargs):
    model = ARDRegression(max_iter=kwargs.get('max_iter', kwargs.get('n_iter', 300)),
                         tol=kwargs.get('tol', 1e-3),
                         alpha_1=kwargs.get('alpha_1', 1e-6),
                         alpha_2=kwargs.get('alpha_2', 1e-6),
                         lambda_1=kwargs.get('lambda_1', 1e-6),
                         lambda_2=kwargs.get('lambda_2', 1e-6),
                         compute_score=kwargs.get('compute_score', False),
                         threshold_lambda=kwargs.get('threshold_lambda', 10000.0),
                         fit_intercept=kwargs.get('fit_intercept', True),
                         copy_X=kwargs.get('copy_X', True),
                         verbose=kwargs.get('verbose', False))

    return run_regression(
        model, "ARDRegression",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, X_scaler_type, y_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        progress_tracker=progress_tracker, modeling_mode=kwargs.get('modeling_mode', 'simple')
    )
