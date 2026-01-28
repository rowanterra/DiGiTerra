from sklearn.linear_model import LinearRegression
from python_scripts.preprocessing.run_regression_pipeline import run_regression



def train_linear(model_type, train_data, target_variables, use_stratified_split,
                X, y, stratifyColumn, units,
                X_scaler_type, y_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols, testSize,
                feature_selection_method='none', feature_selection_k=None,
                outlier_method='none', outlier_action='remove',
                hyperparameter_search='none', search_cv_folds=5, search_n_iter=50,
                progress_tracker=None, modeling_mode='simple', **kwargs):

    model = LinearRegression(fit_intercept=kwargs.get('fit_intercept', True), 
                            copy_X=kwargs.get('copy_X', True), 
                            n_jobs=kwargs.get('n_jobs', None),
                            positive=kwargs.get('positive', False))

    return run_regression(
        model, "LinearRegression",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, X_scaler_type, y_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        progress_tracker=progress_tracker, modeling_mode=modeling_mode
    )
