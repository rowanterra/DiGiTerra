from sklearn.linear_model import LogisticRegression
from python_scripts.preprocessing.run_regression_pipeline import run_regression

def train_logistic(model_type, train_data, target_variables, use_stratified_split,
                X, y, stratifyColumn, units,
                X_scaler_type, y_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols, testSize,
                feature_selection_method='none', feature_selection_k=None,
                outlier_method='none', outlier_action='remove',
                hyperparameter_search='none', search_cv_folds=5, search_n_iter=50,
                progress_tracker=None, modeling_mode='simple', **kwargs):

    model = LogisticRegression(penalty=kwargs.get('penalty', 'l2'),
                              dual=kwargs.get('dual', False),
                              tol=kwargs.get('tol', 1e-4),
                              C=kwargs.get('C', 1.0),
                              fit_intercept=kwargs.get('fit_intercept', True),
                              intercept_scaling=kwargs.get('intercept_scaling', 1),
                              class_weight=kwargs.get('class_weight', None),
                              random_state=kwargs.get('random_state', seed),
                              solver=kwargs.get('solver', 'lbfgs'),
                              max_iter=kwargs.get('max_iter', 100),
                              multi_class=kwargs.get('multi_class', 'auto'),
                              verbose=kwargs.get('verbose', 0),
                              warm_start=kwargs.get('warm_start', False),
                              n_jobs=kwargs.get('n_jobs', None),
                              l1_ratio=kwargs.get('l1_ratio', None))

    return run_regression(
        model, "LogisticRegression",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, X_scaler_type, y_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        progress_tracker=progress_tracker, modeling_mode=modeling_mode
    )
