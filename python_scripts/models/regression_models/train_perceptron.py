from sklearn.linear_model import Perceptron
from python_scripts.preprocessing.run_regression_pipeline import run_regression

def train_perceptron(model_type, train_data, target_variables, use_stratified_split,
                X, y, stratifyColumn, units,
                X_scaler_type, y_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols, testSize,
                feature_selection_method='none', feature_selection_k=None,
                outlier_method='none', outlier_action='remove',
                hyperparameter_search='none', search_cv_folds=5, search_n_iter=50,
                progress_tracker=None, modeling_mode='simple', **kwargs):

    model = Perceptron(penalty=kwargs.get('penalty', None),
                      alpha=kwargs.get('alpha', 0.0001),
                      fit_intercept=kwargs.get('fit_intercept', True),
                      max_iter=kwargs.get('max_iter', 1000),
                      tol=kwargs.get('tol', 1e-3),
                      shuffle=kwargs.get('shuffle', True),
                      verbose=kwargs.get('verbose', 0),
                      eta0=kwargs.get('eta0', 1.0),
                      n_jobs=kwargs.get('n_jobs', None),
                      random_state=kwargs.get('random_state', seed),
                      early_stopping=kwargs.get('early_stopping', False),
                      validation_fraction=kwargs.get('validation_fraction', 0.1),
                      n_iter_no_change=kwargs.get('n_iter_no_change', 5),
                      class_weight=kwargs.get('class_weight', None),
                      warm_start=kwargs.get('warm_start', False))

    return run_regression(
        model, "Perceptron",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, X_scaler_type, y_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        progress_tracker=progress_tracker, modeling_mode=modeling_mode
    )
