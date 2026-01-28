from sklearn.linear_model import RidgeCV
from python_scripts.preprocessing.run_regression_pipeline import run_regression

def train_ridge_cv(model_type, train_data, target_variables, use_stratified_split,
                X, y, stratifyColumn, units,
                X_scaler_type, y_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols, testSize,
                feature_selection_method='none', feature_selection_k=None,
                outlier_method='none', outlier_action='remove',
                hyperparameter_search='none', search_cv_folds=5, search_n_iter=50,
                progress_tracker=None, **kwargs):

    model = RidgeCV(alphas=kwargs.get('alphas', (0.1, 1.0, 10.0)),
                   fit_intercept=kwargs.get('fit_intercept', True),
                   scoring=kwargs.get('scoring', None),
                   cv=kwargs.get('cv', None),
                   gcv_mode=kwargs.get('gcv_mode', 'auto'),
                   store_cv_results=kwargs.get('store_cv_results', False),
                   alpha_per_target=kwargs.get('alpha_per_target', False),
                   )

    return run_regression(
        model, "RidgeCV",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, X_scaler_type, y_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        progress_tracker=progress_tracker
    )
