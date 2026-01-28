from sklearn.linear_model import ElasticNetCV
from python_scripts.preprocessing.run_regression_pipeline import run_regression

def train_elasticnet_cv(model_type, train_data, target_variables, use_stratified_split,
                X, y, stratifyColumn, units,
                X_scaler_type, y_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols, testSize,
                feature_selection_method='none', feature_selection_k=None,
                outlier_method='none', outlier_action='remove',
                hyperparameter_search='none', search_cv_folds=5, search_n_iter=50,
                progress_tracker=None, **kwargs):

    model = ElasticNetCV(l1_ratio=kwargs.get('l1_ratio', 0.5),
                        eps=kwargs.get('eps', 1e-3),
                        n_alphas=kwargs.get('n_alphas', 100),
                        alphas=kwargs.get('alphas', None),
                        fit_intercept=kwargs.get('fit_intercept', True),
                        precompute=kwargs.get('precompute', 'auto'),
                        max_iter=kwargs.get('max_iter', 1000),
                        tol=kwargs.get('tol', 1e-4),
                        cv=kwargs.get('cv', None),
                        copy_X=kwargs.get('copy_X', True),
                        verbose=kwargs.get('verbose', 0),
                        n_jobs=kwargs.get('n_jobs', None),
                        positive=kwargs.get('positive', False),
                        random_state=kwargs.get('random_state', seed),
                        selection=kwargs.get('selection', 'cyclic'),
                        )

    return run_regression(
        model, "ElasticNetCV",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, X_scaler_type, y_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        progress_tracker=progress_tracker
    )
