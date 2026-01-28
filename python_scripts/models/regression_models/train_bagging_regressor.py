from sklearn.ensemble import BaggingRegressor
from python_scripts.preprocessing.run_regression_pipeline import run_regression

def train_bagging_regressor(model_type, train_data, target_variables, use_stratified_split,
                X, y, stratifyColumn, units,
                X_scaler_type, y_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols, testSize,
                feature_selection_method='none', feature_selection_k=None,
                outlier_method='none', outlier_action='remove',
                hyperparameter_search='none', search_cv_folds=5, search_n_iter=50,
                progress_tracker=None, **kwargs):

    model = BaggingRegressor(estimator=kwargs.get('estimator', None),
                            n_estimators=kwargs.get('n_estimators', 10),
                            max_samples=kwargs.get('max_samples', 1.0),
                            max_features=kwargs.get('max_features', 1.0),
                            bootstrap=kwargs.get('bootstrap', True),
                            bootstrap_features=kwargs.get('bootstrap_features', False),
                            oob_score=kwargs.get('oob_score', False),
                            warm_start=kwargs.get('warm_start', False),
                            n_jobs=kwargs.get('n_jobs', None),
                            random_state=kwargs.get('random_state', seed),
                            verbose=kwargs.get('verbose', 0),
                            )

    return run_regression(
        model, "BaggingRegressor",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, X_scaler_type, y_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        progress_tracker=progress_tracker
    )
