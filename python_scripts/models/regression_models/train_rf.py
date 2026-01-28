from sklearn.ensemble import RandomForestRegressor
from python_scripts.preprocessing.run_regression_pipeline import run_regression

def train_rf(model_type, train_data, target_variables, use_stratified_split,
                X, y, stratifyColumn, units,
                X_scaler_type, y_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols,testSize,
                feature_selection_method='none', feature_selection_k=None,
                outlier_method='none', outlier_action='remove',
                hyperparameter_search='none', search_cv_folds=5, search_n_iter=50,
                progress_tracker=None, modeling_mode='simple', **kwargs):

    model = RandomForestRegressor(n_estimators=kwargs.get('n_estimators', 100),
                                                criterion=kwargs.get('criterion', 'squared_error'),
                                                max_depth=kwargs.get('max_depth', None), 
                                                min_samples_split=kwargs.get('min_samples_split', 2), 
                                                min_samples_leaf=kwargs.get('min_samples_leaf', 1), 
                                                min_weight_fraction_leaf=kwargs.get('min_weight_fraction_leaf', 0.0),
                                                max_features=kwargs.get('max_features', 'sqrt'),
                                                max_leaf_nodes=kwargs.get('max_leaf_nodes', None),
                                                min_impurity_decrease=kwargs.get('min_impurity_decrease', 0.0),
                                                bootstrap=kwargs.get('bootstrap', True),
                                                oob_score=kwargs.get('oob_score', False),
                                                n_jobs=kwargs.get('n_jobs', None),
                                                random_state=kwargs.get('random_state', None),
                                                verbose=kwargs.get('verbose', 0),
                                                warm_start=kwargs.get('warm_start', False),
                                                ccp_alpha=kwargs.get('ccp_alpha', 0.0),
                                                max_samples=kwargs.get('max_samples', None),
                                                )

    return run_regression(
        model, "RandomForest",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, X_scaler_type, y_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols,testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        progress_tracker=progress_tracker, modeling_mode=modeling_mode
    )
