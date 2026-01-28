from sklearn.ensemble import HistGradientBoostingRegressor
from python_scripts.preprocessing.run_regression_pipeline import run_regression

def train_hist_gradient_boosting_regressor(model_type, train_data, target_variables, use_stratified_split,
                X, y, stratifyColumn, units,
                X_scaler_type, y_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols, testSize,
                feature_selection_method='none', feature_selection_k=None,
                outlier_method='none', outlier_action='remove',
                hyperparameter_search='none', search_cv_folds=5, search_n_iter=50,
                progress_tracker=None, **kwargs):

    model = HistGradientBoostingRegressor(loss=kwargs.get('loss', 'squared_error'),
                                          learning_rate=kwargs.get('learning_rate', 0.1),
                                          max_iter=kwargs.get('max_iter', 100),
                                          max_leaf_nodes=kwargs.get('max_leaf_nodes', 31),
                                          max_depth=kwargs.get('max_depth', None),
                                          min_samples_leaf=kwargs.get('min_samples_leaf', 20),
                                          l2_regularization=kwargs.get('l2_regularization', 0.0),
                                          max_bins=kwargs.get('max_bins', 255),
                                          categorical_features=kwargs.get('categorical_features', None),
                                          monotonic_cst=kwargs.get('monotonic_cst', None),
                                          interaction_cst=kwargs.get('interaction_cst', None),
                                          warm_start=kwargs.get('warm_start', False),
                                          early_stopping=kwargs.get('early_stopping', 'auto'),
                                          scoring=kwargs.get('scoring', 'loss'),
                                          validation_fraction=kwargs.get('validation_fraction', 0.1),
                                          n_iter_no_change=kwargs.get('n_iter_no_change', 10),
                                          tol=kwargs.get('tol', 1e-7),
                                          verbose=kwargs.get('verbose', 0),
                                          random_state=kwargs.get('random_state', seed),
                                          )

    return run_regression(
        model, "HistGradientBoostingRegressor",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, X_scaler_type, y_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        progress_tracker=progress_tracker
    )
