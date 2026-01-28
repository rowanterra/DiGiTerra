from sklearn.neighbors import RadiusNeighborsRegressor
from python_scripts.preprocessing.run_regression_pipeline import run_regression
import pandas as pd
import numpy as np

def train_radius_neighbors_regressor(model_type, train_data, target_variables, use_stratified_split,
                X, y, stratifyColumn, units,
                X_scaler_type, y_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols, testSize,
                feature_selection_method='none', feature_selection_k=None,
                outlier_method='none', outlier_action='remove',
                hyperparameter_search='none', search_cv_folds=5, search_n_iter=50,
                progress_tracker=None, **kwargs):
    # RadiusNeighborsRegressor cannot handle NaN values
    # Ensure data is clean before training
    if isinstance(X, pd.DataFrame):
        X = X.dropna()
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.dropna()
    # Align X and y indices after dropping NaN
    if isinstance(X, pd.DataFrame) and isinstance(y, (pd.DataFrame, pd.Series)):
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        train_data = train_data.loc[common_idx]

    model = RadiusNeighborsRegressor(radius=kwargs.get('radius', 1.0),
                                     weights=kwargs.get('weights', 'uniform'),
                                     algorithm=kwargs.get('algorithm', 'auto'),
                                     leaf_size=kwargs.get('leaf_size', 30),
                                     p=kwargs.get('p', 2),
                                     metric=kwargs.get('metric', 'minkowski'),
                                     metric_params=kwargs.get('metric_params', None),
                                     n_jobs=kwargs.get('n_jobs', None),
                                     )

    return run_regression(
        model, "RadiusNeighborsRegressor",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, X_scaler_type, y_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        progress_tracker=progress_tracker
    )
