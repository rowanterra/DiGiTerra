import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor

from python_scripts.preprocessing.run_regression_pipeline import run_regression


class RadiusNeighborsWithFallback(RadiusNeighborsRegressor):
    """RadiusNeighborsRegressor that answers every row.

    scikit-learn returns NaN for a query with no training row inside the radius.
    On standardized features a radius of 1 leaves most rows without a neighbor,
    and a NaN prediction breaks every metric and chart downstream. A row with an
    empty neighborhood takes the value of its single nearest training row
    instead, and n_fallback_ records how many rows that was on the last predict
    call so the report can state it.
    """

    def fit(self, X, y):
        super().fit(X, y)
        self._nearest = KNeighborsRegressor(
            n_neighbors=1, algorithm=self.algorithm, leaf_size=self.leaf_size, p=self.p, metric=self.metric
        ).fit(X, y)
        self.n_fallback_ = 0
        return self

    def predict(self, X):
        pred = np.asarray(super().predict(X), dtype=float)
        empty = np.isnan(pred) if pred.ndim == 1 else np.isnan(pred).any(axis=1)
        self.n_fallback_ = int(empty.sum())
        if self.n_fallback_:
            Xa = np.asarray(X)
            pred[empty] = self._nearest.predict(Xa[empty])
        return pred


def train_radius_neighbors_regressor(
    model_type,
    train_data,
    target_variables,
    use_stratified_split,
    X,
    y,
    stratifyColumn,
    units,
    X_scaler_type,
    y_scaler_type,
    seed,
    quantileBinDict,
    sigfig,
    useTransformer,
    categorical_cols,
    testSize,
    feature_selection_method="none",
    feature_selection_k=None,
    outlier_method="none",
    outlier_action="remove",
    hyperparameter_search="none",
    search_cv_folds=5,
    search_n_iter=50,
    progress_tracker=None,
    **kwargs,
):
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

    model = RadiusNeighborsWithFallback(
        radius=kwargs.get("radius", 1.0),
        weights=kwargs.get("weights", "uniform"),
        algorithm=kwargs.get("algorithm", "auto"),
        leaf_size=kwargs.get("leaf_size", 30),
        p=kwargs.get("p", 2),
        metric=kwargs.get("metric", "minkowski"),
        metric_params=kwargs.get("metric_params", None),
        n_jobs=kwargs.get("n_jobs", None),
    )

    return run_regression(
        model,
        "RadiusNeighborsRegressor",
        train_data,
        target_variables,
        use_stratified_split,
        X,
        y,
        stratifyColumn,
        units,
        X_scaler_type,
        y_scaler_type,
        seed,
        sigfig,
        quantileBinDict,
        useTransformer,
        categorical_cols,
        testSize,
        feature_selection_method=feature_selection_method,
        feature_selection_k=feature_selection_k,
        outlier_method=outlier_method,
        outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search,
        search_cv_folds=search_cv_folds,
        search_n_iter=search_n_iter,
        progress_tracker=progress_tracker,
    )
