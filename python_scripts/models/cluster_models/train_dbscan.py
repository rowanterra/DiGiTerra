from sklearn.cluster import DBSCAN

from python_scripts.preprocessing.run_clustering_pipeline import run_clustering


def train_dbscan(train_data, X, units, X_scaler_type,
                 seed, sigfig, quantileBinDict, useTransformer, categorical_cols, test_size, **kwargs):
    model = DBSCAN(
        eps=kwargs.get("eps", 0.5),
        min_samples=kwargs.get("min_samples", 5),
    )

    return run_clustering(
        model, "dbscan",
        train_data,
        X,
        units, X_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, test_size,
        k_min=2, k_max=8
    )
