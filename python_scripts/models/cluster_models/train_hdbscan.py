from sklearn.cluster import HDBSCAN
from python_scripts.preprocessing.run_clustering_pipeline import run_clustering

def train_hdbscan(train_data, X, units, X_scaler_type,
                 seed, sigfig, quantileBinDict, useTransformer, categorical_cols, test_size, **kwargs):
    model = HDBSCAN(
        min_cluster_size=kwargs.get("min_cluster_size", 5),
        min_samples=kwargs.get("min_samples", None),
        cluster_selection_epsilon=kwargs.get("cluster_selection_epsilon", 0.0),
        max_cluster_size=kwargs.get("max_cluster_size", None),
        metric=kwargs.get("metric", "euclidean"),
        metric_params=kwargs.get("metric_params", None),
        alpha=kwargs.get("alpha", 1.0),
        algorithm=kwargs.get("algorithm", "auto"),
        leaf_size=kwargs.get("leaf_size", 40),
        cluster_selection_method=kwargs.get("cluster_selection_method", "eom"),
        allow_single_cluster=kwargs.get("allow_single_cluster", False),
        store_centers=kwargs.get("store_centers", None),
        # Note: 'copy' parameter removed in scikit-learn 1.2+ for HDBSCAN
        n_jobs=kwargs.get("n_jobs", None),
    )

    return run_clustering(
        model, "hdbscan",
        train_data,
        X,
        units, X_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, test_size,
        k_min=2, k_max=8
    )
