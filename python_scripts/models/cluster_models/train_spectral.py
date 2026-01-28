from sklearn.cluster import SpectralClustering

from python_scripts.preprocessing.run_clustering_pipeline import run_clustering


def train_spectral(train_data, X, units, X_scaler_type,
                   seed, sigfig, quantileBinDict, useTransformer, categorical_cols, test_size, **kwargs):
    model = SpectralClustering(
        n_clusters=kwargs.get("n_clusters", 3),
        eigen_solver=kwargs.get("eigen_solver", None),
        n_components=kwargs.get("n_components", None),
        random_state=kwargs.get("random_state", seed),
        n_init=kwargs.get("n_init", 10),
        gamma=kwargs.get("gamma", 1.0),
        affinity=kwargs.get("affinity", "rbf"),
        n_neighbors=kwargs.get("n_neighbors", 10),
        eigen_tol=kwargs.get("eigen_tol", 0.0),
        assign_labels=kwargs.get("assign_labels", "kmeans"),
        degree=kwargs.get("degree", 3),
        coef0=kwargs.get("coef0", 1),
        kernel_params=kwargs.get("kernel_params", None),
        n_jobs=kwargs.get("n_jobs", None),
        verbose=kwargs.get("verbose", False),
    )

    return run_clustering(
        model, "spectral",
        train_data,
        X,
        units, X_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, test_size,
        k_min=2, k_max=8
    )
