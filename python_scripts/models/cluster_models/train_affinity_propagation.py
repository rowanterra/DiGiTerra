from sklearn.cluster import AffinityPropagation
from python_scripts.preprocessing.run_clustering_pipeline import run_clustering

def train_affinity_propagation(train_data, X, units, X_scaler_type,
                 seed, sigfig, quantileBinDict, useTransformer, categorical_cols, test_size, **kwargs):
    model = AffinityPropagation(
        damping=kwargs.get("damping", 0.5),
        max_iter=kwargs.get("max_iter", 200),
        convergence_iter=kwargs.get("convergence_iter", 15),
        # Note: 'copy' parameter removed in scikit-learn 1.2+
        preference=kwargs.get("preference", None),
        affinity=kwargs.get("affinity", "euclidean"),
        verbose=kwargs.get("verbose", False),
        random_state=kwargs.get("random_state", seed),
    )

    return run_clustering(
        model, "affinity_propagation",
        train_data,
        X,
        units, X_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, test_size,
        k_min=2, k_max=8
    )
