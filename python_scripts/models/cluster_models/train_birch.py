from sklearn.cluster import Birch

from python_scripts.preprocessing.run_clustering_pipeline import run_clustering


def train_birch(train_data, X, units, X_scaler_type,
                seed, sigfig, quantileBinDict, useTransformer, categorical_cols, test_size, **kwargs):
    model = Birch(
        threshold=kwargs.get("threshold", 0.5),
        branching_factor=kwargs.get("branching_factor", 50),
        n_clusters=kwargs.get("n_clusters", 3),
        compute_labels=kwargs.get("compute_labels", True),
        # Note: 'copy' parameter was removed in scikit-learn 1.2+
    )

    return run_clustering(
        model, "birch",
        train_data,
        X,
        units, X_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, test_size,
        k_min=2, k_max=8
    )
