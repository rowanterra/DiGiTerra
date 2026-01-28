from sklearn.cluster import MeanShift
from python_scripts.preprocessing.run_clustering_pipeline import run_clustering

def train_meanshift(train_data, X, units, X_scaler_type,
                 seed, sigfig, quantileBinDict, useTransformer, categorical_cols, test_size, **kwargs):
    model = MeanShift(
        bandwidth=kwargs.get("bandwidth", None),
        seeds=kwargs.get("seeds", None),
        bin_seeding=kwargs.get("bin_seeding", False),
        min_bin_freq=kwargs.get("min_bin_freq", 1),
        cluster_all=kwargs.get("cluster_all", True),
        n_jobs=kwargs.get("n_jobs", None),
        max_iter=kwargs.get("max_iter", 300),
    )

    return run_clustering(
        model, "meanshift",
        train_data,
        X,
        units, X_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, test_size,
        k_min=2, k_max=8
    )
