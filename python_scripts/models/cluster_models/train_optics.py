from sklearn.cluster import OPTICS
import numpy as np
from python_scripts.preprocessing.run_clustering_pipeline import run_clustering

def train_optics(train_data, X, units, X_scaler_type,
                 seed, sigfig, quantileBinDict, useTransformer, categorical_cols, test_size, **kwargs):
    model = OPTICS(
        min_samples=kwargs.get("min_samples", 5),
        max_eps=kwargs.get("max_eps", np.inf),
        metric=kwargs.get("metric", "minkowski"),
        p=kwargs.get("p", 2),
        metric_params=kwargs.get("metric_params", None),
        cluster_method=kwargs.get("cluster_method", "xi"),
        eps=kwargs.get("eps", None),
        xi=kwargs.get("xi", 0.05),
        predecessor_correction=kwargs.get("predecessor_correction", True),
        min_cluster_size=kwargs.get("min_cluster_size", None),
        algorithm=kwargs.get("algorithm", "auto"),
        leaf_size=kwargs.get("leaf_size", 30),
        memory=kwargs.get("memory", None),
        n_jobs=kwargs.get("n_jobs", None),
    )

    return run_clustering(
        model, "optics",
        train_data,
        X,
        units, X_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, test_size,
        k_min=2, k_max=8
    )
