from sklearn.cluster import MiniBatchKMeans
from python_scripts.preprocessing.run_clustering_pipeline import run_clustering

def train_minibatch_kmeans(train_data,
                X, units,
                X_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols, test_size, k_min, k_max, **kwargs):
    
    model = MiniBatchKMeans(
            n_clusters=kwargs.get('n_clusters', 8),
            init=kwargs.get('init', 'k-means++'),
            max_iter=kwargs.get('max_iter', 100),
            batch_size=kwargs.get('batch_size', 1024),
            verbose=kwargs.get('verbose', 0),
            compute_labels=kwargs.get('compute_labels', True),
            random_state=kwargs.get('random_state', seed),
            tol=kwargs.get('tol', 0.0),
            max_no_improvement=kwargs.get('max_no_improvement', 10),
            n_init=kwargs.get('n_init', 3),
            reassignment_ratio=kwargs.get('reassignment_ratio', 0.01),
            )

    return run_clustering(
        model,
        "minibatch_kmeans",
        train_data,
        X,
        units, X_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, test_size, k_min, k_max)
