from sklearn.cluster import BisectingKMeans
from python_scripts.preprocessing.run_clustering_pipeline import run_clustering

def train_bisecting_kmeans(train_data,
                X, units,
                X_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols, test_size, k_min, k_max, **kwargs):
    
    model = BisectingKMeans(
            n_clusters=kwargs.get('n_clusters', 8),
            init=kwargs.get('init', 'k-means++'),
            n_init=kwargs.get('n_init', 1),
            random_state=kwargs.get('random_state', seed),
            max_iter=kwargs.get('max_iter', 300),
            verbose=kwargs.get('verbose', 0),
            tol=kwargs.get('tol', 1e-4),
            copy_x=kwargs.get('copy_x', True),
            algorithm=kwargs.get('algorithm', 'lloyd'),
            bisecting_strategy=kwargs.get('bisecting_strategy', 'biggest_inertia'),
            )

    return run_clustering(
        model,
        "bisecting_kmeans",
        train_data,
        X,
        units, X_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, test_size, k_min, k_max)
