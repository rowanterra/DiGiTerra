from sklearn.cluster import KMeans
from python_scripts.preprocessing.run_clustering_pipeline import run_clustering


def train_kmeans(train_data,
                X, units,
                X_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols,test_size, k_min, k_max, **kwargs):
    
    model = KMeans(
            n_clusters=kwargs.get('n_clusters', 8),
            init=kwargs.get('init', 'k-means++'),
            n_init=kwargs.get('n_init', 'auto'),
            max_iter=kwargs.get('max_iter', 300),
            tol=kwargs.get('tol', 0.0001),
            verbose=kwargs.get('verbose', 0),
            copy_x=kwargs.get('copy_x', True),
            algorithm=kwargs.get('algorithm', 'lloyd'),
            random_state=kwargs.get('random_state', seed),
            )

    return run_clustering(
        model,
        "kmeans",
        train_data,
        X,
        units, X_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, test_size, k_min, k_max)
