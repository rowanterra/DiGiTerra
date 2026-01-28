from python_scripts.preprocessing.run_clustering_pipeline import run_clustering
from sklearn.mixture import GaussianMixture

def train_gmm(train_data,
                X, units,
                X_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols,test_size, k_min, k_max, **kwargs):
    

    model = GaussianMixture(
                    n_components=kwargs.get('n_components', 1),
                    covariance_type=kwargs.get('covariance_type', 'full'),
                    tol=kwargs.get('tol', 0.001),
                    reg_covar=kwargs.get('reg_covar', 0.000001),
                    max_iter=kwargs.get('max_iter', 100),
                    n_init=kwargs.get('n_init', 1),
                    init_params=kwargs.get('init_params', 'kmeans'),
                    weights_init=kwargs.get('weights_init', None),
                    means_init=kwargs.get('means_init', None),
                    precisions_init=kwargs.get('precisions_init', None),
                    warm_start=kwargs.get('warm_start', False),
                    verbose=kwargs.get('verbose', 0),
                    verbose_interval=kwargs.get('verbose_interval', 10),
    )

    return run_clustering(
        model,
        "gmm",
        train_data,
        X,
        units, X_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols,test_size, k_min, k_max)
