from python_scripts.preprocessing.run_clustering_pipeline import run_clustering
from sklearn.cluster import AgglomerativeClustering

def train_agglomerative(train_data,
                X, units,
                X_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols,test_size, k_min, k_max, **kwargs):
    
    model = AgglomerativeClustering(n_clusters=kwargs.get('n_clusters', 2), 
                                    metric=kwargs.get('metric', "euclidean"), 
                                    memory=kwargs.get('memory', None),
                                    connectivity=kwargs.get('connectivity', None), 
                                    compute_full_tree=kwargs.get('compute_full_tree', "auto"), 
                                    linkage=kwargs.get('linkage', "ward"), 
                                    distance_threshold=kwargs.get('distance_threshold', None), 
                                    compute_distances=kwargs.get('compute_distances', False)  )

    return run_clustering(
        model,
        "agglo",
        train_data,
        X,
        units, X_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols,test_size, k_min, k_max)
