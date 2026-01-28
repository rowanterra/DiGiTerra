from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from python_scripts.preprocessing.run_classification_pipeline import run_classification

def train_linear_discriminant_analysis(train_data, target_variables, use_stratified_split,
                X, y, stratifyColumn, units,
                X_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols, testSize,
                feature_selection_method='none', feature_selection_k=None,
                outlier_method='none', outlier_action='remove',
                hyperparameter_search='none', search_cv_folds=5, search_n_iter=50, **kwargs):

    model = LinearDiscriminantAnalysis(solver=kwargs.get('solver', 'svd'),
                                     shrinkage=kwargs.get('shrinkage', None),
                                     priors=kwargs.get('priors', None),
                                     n_components=kwargs.get('n_components', None),
                                     store_covariance=kwargs.get('store_covariance', False),
                                     tol=kwargs.get('tol', 1e-4),
                                     covariance_estimator=kwargs.get('covariance_estimator', None),
                                     )

    return run_classification(
        model, "LinearDiscriminantAnalysis",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, X_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        modeling_mode=kwargs.get('modeling_mode', 'simple')
    )
