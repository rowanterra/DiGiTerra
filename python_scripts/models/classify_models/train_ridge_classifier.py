from sklearn.linear_model import RidgeClassifier
from python_scripts.preprocessing.run_classification_pipeline import run_classification

def train_ridge_classifier(train_data, target_variables, use_stratified_split,
                X, y, stratifyColumn, units,
                X_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols, testSize,
                feature_selection_method='none', feature_selection_k=None,
                outlier_method='none', outlier_action='remove',
                hyperparameter_search='none', search_cv_folds=5, search_n_iter=50, **kwargs):

    model = RidgeClassifier(alpha=kwargs.get('alpha', 1.0),
                           fit_intercept=kwargs.get('fit_intercept', True),
                           copy_X=kwargs.get('copy_X', True),
                           max_iter=kwargs.get('max_iter', None),
                           tol=kwargs.get('tol', 1e-4),
                           class_weight=kwargs.get('class_weight', None),
                           solver=kwargs.get('solver', 'auto'),
                           positive=kwargs.get('positive', False),
                           random_state=kwargs.get('random_state', seed),
                           )

    return run_classification(
        model, "RidgeClassifier",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, X_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        modeling_mode=kwargs.get('modeling_mode', 'simple')
    )
