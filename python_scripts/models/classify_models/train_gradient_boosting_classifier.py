from sklearn.ensemble import GradientBoostingClassifier
from python_scripts.preprocessing.run_classification_pipeline import run_classification

def train_gradient_boosting_classifier(train_data, target_variables, use_stratified_split,
                X, y, stratifyColumn, units,
                X_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols, testSize,
                feature_selection_method='none', feature_selection_k=None,
                outlier_method='none', outlier_action='remove',
                hyperparameter_search='none', search_cv_folds=5, search_n_iter=50, **kwargs):

    model = GradientBoostingClassifier(loss=kwargs.get('loss', 'log_loss'),
                                       learning_rate=kwargs.get('learning_rate', 0.1),
                                       n_estimators=kwargs.get('n_estimators', 100),
                                       subsample=kwargs.get('subsample', 1.0),
                                       criterion=kwargs.get('criterion', 'friedman_mse'),
                                       min_samples_split=kwargs.get('min_samples_split', 2),
                                       min_samples_leaf=kwargs.get('min_samples_leaf', 1),
                                       min_weight_fraction_leaf=kwargs.get('min_weight_fraction_leaf', 0.0),
                                       max_depth=kwargs.get('max_depth', 3),
                                       min_impurity_decrease=kwargs.get('min_impurity_decrease', 0.0),
                                       init=kwargs.get('init', None),
                                       random_state=kwargs.get('random_state', seed),
                                       max_features=kwargs.get('max_features', None),
                                       verbose=kwargs.get('verbose', 0),
                                       max_leaf_nodes=kwargs.get('max_leaf_nodes', None),
                                       warm_start=kwargs.get('warm_start', False),
                                       validation_fraction=kwargs.get('validation_fraction', 0.1),
                                       n_iter_no_change=kwargs.get('n_iter_no_change', None),
                                       tol=kwargs.get('tol', 1e-4),
                                       ccp_alpha=kwargs.get('ccp_alpha', 0.0),
                                       )

    return run_classification(
        model, "GradientBoostingClassifier",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, X_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        modeling_mode=kwargs.get('modeling_mode', 'simple')
    )
