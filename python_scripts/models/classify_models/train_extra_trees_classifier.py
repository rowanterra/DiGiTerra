from sklearn.ensemble import ExtraTreesClassifier

from python_scripts.preprocessing.run_classification_pipeline import run_classification


def train_extra_trees_classifier(train_data, target_variables, use_stratified_split,
                                 X, y, stratifyColumn, units,
                                 X_scaler_type,
                                 seed, quantileBinDict, sigfig, useTransformer, categorical_cols, testSize,
                                 feature_selection_method='none', feature_selection_k=None,
                                 outlier_method='none', outlier_action='remove',
                                 hyperparameter_search='none', search_cv_folds=5, search_n_iter=50, **kwargs):
    model = ExtraTreesClassifier(
        n_estimators=kwargs.get("n_estimators", 100),
        criterion=kwargs.get("criterion", "gini"),
        max_depth=kwargs.get("max_depth", None),
        min_samples_split=kwargs.get("min_samples_split", 2),
        min_samples_leaf=kwargs.get("min_samples_leaf", 1),
        min_weight_fraction_leaf=kwargs.get("min_weight_fraction_leaf", 0.0),
        max_features=kwargs.get("max_features", "sqrt"),
        max_leaf_nodes=kwargs.get("max_leaf_nodes", None),
        min_impurity_decrease=kwargs.get("min_impurity_decrease", 0.0),
        bootstrap=kwargs.get("bootstrap", False),
        oob_score=kwargs.get("oob_score", False),
        n_jobs=kwargs.get("n_jobs", None),
        random_state=kwargs.get("random_state", seed),
        verbose=kwargs.get("verbose", 0),
        warm_start=kwargs.get("warm_start", False),
        class_weight=kwargs.get("class_weight", None),
        ccp_alpha=kwargs.get("ccp_alpha", 0.0),
        max_samples=kwargs.get("max_samples", None),
    )

    return run_classification(
        model, "ExtraTreesClassifier",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, X_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        modeling_mode=kwargs.get('modeling_mode', 'simple')
    )
