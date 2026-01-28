from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from python_scripts.preprocessing.run_classification_pipeline import run_classification


def train_sgd_classifier(train_data, target_variables, use_stratified_split,
                         X, y, stratifyColumn, units,
                         X_scaler_type,
                         seed, quantileBinDict, sigfig, useTransformer, categorical_cols, testSize,
                         feature_selection_method='none', feature_selection_k=None,
                         outlier_method='none', outlier_action='remove',
                         hyperparameter_search='none', search_cv_folds=5, search_n_iter=50, **kwargs):
    model = MultiOutputClassifier(SGDClassifier(
        loss=kwargs.get("loss", "hinge"),
        penalty=kwargs.get("penalty", "l2"),
        alpha=kwargs.get("alpha", 0.0001),
        l1_ratio=kwargs.get("l1_ratio", 0.15),
        fit_intercept=kwargs.get("fit_intercept", True),
        max_iter=kwargs.get("max_iter", 1000),
        tol=kwargs.get("tol", 1e-3),
        shuffle=kwargs.get("shuffle", True),
        verbose=kwargs.get("verbose", 0),
        epsilon=kwargs.get("epsilon", 0.1),
        n_jobs=kwargs.get("n_jobs", None),
        random_state=kwargs.get("random_state", seed),
        learning_rate=kwargs.get("learning_rate", "optimal"),
        eta0=kwargs.get("eta0", 0.0),
        power_t=kwargs.get("power_t", 0.5),
        early_stopping=kwargs.get("early_stopping", False),
        validation_fraction=kwargs.get("validation_fraction", 0.1),
        n_iter_no_change=kwargs.get("n_iter_no_change", 5),
        class_weight=kwargs.get("class_weight", None),
        warm_start=kwargs.get("warm_start", False),
        average=kwargs.get("average", False),
    ))

    return run_classification(
        model, "SGDClassifier",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, X_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        modeling_mode=kwargs.get('modeling_mode', 'simple')
    )
