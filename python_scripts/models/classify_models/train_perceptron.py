from sklearn.linear_model import Perceptron

from python_scripts.models.defaults import sk_default
from python_scripts.preprocessing.run_classification_pipeline import run_classification


def train_perceptron(
    train_data,
    target_variables,
    use_stratified_split,
    X,
    y,
    stratifyColumn,
    units,
    X_scaler_type,
    seed,
    quantileBinDict,
    sigfig,
    useTransformer,
    categorical_cols,
    testSize,
    feature_selection_method="none",
    feature_selection_k=None,
    outlier_method="none",
    outlier_action="remove",
    hyperparameter_search="none",
    search_cv_folds=5,
    search_n_iter=50,
    **kwargs,
):
    # Perceptron is a classifier. The earlier trainer had the regression signature
    # and failed on every numeric target, so the model was registered without one.
    names = (
        "penalty",
        "alpha",
        "l1_ratio",
        "fit_intercept",
        "max_iter",
        "tol",
        "shuffle",
        "verbose",
        "eta0",
        "n_jobs",
        "early_stopping",
        "validation_fraction",
        "n_iter_no_change",
        "class_weight",
        "warm_start",
    )
    model = Perceptron(
        **{n: kwargs.get(n, sk_default(Perceptron, n)) for n in names},
        random_state=kwargs.get("random_state", seed),
    )

    return run_classification(
        model,
        "Perceptron",
        train_data,
        target_variables,
        use_stratified_split,
        X,
        y,
        stratifyColumn,
        units,
        X_scaler_type,
        seed,
        sigfig,
        quantileBinDict,
        useTransformer,
        categorical_cols,
        testSize,
        feature_selection_method=feature_selection_method,
        feature_selection_k=feature_selection_k,
        outlier_method=outlier_method,
        outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search,
        search_cv_folds=search_cv_folds,
        search_n_iter=search_n_iter,
        modeling_mode=kwargs.get("modeling_mode", "simple"),
    )
