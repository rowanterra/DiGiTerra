from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from python_scripts.models.defaults import sk_default
from python_scripts.preprocessing.run_regression_pipeline import run_regression

# The defaults of the scikit-learn classes that take each parameter.
_DEFAULTS = {
    "degree": sk_default(PolynomialFeatures, "degree"),
    "interaction_only": sk_default(PolynomialFeatures, "interaction_only"),
    "fit_intercept": sk_default(LinearRegression, "fit_intercept"),
    "positive": sk_default(LinearRegression, "positive"),
}


class PolynomialRegression(RegressorMixin, BaseEstimator):
    """Least squares on polynomial features: PolynomialFeatures followed by LinearRegression.

    One estimator with flat parameters rather than a Pipeline, so the guided table
    lists degree, interaction_only, fit_intercept and positive by name and a bundle
    names one type of this project. Each parameter falls back to the default of the
    scikit-learn class that uses it; include_bias is fixed at False because the
    regression fits its own intercept.
    """

    def __init__(
        self,
        *,
        degree=_DEFAULTS["degree"],
        interaction_only=_DEFAULTS["interaction_only"],
        fit_intercept=_DEFAULTS["fit_intercept"],
        positive=_DEFAULTS["positive"],
    ):
        self.degree = degree
        self.interaction_only = interaction_only
        self.fit_intercept = fit_intercept
        self.positive = positive

    def fit(self, X, y):
        self.features_ = PolynomialFeatures(
            degree=self.degree, interaction_only=self.interaction_only, include_bias=False
        )
        Z = self.features_.fit_transform(X)
        self.linear_ = LinearRegression(fit_intercept=self.fit_intercept, positive=self.positive).fit(Z, y)
        self.n_features_in_ = self.features_.n_features_in_
        if hasattr(self.features_, "feature_names_in_"):
            self.feature_names_in_ = self.features_.feature_names_in_
        return self

    def predict(self, X):
        return self.linear_.predict(self.features_.transform(X))


def train_polynomial(
    model_type,
    train_data,
    target_variables,
    use_stratified_split,
    X,
    y,
    stratifyColumn,
    units,
    X_scaler_type,
    y_scaler_type,
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
    progress_tracker=None,
    modeling_mode="simple",
    **kwargs,
):
    names = ("degree", "interaction_only", "fit_intercept", "positive")
    model = PolynomialRegression(**{n: kwargs[n] for n in names if kwargs.get(n) is not None})

    return run_regression(
        model,
        "PolynomialRegression",
        train_data,
        target_variables,
        use_stratified_split,
        X,
        y,
        stratifyColumn,
        units,
        X_scaler_type,
        y_scaler_type,
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
        progress_tracker=progress_tracker,
        modeling_mode=modeling_mode,
    )
