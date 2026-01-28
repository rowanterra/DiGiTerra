from sklearn.naive_bayes import CategoricalNB
from sklearn.multioutput import MultiOutputClassifier
from python_scripts.preprocessing.run_classification_pipeline import run_classification
import logging

logger = logging.getLogger(__name__)

def train_categorical_nb(train_data, target_variables, use_stratified_split,
                X, y, stratifyColumn, units,
                X_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols, testSize,
                feature_selection_method='none', feature_selection_k=None,
                outlier_method='none', outlier_action='remove',
                hyperparameter_search='none', search_cv_folds=5, search_n_iter=50, **kwargs):
    # CategoricalNB requires non-negative integer values (categories)
    # If a scaler that can produce negative values is used, it will be automatically changed to MinMaxScaler
    effective_scaler = X_scaler_type
    if X_scaler_type in ['standard', 'robust']:
        # StandardScaler and RobustScaler can produce negative values, so we use MinMaxScaler instead
        # Note: CategoricalNB actually expects categorical (integer) data, not continuous scaled data
        # But if user wants scaling, we'll use MinMaxScaler to ensure non-negative values
        effective_scaler = 'minmax'
        logger.info(f"CategoricalNB requires non-negative values. Changing scaler from {X_scaler_type} to MinMaxScaler.")

    model = MultiOutputClassifier(CategoricalNB(alpha=kwargs.get('alpha', 1.0),
                                              fit_prior=kwargs.get('fit_prior', True),
                                              class_prior=kwargs.get('class_prior', None),
                                              min_categories=kwargs.get('min_categories', None),
                                              force_alpha=kwargs.get('force_alpha', True),
                                              ))

    return run_classification(
        model, "CategoricalNB",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, effective_scaler,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        modeling_mode=kwargs.get('modeling_mode', 'simple')
    )
