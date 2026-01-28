from sklearn.naive_bayes import ComplementNB
from sklearn.multioutput import MultiOutputClassifier
from python_scripts.preprocessing.run_classification_pipeline import run_classification
import logging

logger = logging.getLogger(__name__)

def train_complement_nb(train_data, target_variables, use_stratified_split,
                X, y, stratifyColumn, units,
                X_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols, testSize,
                feature_selection_method='none', feature_selection_k=None,
                outlier_method='none', outlier_action='remove',
                hyperparameter_search='none', search_cv_folds=5, search_n_iter=50, **kwargs):
    """
    Train ComplementNB classifier.
    
    Note: ComplementNB requires non-negative values. If StandardScaler is selected,
    it will be automatically changed to MinMaxScaler to ensure non-negative values.
    """
    # ComplementNB requires non-negative values
    # StandardScaler can produce negative values, so we use MinMaxScaler instead
    effective_scaler = X_scaler_type
    if X_scaler_type in ['StandardScaler', 'standard', 'Standard', 'robust']:
        effective_scaler = 'minmax'
        logger.info(f"ComplementNB requires non-negative values. Changing scaler from {X_scaler_type} to MinMaxScaler.")
    
    model = MultiOutputClassifier(ComplementNB(alpha=kwargs.get('alpha', 1.0),
                                              fit_prior=kwargs.get('fit_prior', True),
                                              class_prior=kwargs.get('class_prior', None),
                                              norm=kwargs.get('norm', False),
                                              force_alpha=kwargs.get('force_alpha', True),
                                              ))

    return run_classification(
        model, "ComplementNB",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, effective_scaler,  # Use effective_scaler instead of X_scaler_type
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        modeling_mode=kwargs.get('modeling_mode', 'simple')
    )
