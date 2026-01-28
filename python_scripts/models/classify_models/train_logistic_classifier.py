from sklearn.linear_model import LogisticRegression
from python_scripts.preprocessing.run_classification_pipeline import run_classification

def train_logistic_classifier(train_data, target_variables, use_stratified_split,
                X, y, stratifyColumn, units,
                X_scaler_type,
                seed, quantileBinDict, sigfig,useTransformer, categorical_cols, testSize,
                feature_selection_method='none', feature_selection_k=None,
                outlier_method='none', outlier_action='remove',
                hyperparameter_search='none', search_cv_folds=5, search_n_iter=50, **kwargs):

    # Note: multi_class parameter removed in sklearn 1.8.0+ - multiclass handling is now automatic
    # Build parameters dict, excluding multi_class for compatibility with newer sklearn versions
    model_params = {
        'dual': kwargs.get('Class_LogisticDual', False),
        'fit_intercept': kwargs.get('Class_LogisticFitIntercept', True),
        'warm_start': kwargs.get('Class_LogisticWarmStart', False),
        'solver': kwargs.get('Class_LogisticSolver', 'lbfgs'),
        'C': kwargs.get('Class_CLogistic', 1),
        'penalty': kwargs.get('Class_Logistic_penalty', 'l2'),
        'tol': kwargs.get('Class_LogisticTol', 0.0001),
        'intercept_scaling': kwargs.get('Class_Logisticintercept_scaling', 1),
        'class_weight': kwargs.get('Class_LogisticClassWeight', None),
        'max_iter': kwargs.get('Class_LogisticMaxIterations', 100),
        'verbose': kwargs.get('Class_LogisticVerbose', 0),
        'n_jobs': kwargs.get('Class_LogisticNJobs', None),
        'l1_ratio': kwargs.get('Class_Logisticl1Ratio', None),
        'random_state': seed
    }
    
    # Try creating model without multi_class first (for sklearn >= 1.8.0)
    # If that fails, try with multi_class for older versions
    try:
        model = LogisticRegression(**model_params)
    except TypeError as e:
        if 'multi_class' in str(e):
            # If error mentions multi_class, it's likely an older sklearn expecting it
            # But since we're getting "unexpected keyword argument", it's newer sklearn
            # So the issue is we're passing it when we shouldn't - already handled by not including it
            raise
        else:
            # Different TypeError - try with multi_class for older sklearn compatibility
            model_params['multi_class'] = kwargs.get('Class_LogisticMultiClass', 'auto')
            model = LogisticRegression(**model_params)
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"Target variable shape: {y.shape if hasattr(y, 'shape') else 'N/A'}")

    return run_classification(
        model, "LogisticClassifier",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, X_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols,testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        modeling_mode=kwargs.get('modeling_mode', 'simple')
    )
