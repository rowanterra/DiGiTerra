from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from python_scripts.preprocessing.run_classification_pipeline import run_classification

def train_svc(train_data, target_variables, use_stratified_split,
                X, y, stratifyColumn, units,
                X_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols, testSize,
                feature_selection_method='none', feature_selection_k=None,
                outlier_method='none', outlier_action='remove',
                hyperparameter_search='none', search_cv_folds=5, search_n_iter=50, **kwargs):

    model = MultiOutputClassifier(SVC(C=kwargs.get('C', 1.0), 
                kernel=kwargs.get('kernel', 'rbf'), 
                degree=kwargs.get('degree', 3), 
                gamma=kwargs.get('gamma', 'scale'),
                coef0=kwargs.get('coef0', 0.0), 
                shrinking=kwargs.get('shrinking', True), 
                probability=kwargs.get('probability', False), 
                tol=kwargs.get('tol', 1e-3), 
                cache_size=kwargs.get('cache_size', 200), 
                class_weight=kwargs.get('class_weight', None), 
                verbose=kwargs.get('verbose', False), 
                max_iter=kwargs.get('max_iter', -1), 
                decision_function_shape=kwargs.get('decision_function_shape', 'ovr'),
                break_ties=kwargs.get('break_ties', False), 
                random_state=kwargs.get('random_state', seed), 
    ))

    return run_classification(
        model, "SVC",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, X_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        modeling_mode=kwargs.get('modeling_mode', 'simple')
    )
