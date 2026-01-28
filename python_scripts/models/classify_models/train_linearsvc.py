from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from python_scripts.preprocessing.run_classification_pipeline import run_classification

def train_linearsvc(train_data, target_variables, use_stratified_split,
                X, y, stratifyColumn, units,
                X_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols, testSize,
                feature_selection_method='none', feature_selection_k=None,
                outlier_method='none', outlier_action='remove',
                hyperparameter_search='none', search_cv_folds=5, search_n_iter=50, **kwargs):

    model = MultiOutputClassifier(LinearSVC(penalty=kwargs.get('penalty', 'l2'),
                                           loss=kwargs.get('loss', 'squared_hinge'),
                                           dual=kwargs.get('dual', True),
                                           tol=kwargs.get('tol', 1e-4),
                                           C=kwargs.get('C', 1.0),
                                           multi_class=kwargs.get('multi_class', 'ovr'),
                                           fit_intercept=kwargs.get('fit_intercept', True),
                                           intercept_scaling=kwargs.get('intercept_scaling', 1.0),
                                           class_weight=kwargs.get('class_weight', None),
                                           verbose=kwargs.get('verbose', 0),
                                           random_state=kwargs.get('random_state', seed),
                                           max_iter=kwargs.get('max_iter', 1000),
                                           ))

    return run_classification(
        model, "LinearSVC",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, X_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        modeling_mode=kwargs.get('modeling_mode', 'simple')
    )
