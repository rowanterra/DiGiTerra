from sklearn.ensemble import AdaBoostClassifier
from python_scripts.preprocessing.run_classification_pipeline import run_classification

def train_adaboost_classifier(train_data, target_variables, use_stratified_split,
                X, y, stratifyColumn, units,
                X_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols, testSize,
                feature_selection_method='none', feature_selection_k=None,
                outlier_method='none', outlier_action='remove',
                hyperparameter_search='none', search_cv_folds=5, search_n_iter=50, **kwargs):

    model = AdaBoostClassifier(estimator=kwargs.get('estimator', None),
                               n_estimators=kwargs.get('n_estimators', 50),
                               learning_rate=kwargs.get('learning_rate', 1.0),
                               random_state=kwargs.get('random_state', seed),
                               )

    return run_classification(
        model, "AdaBoostClassifier",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, X_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols, testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        modeling_mode=kwargs.get('modeling_mode', 'simple')
    )
