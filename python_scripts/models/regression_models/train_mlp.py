from sklearn.neural_network import MLPRegressor
from python_scripts.preprocessing.run_regression_pipeline import run_regression

def train_mlp(model_type, train_data, target_variables, use_stratified_split,
                X, y, stratifyColumn, units,
                X_scaler_type, y_scaler_type,
                seed, quantileBinDict, sigfig, useTransformer, categorical_cols,testSize,
                feature_selection_method='none', feature_selection_k=None,
                outlier_method='none', outlier_action='remove',
                hyperparameter_search='none', search_cv_folds=5, search_n_iter=50,
                progress_tracker=None,**kwargs):

    model = MLPRegressor(hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (100,)), 
                        activation=kwargs.get('activation', 'relu'), 
                        solver=kwargs.get('solver', 'adam'), 
                        alpha=kwargs.get('alpha', 0.0001), 
                        learning_rate=kwargs.get('learning_rate', 'constant'), 
                        max_iter=kwargs.get('max_iter', 10000),
                        batch_size=kwargs.get('batch_size', 'auto'),
                        beta_1=kwargs.get('beta_1', .9),
                        beta_2=kwargs.get('beta_2', .999),
                        early_stopping=kwargs.get('early_stopping', False),
                        epsilon=kwargs.get('epsilon', 1e-8),
                        learning_rate_init=kwargs.get('learning_rate_init', .001),
                        max_fun=kwargs.get('max_fun', 15000),
                        momentum=kwargs.get('momentum', .9),
                        n_iter_no_change=kwargs.get('n_iter_no_change', 10),
                        nesterovs_momentum=kwargs.get('nesterovs_momentum', True),
                        power_t=kwargs.get('power_t', .5),
                        random_state=kwargs.get('random_state', None),
                        shuffle=kwargs.get('shuffle', True),
                        tol=kwargs.get('tol',.0001),
                        validation_fraction=kwargs.get('validation_fraction', .1),
                        verbose=kwargs.get('verbose', False),
                        warm_start=kwargs.get('warm_start',False),
                        )

    return run_regression(
        model, "MLP",
        train_data, target_variables, use_stratified_split,
        X, y, stratifyColumn,
        units, X_scaler_type, y_scaler_type,
        seed, sigfig, quantileBinDict, useTransformer, categorical_cols,testSize,
        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
        outlier_method=outlier_method, outlier_action=outlier_action,
        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
        progress_tracker=progress_tracker, modeling_mode=kwargs.get('modeling_mode', 'simple')
    )
