"""Hyperparameter search utilities for DiGiTerra."""
import logging
from typing import Dict, Any, Optional
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
import numpy as np
import time
import threading

logger = logging.getLogger(__name__)


def get_param_grid(model_name: str, problem_type: str = 'regression') -> Dict[str, list]:
    """
    Get parameter grid for a given model.
    
    Args:
        model_name: Name of the model
        problem_type: 'regression', 'classification', or 'clustering'
    
    Returns:
        Dictionary of parameter grids
    """
    grids = {}
    
    if problem_type == 'regression':
        if model_name in ['Ridge', 'ridge', 'RidgeRegressor']:
            grids = {
                'alpha': [0.1, 1.0, 10.0, 100.0],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
            }
        elif model_name in ['Lasso', 'lasso', 'LassoRegressor']:
            grids = {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'max_iter': [1000, 2000, 5000]
            }
        elif model_name in ['RF', 'RandomForest', 'random_forest', 'RandomForestRegressor']:
            grids = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_name in ['ExtraTrees', 'ExtraTreesRegressor']:
            grids = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        elif model_name in ['MLP', 'MLPRegressor']:
            grids = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        elif model_name in ['gradient_boosting', 'GradientBoosting', 'GradientBoostingRegressor']:
            grids = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        elif model_name in ['SVM', 'SVR', 'SVRRegressor', 'SVR']:
            grids = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear', 'poly']
            }
        elif model_name in ['Linear', 'LinearRegression']:
            grids = {
                'fit_intercept': [True, False],
                'positive': [True, False]
            }
        elif model_name in ['BayesianRidge', 'BayesianRidgeRegressor']:
            grids = {
                'alpha_1': [1e-6, 1e-5, 1e-4],
                'alpha_2': [1e-6, 1e-5, 1e-4],
                'lambda_1': [1e-6, 1e-5, 1e-4],
                'lambda_2': [1e-6, 1e-5, 1e-4]
            }
        elif model_name in ['ARDRegression', 'ARD']:
            grids = {
                'alpha_1': [1e-6, 1e-5, 1e-4],
                'alpha_2': [1e-6, 1e-5, 1e-4],
                'lambda_1': [1e-6, 1e-5, 1e-4],
                'lambda_2': [1e-6, 1e-5, 1e-4]
            }
        elif model_name in ['ElasticNet', 'ElasticNetRegressor']:
            grids = {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.5, 0.7, 0.9],
                'max_iter': [1000, 2000, 5000]
            }
        elif model_name in ['K-Nearest', 'KNN', 'KNeighborsRegressor']:
            grids = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]  # Manhattan or Euclidean
            }
        elif model_name in ['Perceptron']:
            # Note: Perceptron is actually a classifier, but if used for regression, use SGD-like params
            grids = {
                'penalty': ['l2', 'l1', 'elasticnet', None],
                'alpha': [0.0001, 0.001, 0.01],
                'max_iter': [1000, 2000, 5000]
            }
        elif model_name in ['LogisticRegression', 'Logistic']:
            grids = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2', 'elasticnet', None],
                'solver': ['lbfgs', 'liblinear', 'saga'],
                'max_iter': [100, 200, 500]
            }
    
    elif problem_type == 'classification':
        if model_name in ['Logistic_classifier', 'LogisticClassifier', 'LogisticRegression']:
            grids = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'lbfgs']
            }
        elif model_name in ['RF_classifier', 'RandomForest', 'RandomForestClassifier']:
            grids = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10]
            }
        elif model_name in ['ExtraTrees_classifier', 'ExtraTreesClassifier']:
            grids = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        elif model_name in ['SVC_classifier', 'SVC', 'SVCClassifier']:
            grids = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'linear']
            }
        elif model_name in ['MLP_classifier', 'MLPClassifier']:
            grids = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        elif model_name in ['GaussianNB_classifier', 'GaussianNB']:
            grids = {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            }
        elif model_name in ['SGD_classifier', 'SGDClassifier']:
            grids = {
                'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge'],
                'penalty': ['l2', 'l1', 'elasticnet'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']
            }
    
    # Default grid if model not found - use safe defaults that work for most models
    if not grids:
        # Don't use 'alpha' as default for regression - it doesn't work for tree-based models
        # Use n_estimators for ensemble models, or C for linear models
        model_name_lower = model_name.lower() if model_name else ''
        if problem_type == 'regression':
            # Try to infer from model name if possible - check for tree-based models first
            if any(keyword in model_name_lower for keyword in ['tree', 'forest', 'boosting', 'extra', 'random']):
                grids = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10, None]}
            elif any(keyword in model_name_lower for keyword in ['svm', 'svr', 'svc']):
                grids = {'C': [0.1, 1.0, 10.0, 100.0], 'gamma': ['scale', 'auto']}
            elif any(keyword in model_name_lower for keyword in ['knn', 'neighbor', 'nearest']):
                grids = {'n_neighbors': [3, 5, 7, 9, 11]}
            else:
                # Default for linear models
                grids = {'alpha': [0.1, 1.0, 10.0]}  # Safe for linear models
        else:
            # Classification defaults
            if any(keyword in model_name_lower for keyword in ['tree', 'forest', 'boosting', 'extra', 'random']):
                grids = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10, None]}
            else:
                grids = {'C': [0.1, 1.0, 10.0]}
    
    return grids


def apply_hyperparameter_search(
    model,
    X_train,
    y_train,
    search_type: str,
    param_grid: Optional[Dict[str, list]] = None,
    cv_folds: int = 5,
    n_iter: int = 50,
    scoring: Optional[str] = None,
    problem_type: str = 'regression',
    model_name: Optional[str] = None,
    progress_tracker=None
):
    """
    Apply hyperparameter search to a model.
    
    Args:
        model: Base model instance
        X_train: Training features
        y_train: Training target (can be array or Series)
        search_type: 'grid', 'randomized', or 'none'
        param_grid: Parameter grid (if None, will be generated)
        cv_folds: Number of CV folds
        n_iter: Number of iterations for randomized search
        scoring: Scoring metric (None for default)
        problem_type: 'regression', 'classification', or 'clustering'
    
    Returns:
        Fitted model (either GridSearchCV/RandomizedSearchCV wrapper or original model)
    """
    if search_type == 'none' or search_type is None:
        return model
    
    # Convert y_train to array if needed
    if hasattr(y_train, 'values'):
        y_train_array = y_train.values
    else:
        y_train_array = y_train
    
    # Check if model is wrapped in MultiOutputRegressor/MultiOutputClassifier
    is_multioutput = isinstance(model, (MultiOutputRegressor, MultiOutputClassifier))
    inner_model_name = None
    if is_multioutput:
        # Get the inner estimator class name
        inner_estimator = model.estimator
        inner_model_name = inner_estimator.__class__.__name__
    
    if param_grid is None:
        # Use provided model_name or infer from model class
        if model_name is None:
            if is_multioutput:
                model_name = inner_model_name
            else:
                model_name = model.__class__.__name__
        param_grid = get_param_grid(model_name, problem_type)
    
    # If model is wrapped in MultiOutputRegressor/MultiOutputClassifier, 
    # prefix all parameter names with 'estimator__' (unless already prefixed)
    if is_multioutput and param_grid:
        # Check if parameters are already prefixed
        needs_prefix = not any(k.startswith('estimator__') for k in param_grid.keys())
        if needs_prefix:
            param_grid = {f'estimator__{k}': v for k, v in param_grid.items()}
    
    if not param_grid:
        # No grid available, return original model
        final_model_name = model_name if model_name else (inner_model_name if is_multioutput else model.__class__.__name__)
        logger.warning(f"No parameter grid available for {final_model_name}. Skipping hyperparameter search.")
        return model
    
    # Set default scoring
    if scoring is None:
        if problem_type == 'regression':
            scoring = 'neg_mean_squared_error'
        elif problem_type == 'classification':
            scoring = 'accuracy'
        else:
            scoring = 'neg_mean_squared_error'
    
    if search_type == 'grid':
        search_model = GridSearchCV(
            model,
            param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
    elif search_type == 'randomized':
        search_model = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=n_iter,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
    elif search_type == 'bayesian':
        # Bayesian optimization requires scikit-optimize
        try:
            from skopt import BayesSearchCV
            search_model = BayesSearchCV(
                model,
                param_grid,
                n_iter=n_iter,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
        except ImportError:
            logger.warning("scikit-optimize not installed. Falling back to randomized search.")
            search_model = RandomizedSearchCV(
                model,
                param_grid,
                n_iter=n_iter,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
    else:
        raise ValueError(f"Unknown search type: {search_type}")
    
    total_combinations = len(param_grid)
    if search_type == 'grid':
        # Calculate total combinations
        from itertools import product
        param_values = [v if isinstance(v, list) else [v] for v in param_grid.values()]
        total_combinations = len(list(product(*param_values)))
    
    logger.info(f"Starting {search_type} hyperparameter search with {total_combinations} parameter combinations...")
    
    # Update progress during hyperparameter search
    search_complete = None
    if progress_tracker:
        import time
        import threading
        
        start_time = time.time()
        search_complete = threading.Event()  # Event to signal when search is done
        progress_tracker.update_stage('hyperparameter_search', 'running', 5, 
                                    f'Starting {search_type} search ({total_combinations} combinations, {cv_folds} CV folds)...')
        
        # Use a thread to periodically update progress based on elapsed time
        # This is an approximation since sklearn doesn't provide callbacks
        def update_progress_estimate():
            while not search_complete.is_set():
                elapsed = time.time() - start_time
                # Estimate: assume each combination takes roughly the same time
                # This is a rough approximation
                if total_combinations > 0:
                    # Estimate 10% overhead, then distribute progress
                    estimated_total_time = total_combinations * cv_folds * 0.5  # Rough estimate: 0.5s per CV fold
                    if estimated_total_time > 0:
                        progress = min(90, 5 + (elapsed / estimated_total_time) * 85)
                        progress_tracker.update_stage('hyperparameter_search', 'running', progress,
                                                    f'Searching... ({int(elapsed)}s elapsed, ~{total_combinations} combinations)')
                # Wait with timeout so we can check the event periodically
                search_complete.wait(timeout=2)  # Check every 2 seconds
        
        progress_thread = threading.Thread(target=update_progress_estimate, daemon=True)
        progress_thread.start()
        
    search_model.fit(X_train, y_train_array)
    
    # Signal that search is complete so the progress thread exits
    if search_complete:
        search_complete.set()
    
    if progress_tracker:
        progress_tracker.update_stage('hyperparameter_search', 'running', 95, 'Finalizing best parameters...')
    
    logger.info(f"Best parameters found: {search_model.best_params_}")
    logger.info(f"Best score: {search_model.best_score_}")
    return search_model


def _estimate_param_combinations(model_name: str, n_iter: int, search_type: str) -> int:
    """Estimate number of parameter combinations for progress tracking."""
    if search_type == 'grid':
        # This is a rough estimate - actual count depends on param_grid
        return 50  # Default estimate
    elif search_type in ['randomized', 'bayesian']:
        return n_iter
    return 50
