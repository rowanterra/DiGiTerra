"""Feature selection utilities for DiGiTerra."""
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, chi2, RFE, SelectFromModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def apply_feature_selection(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    method: str,
    k: Optional[int] = None,
    problem_type: str = 'regression'
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[object]]:
    """
    Apply feature selection to training and test data.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        method: Selection method ('SelectKBest', 'RFE', 'SelectFromModel', or 'none')
        k: Number of features to select (for SelectKBest and RFE)
        problem_type: 'regression', 'classification', or 'clustering'
    
    Returns:
        Tuple of (X_train_selected, X_test_selected, selector_object)
    """
    if method == 'none' or method is None:
        return X_train, X_test, None
    
    selector = None
    
    if method == 'SelectKBest':
        if k is None or k <= 0:
            raise ValueError("SelectKBest requires k > 0")
        if k >= X_train.shape[1]:
            # Don't select if k >= number of features
            return X_train, X_test, None
        
        if problem_type == 'regression':
            score_func = f_regression
        elif problem_type == 'classification':
            score_func = f_classif
        else:  # clustering
            # Use f_regression as default for clustering
            score_func = f_regression
        
        selector = SelectKBest(score_func=score_func, k=min(k, X_train.shape[1]))
        X_train_selected = pd.DataFrame(
            selector.fit_transform(X_train, y_train),
            index=X_train.index,
            columns=X_train.columns[selector.get_support()]
        )
        X_test_selected = pd.DataFrame(
            selector.transform(X_test),
            index=X_test.index,
            columns=X_train.columns[selector.get_support()]
        )
    
    elif method == 'RFE':
        if k is None or k <= 0:
            raise ValueError("RFE requires k > 0")
        if k >= X_train.shape[1]:
            return X_train, X_test, None
        
        # Use a simple estimator for RFE
        if problem_type == 'regression':
            estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        else:  # classification
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
        selector = RFE(estimator=estimator, n_features_to_select=min(k, X_train.shape[1]))
        X_train_selected = pd.DataFrame(
            selector.fit_transform(X_train, y_train),
            index=X_train.index,
            columns=X_train.columns[selector.get_support()]
        )
        X_test_selected = pd.DataFrame(
            selector.transform(X_test),
            index=X_test.index,
            columns=X_train.columns[selector.get_support()]
        )
    
    elif method == 'SelectFromModel':
        # Use model-based selection
        if problem_type == 'regression':
            estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        else:  # classification
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
        selector = SelectFromModel(estimator=estimator)
        X_train_selected = pd.DataFrame(
            selector.fit_transform(X_train, y_train),
            index=X_train.index,
            columns=X_train.columns[selector.get_support()]
        )
        X_test_selected = pd.DataFrame(
            selector.transform(X_test),
            index=X_test.index,
            columns=X_train.columns[selector.get_support()]
        )
    
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    
    return X_train_selected, X_test_selected, selector
