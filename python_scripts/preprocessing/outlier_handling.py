"""Outlier detection and handling utilities for DiGiTerra."""
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def apply_outlier_handling(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: Optional[pd.Series] = None,
    method: str = 'none',
    action: str = 'remove'
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Apply outlier detection and handling to training and test data.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target (optional, not used for detection)
        method: Detection method ('IQR', 'IsolationForest', 'LocalOutlierFactor', 'ZScore', or 'none')
        action: Action to take ('remove' or 'cap')
    
    Returns:
        Tuple of (X_train_processed, X_test_processed, outlier_mask)
        outlier_mask is True for inliers, False for outliers
    """
    if method == 'none' or method is None:
        return X_train, X_test, np.ones(len(X_train), dtype=bool)
    
    outlier_mask = np.ones(len(X_train), dtype=bool)
    
    if method == 'IQR':
        # Interquartile Range method
        Q1 = X_train.quantile(0.25)
        Q3 = X_train.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Mark outliers (any feature outside bounds)
        outlier_mask = ((X_train >= lower_bound) & (X_train <= upper_bound)).all(axis=1)
    
    elif method == 'ZScore':
        # Z-Score method (3 sigma rule)
        mean = X_train.mean()
        std = X_train.std()
        z_scores = np.abs((X_train - mean) / std)
        # Mark as outlier if any feature has |z| > 3
        outlier_mask = (z_scores < 3).all(axis=1)
    
    elif method == 'IsolationForest':
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = iso_forest.fit_predict(X_train)
        outlier_mask = outlier_labels == 1  # 1 = inlier, -1 = outlier
    
    elif method == 'LocalOutlierFactor':
        lof = LocalOutlierFactor(contamination=0.1, n_neighbors=20)
        outlier_labels = lof.fit_predict(X_train)
        outlier_mask = outlier_labels == 1  # 1 = inlier, -1 = outlier
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    # Apply action
    if action == 'remove':
        X_train_processed = X_train[outlier_mask].copy()
        # For test set, we don't remove outliers (they're new data)
        # But we can cap them if needed
        X_test_processed = X_test.copy()
    elif action == 'cap':
        # Cap outliers at bounds
        if method == 'IQR':
            Q1 = X_train.quantile(0.25)
            Q3 = X_train.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        elif method == 'ZScore':
            mean = X_train.mean()
            std = X_train.std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
        else:
            # For IsolationForest and LOF, use percentiles
            lower_bound = X_train.quantile(0.01)
            upper_bound = X_train.quantile(0.99)
        
        X_train_processed = X_train.copy()
        X_train_processed = X_train_processed.clip(lower=lower_bound, upper=upper_bound, axis=1)
        
        X_test_processed = X_test.copy()
        X_test_processed = X_test_processed.clip(lower=lower_bound, upper=upper_bound, axis=1)
    else:
        raise ValueError(f"Unknown action: {action}")
    
    return X_train_processed, X_test_processed, outlier_mask
