"""
Comprehensive metrics calculation for classification, regression, and clustering.
This module ensures all relevant scikit-learn statistics are calculated and available for export.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)
from sklearn.metrics import (
    # Classification
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    # Regression
    mean_squared_error, mean_absolute_error, r2_score,
    median_absolute_error, max_error, explained_variance_score,
    mean_absolute_percentage_error, mean_pinball_loss,
    # Clustering
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
)
from sklearn.preprocessing import label_binarize


def calculate_classification_metrics(y_true, y_pred, y_score=None, classes=None, model=None, X_test=None):
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_score: Prediction scores (probabilities or decision function output)
        classes: Class labels
        model: Trained model (used to get scores if y_score not provided)
        X_test: Test features (used to get scores if y_score not provided)
        
    Returns:
        Dictionary containing all classification metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    if classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
    
    per_class_metrics = {}
    for cls in classes:
        per_class_metrics[f'precision_class_{cls}'] = precision_score(
            y_true, y_pred, labels=[cls], average='micro', zero_division=0
        )
        per_class_metrics[f'recall_class_{cls}'] = recall_score(
            y_true, y_pred, labels=[cls], average='micro', zero_division=0
        )
        per_class_metrics[f'f1_class_{cls}'] = f1_score(
            y_true, y_pred, labels=[cls], average='micro', zero_division=0
        )
    metrics['per_class'] = per_class_metrics
    
    # ROC AUC and Average Precision (if scores available)
    if y_score is None and model is not None and X_test is not None:
        try:
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)
            elif hasattr(model, "decision_function"):
                dfc = model.decision_function(X_test)
                if getattr(dfc, "ndim", 1) == 1:
                    y_score = np.c_[-dfc, dfc]
                else:
                    y_score = dfc
        except Exception:
            y_score = None
    
    if y_score is not None:
        try:
            # Binarize for multi-class
            y_bin = label_binarize(y_true, classes=classes)
            n_classes = len(classes)
            
            if n_classes == 2:
                # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_score[:, 1] if y_score.shape[1] > 1 else y_score[:, 0])
                metrics['average_precision'] = average_precision_score(
                    y_true, y_score[:, 1] if y_score.shape[1] > 1 else y_score[:, 0]
                )
            else:
                # Multi-class
                metrics['roc_auc_macro'] = roc_auc_score(y_bin, y_score, average='macro', multi_class='ovr')
                metrics['roc_auc_micro'] = roc_auc_score(y_bin, y_score, average='micro', multi_class='ovr')
                metrics['roc_auc_weighted'] = roc_auc_score(y_bin, y_score, average='weighted', multi_class='ovr')
                metrics['average_precision_macro'] = average_precision_score(y_bin, y_score, average='macro')
                metrics['average_precision_micro'] = average_precision_score(y_bin, y_score, average='micro')
                metrics['average_precision_weighted'] = average_precision_score(y_bin, y_score, average='weighted')
                
                # Per-class ROC AUC and AP
                per_class_roc_auc = {}
                per_class_ap = {}
                for i, cls in enumerate(classes):
                    try:
                        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                        from sklearn.metrics import auc
                        per_class_roc_auc[f'roc_auc_class_{cls}'] = auc(fpr, tpr)
                        per_class_ap[f'ap_class_{cls}'] = average_precision_score(y_bin[:, i], y_score[:, i])
                    except Exception:
                        per_class_roc_auc[f'roc_auc_class_{cls}'] = np.nan
                        per_class_ap[f'ap_class_{cls}'] = np.nan
                metrics['per_class_roc_auc'] = per_class_roc_auc
                metrics['per_class_ap'] = per_class_ap
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC/AP metrics: {e}")
    
    return metrics


def calculate_regression_metrics(y_true, y_pred, target_names=None, sigfig=3):
    """
    Calculate comprehensive regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        target_names: Names of target variables
        sigfig: Significant figures for rounding
        
    Returns:
        DataFrame with all regression metrics per target and overall
    """
    if isinstance(y_true, pd.DataFrame):
        yt = y_true.values
        names = list(y_true.columns)
    else:
        yt = np.asarray(y_true)
        names = target_names or ["y"]
    
    yp = np.asarray(y_pred)
    if yp.ndim == 1:
        yp = yp.reshape(-1, 1)
    if yt.ndim == 1:
        yt = yt.reshape(-1, 1)
    
    n_targets = yt.shape[1]
    metrics_list = []
    
    for i in range(n_targets):
        yt_i = yt[:, i]
        yp_i = yp[:, i]
        
        metrics_dict = {
            "Target": names[i] if i < len(names) else f"Target_{i}",
            "R²": r2_score(yt_i, yp_i),
            "Explained Variance": explained_variance_score(yt_i, yp_i),
            "MSE": mean_squared_error(yt_i, yp_i),
            "RMSE": np.sqrt(mean_squared_error(yt_i, yp_i)),
            "MAE": mean_absolute_error(yt_i, yp_i),
            "Median AE": median_absolute_error(yt_i, yp_i),
            "Max Error": max_error(yt_i, yp_i),
        }
        
        # Try to calculate MAPE (may fail if y_true contains zeros)
        try:
            metrics_dict["MAPE"] = mean_absolute_percentage_error(yt_i, yp_i)
        except Exception:
            metrics_dict["MAPE"] = np.nan
        
        metrics_list.append(metrics_dict)
    
    df = pd.DataFrame(metrics_list)
    
    # Calculate overall metrics (mean across targets)
    overall_dict = {
        "Target": "Overall",
        "R²": np.mean([m["R²"] for m in metrics_list]),
        "Explained Variance": np.mean([m["Explained Variance"] for m in metrics_list]),
        "MSE": np.mean([m["MSE"] for m in metrics_list]),
        "RMSE": np.mean([m["RMSE"] for m in metrics_list]),
        "MAE": np.mean([m["MAE"] for m in metrics_list]),
        "Median AE": np.mean([m["Median AE"] for m in metrics_list]),
        "Max Error": np.mean([m["Max Error"] for m in metrics_list]),
    }
    
    try:
        overall_dict["MAPE"] = np.nanmean([m["MAPE"] for m in metrics_list])
    except Exception:
        overall_dict["MAPE"] = np.nan
    
    overall_df = pd.DataFrame([overall_dict])
    result = pd.concat([df, overall_df], ignore_index=True)
    
    return result.round(sigfig)


def calculate_clustering_metrics(X, labels):
    """
    Calculate comprehensive clustering metrics.
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        
    Returns:
        Dictionary with clustering metrics
    """
    metrics = {}
    
    try:
        metrics['silhouette_score'] = silhouette_score(X, labels)
    except Exception:
        metrics['silhouette_score'] = np.nan
    
    try:
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
    except Exception:
        metrics['calinski_harabasz_score'] = np.nan
    
    try:
        metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
    except Exception:
        metrics['davies_bouldin_score'] = np.nan
    
    # Additional statistics
    unique_labels = np.unique(labels)
    metrics['n_clusters'] = len(unique_labels)
    metrics['n_samples'] = len(labels)
    
    # Cluster sizes
    cluster_sizes = {}
    for label in unique_labels:
        cluster_sizes[f'cluster_{label}_size'] = np.sum(labels == label)
    metrics['cluster_sizes'] = cluster_sizes
    
    return metrics
