#!/usr/bin/env python
# coding: utf-8

"""
ROC Curve plotting using scikit-learn's RocCurveDisplay.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from sklearn.metrics import RocCurveDisplay, roc_curve, roc_auc_score
from python_scripts.config import VIS_DIR


def plot_roc_curve(y_true, y_score, model_name, pdf_pages=None,
                  sample_weight=None, drop_intermediate=True, pos_label=None,
                  name=None, ax=None, plot_chance_level=False, despine=False,
                  file_suffix='', label_suffix='', **kwargs):
    """
    Plot ROC curve using scikit-learn's RocCurveDisplay.from_predictions.
    
    Args:
        y_true: True binary labels
        y_score: Target scores (probability estimates or decision function output)
        model_name: Name of the model
        pdf_pages: PdfPages object for saving to PDF (optional)
        sample_weight: Sample weights
        drop_intermediate: Whether to drop intermediate thresholds
        pos_label: The class considered as the positive class
        name: Name of ROC curve for legend labeling
        ax: Axes object to plot on
        plot_chance_level: Whether to plot the chance level
        despine: Whether to remove the top and right spines
        file_suffix: Suffix for filename
        label_suffix: Suffix for plot title
        **kwargs: Additional keyword arguments passed to matplotlib plot function
        
    Returns:
        RocCurveDisplay object
    """
    # Compute ROC AUC
    try:
        roc_auc = roc_auc_score(y_true, y_score, sample_weight=sample_weight, pos_label=pos_label)
    except Exception:
        roc_auc = None
    
    # Create display from predictions
    disp = RocCurveDisplay.from_predictions(
        y_true, y_score,
        sample_weight=sample_weight,
        drop_intermediate=drop_intermediate,
        pos_label=pos_label,
        name=name or model_name,
        ax=ax,
        plot_chance_level=plot_chance_level,
        despine=despine,
        **kwargs
    )
    
    # Add title
    title_base = f"{model_name} | ROC Curve"
    if roc_auc is not None:
        title_base += f" (AUC = {roc_auc:.3f})"
    title_with_label = f"{title_base} {label_suffix}" if label_suffix else title_base
    if disp.ax_ is not None:
        disp.ax_.set_title(title_with_label, fontsize=14, pad=20)
    
    plt.tight_layout()
    
    # Save to file
    plot_filename = f"roc_curve{file_suffix}.png"
    plot_path = VIS_DIR / plot_filename
    if disp.figure_ is not None:
        disp.figure_.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"ROC curve plot saved to {plot_path}")
        
        # Save to PDF if provided
        if pdf_pages is not None:
            pdf_pages.savefig(disp.figure_, bbox_inches='tight', facecolor='white')
    
    return disp


def plot_roc_curve_from_estimator(estimator, X, y, model_name, pdf_pages=None,
                                  sample_weight=None, drop_intermediate=True,
                                  response_method='auto', pos_label=None,
                                  name=None, ax=None, plot_chance_level=False,
                                  despine=False, file_suffix='', label_suffix='',
                                  **kwargs):
    """
    Plot ROC curve from an estimator using scikit-learn's RocCurveDisplay.from_estimator.
    
    Args:
        estimator: Fitted classifier
        X: Input values
        y: Target values
        model_name: Name of the model
        pdf_pages: PdfPages object for saving to PDF (optional)
        sample_weight: Sample weights
        drop_intermediate: Whether to drop intermediate thresholds
        response_method: {'predict_proba', 'decision_function', 'auto'}
        pos_label: The class considered as the positive class
        name: Name of ROC curve for legend labeling
        ax: Axes object to plot on
        plot_chance_level: Whether to plot the chance level
        despine: Whether to remove the top and right spines
        file_suffix: Suffix for filename
        label_suffix: Suffix for plot title
        **kwargs: Additional keyword arguments passed to matplotlib plot function
        
    Returns:
        RocCurveDisplay object
    """
    # Create display from estimator
    disp = RocCurveDisplay.from_estimator(
        estimator, X, y,
        sample_weight=sample_weight,
        drop_intermediate=drop_intermediate,
        response_method=response_method,
        pos_label=pos_label,
        name=name or model_name,
        ax=ax,
        plot_chance_level=plot_chance_level,
        despine=despine,
        **kwargs
    )
    
    # Add title
    title_base = f"{model_name} | ROC Curve"
    if disp.roc_auc is not None:
        title_base += f" (AUC = {disp.roc_auc:.3f})"
    title_with_label = f"{title_base} {label_suffix}" if label_suffix else title_base
    if disp.ax_ is not None:
        disp.ax_.set_title(title_with_label, fontsize=14, pad=20)
    
    plt.tight_layout()
    
    # Save to file
    plot_filename = f"roc_curve{file_suffix}.png"
    plot_path = VIS_DIR / plot_filename
    if disp.figure_ is not None:
        disp.figure_.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"ROC curve plot saved to {plot_path}")
        
        # Save to PDF if provided
        if pdf_pages is not None:
            pdf_pages.savefig(disp.figure_, bbox_inches='tight', facecolor='white')
    
    return disp
