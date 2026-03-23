#!/usr/bin/env python
# coding: utf-8

"""
ROC Curve plotting using scikit-learn's RocCurveDisplay.
"""

import logging
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import python_scripts.plotting.plot_style  # noqa: F401
from sklearn.metrics import RocCurveDisplay, roc_curve, roc_auc_score
from python_scripts.config import VIS_DIR

logger = logging.getLogger(__name__)


def _finite_metric(x):
    """True if x is a finite scalar (not None, not NaN, not inf)."""
    if x is None:
        return False
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _finalize_roc_figure(disp, default_legend_name: str):
    """Strip invalid AUC from legend labels and give axes room so y-label is not clipped."""
    if disp.ax_ is None:
        return
    leg = disp.ax_.get_legend()
    if leg is not None:
        for text in leg.get_texts():
            s = text.get_text()
            if "nan" in s.lower():
                cleaned = re.sub(
                    r"\s*\([^)]*AUC\s*=\s*nan[^)]*\)",
                    "",
                    s,
                    flags=re.IGNORECASE,
                ).strip()
                text.set_text(cleaned if cleaned else default_legend_name)
    if disp.figure_ is not None:
        disp.figure_.set_size_inches(7.0, 5.75)
        disp.figure_.tight_layout()
        disp.figure_.subplots_adjust(left=0.16, bottom=0.14)


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
    
    # Add title (NaN is not None — only show AUC when finite)
    title_base = f"{model_name} | ROC Curve"
    if _finite_metric(roc_auc):
        title_base += f" (AUC = {float(roc_auc):.3f})"
    elif roc_auc is not None:
        logger.debug("ROC AUC non-finite or invalid; omitting from title")
    title_with_label = f"{title_base} {label_suffix}" if label_suffix else title_base
    if disp.ax_ is not None:
        disp.ax_.set_title(title_with_label, fontsize=14, pad=20)
    
    _finalize_roc_figure(disp, name or model_name)
    
    # Save to file
    plot_filename = f"roc_curve{file_suffix}.png"
    plot_path = VIS_DIR / plot_filename
    if disp.figure_ is not None:
        disp.figure_.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        logger.debug("ROC curve plot saved to %s", plot_path)
        
        # Save to PDF if provided
        if pdf_pages is not None:
            pdf_pages.savefig(disp.figure_, bbox_inches='tight', facecolor='white')
        # Avoid leaking figures into later plots (e.g. SHAP) when we own the figure
        if ax is None:
            plt.close(disp.figure_)
    
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
    
    # Add title (disp.roc_auc can be NaN — np.nan is not None)
    title_base = f"{model_name} | ROC Curve"
    if _finite_metric(getattr(disp, "roc_auc", None)):
        title_base += f" (AUC = {float(disp.roc_auc):.3f})"
    elif getattr(disp, "roc_auc", None) is not None:
        logger.debug("ROC AUC from estimator is non-finite; omitting from title")
    title_with_label = f"{title_base} {label_suffix}" if label_suffix else title_base
    if disp.ax_ is not None:
        disp.ax_.set_title(title_with_label, fontsize=14, pad=20)
    
    _finalize_roc_figure(disp, name or model_name)
    
    # Save to file
    plot_filename = f"roc_curve{file_suffix}.png"
    plot_path = VIS_DIR / plot_filename
    if disp.figure_ is not None:
        disp.figure_.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        logger.debug("ROC curve plot saved to %s", plot_path)
        
        # Save to PDF if provided
        if pdf_pages is not None:
            pdf_pages.savefig(disp.figure_, bbox_inches='tight', facecolor='white')
        if ax is None:
            plt.close(disp.figure_)
    
    return disp
