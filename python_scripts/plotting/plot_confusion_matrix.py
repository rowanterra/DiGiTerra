#!/usr/bin/env python
# coding: utf-8

"""
Confusion Matrix plotting using scikit-learn's ConfusionMatrixDisplay.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from python_scripts.config import VIS_DIR


def plot_confusion_matrix(y_true, y_pred, model_name, pdf_pages=None, 
                         labels=None, normalize=None, display_labels=None,
                         include_values=True, xticks_rotation='horizontal',
                         values_format=None, cmap='viridis', ax=None,
                         colorbar=True, file_suffix='', label_suffix=''):
    """
    Plot confusion matrix using scikit-learn's ConfusionMatrixDisplay.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        pdf_pages: PdfPages object for saving to PDF (optional)
        labels: List of labels to index the confusion matrix
        normalize: {'true', 'pred', 'all'} or None
        display_labels: Display labels for plot
        include_values: Whether to include values in confusion matrix
        xticks_rotation: Rotation of xtick labels
        values_format: Format specification for values
        cmap: Colormap recognized by matplotlib
        ax: Axes object to plot on
        colorbar: Whether to add a colorbar
        file_suffix: Suffix for filename
        label_suffix: Suffix for plot title
        
    Returns:
        ConfusionMatrixDisplay object
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Create display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    
    # Plot
    disp.plot(include_values=include_values,
             cmap=cmap,
             xticks_rotation=xticks_rotation,
             values_format=values_format,
             ax=ax,
             colorbar=colorbar)
    
    # Add title
    title_base = f"{model_name} | Confusion Matrix"
    title_with_label = f"{title_base} {label_suffix}" if label_suffix else title_base
    if disp.ax_ is not None:
        disp.ax_.set_title(title_with_label, fontsize=14, pad=20)
    
    plt.tight_layout()
    
    # Save to file
    plot_filename = f"confusion_matrix{file_suffix}.png"
    plot_path = VIS_DIR / plot_filename
    if disp.figure_ is not None:
        disp.figure_.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Confusion matrix plot saved to {plot_path}")
        
        # Save to PDF if provided
        if pdf_pages is not None:
            pdf_pages.savefig(disp.figure_, bbox_inches='tight', facecolor='white')
    
    return disp


def plot_confusion_matrix_from_estimator(estimator, X, y, model_name, pdf_pages=None,
                                        labels=None, sample_weight=None, normalize=None,
                                        display_labels=None, include_values=True,
                                        xticks_rotation='horizontal', values_format=None,
                                        cmap='viridis', ax=None, colorbar=True,
                                        file_suffix='', label_suffix=''):
    """
    Plot confusion matrix from an estimator using scikit-learn's ConfusionMatrixDisplay.from_estimator.
    
    Args:
        estimator: Fitted classifier
        X: Input values
        y: Target values
        model_name: Name of the model
        pdf_pages: PdfPages object for saving to PDF (optional)
        labels: List of labels to index the confusion matrix
        sample_weight: Sample weights
        normalize: {'true', 'pred', 'all'} or None
        display_labels: Display labels for plot
        include_values: Whether to include values in confusion matrix
        xticks_rotation: Rotation of xtick labels
        values_format: Format specification for values
        cmap: Colormap recognized by matplotlib
        ax: Axes object to plot on
        colorbar: Whether to add a colorbar
        file_suffix: Suffix for filename
        label_suffix: Suffix for plot title
        
    Returns:
        ConfusionMatrixDisplay object
    """
    # Create display from estimator
    disp = ConfusionMatrixDisplay.from_estimator(
        estimator, X, y,
        labels=labels,
        sample_weight=sample_weight,
        normalize=normalize,
        display_labels=display_labels,
        include_values=include_values,
        xticks_rotation=xticks_rotation,
        values_format=values_format,
        cmap=cmap,
        ax=ax,
        colorbar=colorbar
    )
    
    # Add title
    title_base = f"{model_name} | Confusion Matrix"
    title_with_label = f"{title_base} {label_suffix}" if label_suffix else title_base
    if disp.ax_ is not None:
        disp.ax_.set_title(title_with_label, fontsize=14, pad=20)
    
    plt.tight_layout()
    
    # Save to file
    plot_filename = f"confusion_matrix{file_suffix}.png"
    plot_path = VIS_DIR / plot_filename
    if disp.figure_ is not None:
        disp.figure_.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Confusion matrix plot saved to {plot_path}")
        
        # Save to PDF if provided
        if pdf_pages is not None:
            pdf_pages.savefig(disp.figure_, bbox_inches='tight', facecolor='white')
    
    return disp
