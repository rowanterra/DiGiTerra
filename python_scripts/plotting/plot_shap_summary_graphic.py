#!/usr/bin/env python
# coding: utf-8

import logging
import os
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from python_scripts.config import VIS_DIR
import shap

logger = logging.getLogger(__name__)


def plot_shap_summary(model, X_train, feature_names, model_name, pdf_pages, file_suffix='', label_suffix=''):
    """
    Generate SHAP summary plot with optimized background sampling.
    
    Args:
        model: Trained model with predict method
        X_train: Training data (numpy array or DataFrame)
        feature_names: List of feature names
        model_name: Name of the model
        pdf_pages: PdfPages object for saving to PDF
    """
    # Convert to numpy array if needed
    if hasattr(X_train, 'values'):
        X_train_array = X_train.values
    else:
        X_train_array = X_train
    
    # Sample background data for faster computation (max 100 samples)
    # This reduces computation time significantly
    n_samples = min(100, len(X_train_array))
    if len(X_train_array) > n_samples:
        # Use k-means to summarize background data
        try:
            background_data = shap.kmeans(X_train_array, n_samples)
            logger.debug(f"Using {n_samples} background samples (k-means summary) for SHAP computation")
        except Exception:
            # Fallback to random sampling if k-means fails
            indices = np.random.choice(len(X_train_array), n_samples, replace=False)
            background_data = X_train_array[indices]
            logger.debug(f"Using {n_samples} background samples (random sample) for SHAP computation")
    else:
        background_data = X_train_array
        logger.debug(f"Using all {len(X_train_array)} samples for SHAP computation")
    
    # Create explainer with sampled background
    explainer = shap.KernelExplainer(model.predict, background_data)
    
    # Compute SHAP values for all training samples (or a subset if too many)
    # Limit to 200 samples for SHAP value computation to avoid long computation times
    max_shap_samples = min(200, len(X_train_array))
    if len(X_train_array) > max_shap_samples:
        indices = np.random.choice(len(X_train_array), max_shap_samples, replace=False)
        X_shap = X_train_array[indices]
        logger.debug(f"Computing SHAP values for {max_shap_samples} samples (random subset)")
    else:
        X_shap = X_train_array
    
    shap_values = explainer.shap_values(X_shap)
    
    # Validate SHAP values
    if shap_values is None:
        raise ValueError("SHAP values computation returned None")
    
    # Convert to numpy array if needed
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)
    
    # Ensure shap_values and X_shap have compatible shapes
    if shap_values.shape[0] != X_shap.shape[0]:
        raise ValueError(f"SHAP values shape {shap_values.shape} doesn't match X_shap shape {X_shap.shape}")
    
    # Create the summary plot
    # shap.summary_plot creates its own figure internally, so don't create one beforehand
    plot_created = False
    fig = None
    
    # Track figures that exist before we start, so we can clean up only what we create
    initial_fig_nums = set(plt.get_fignums())
    
    try:
        # shap.summary_plot creates its own figure and axes
        shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False)
        plot_created = True
        # Get the current figure (created by shap.summary_plot)
        fig = plt.gcf()
    except Exception as e:
        logger.warning(f"shap.summary_plot failed: {e}. Trying alternative plot method...")
        # Close any figures created by shap.summary_plot before trying fallback
        current_fig_nums = set(plt.get_fignums())
        for fig_num in current_fig_nums - initial_fig_nums:
            plt.close(fig_num)
        # Fallback: use beeswarm plot
        try:
            shap.plots.beeswarm(shap_values, X_shap, feature_names=feature_names, show=False)
            plot_created = True
            fig = plt.gcf()
        except Exception as e2:
            logger.warning(f"shap.plots.beeswarm also failed: {e2}. Creating simple bar plot...")
            # Close any figures created by shap.plots.beeswarm before creating bar plot
            current_fig_nums = set(plt.get_fignums())
            for fig_num in current_fig_nums - initial_fig_nums:
                plt.close(fig_num)
            # Last resort: create a simple bar plot of mean absolute SHAP values
            fig, ax = plt.subplots(figsize=(10, 8))
            mean_shap = np.abs(shap_values).mean(axis=0)
            
            # Bug 2 Fix: Validate that mean_shap length matches feature_names length
            if len(mean_shap) == 0:
                plt.close(fig)  # Close the figure we just created
                raise ValueError(f"No SHAP values to plot. Mean SHAP is empty.")
            if len(mean_shap) != len(feature_names):
                plt.close(fig)  # Close the figure we just created
                raise ValueError(f"SHAP values shape mismatch: {len(mean_shap)} SHAP values but {len(feature_names)} feature names. "
                               f"This indicates a data alignment issue.")
            
            # Create the bar plot with validated data
            y_pos = np.arange(len(feature_names))
            ax.barh(y_pos, mean_shap)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_names)
            ax.set_xlabel('Mean |SHAP value|', fontsize=12)
            title_base = f"{model_name} | SHAP Feature Importance (Mean Absolute)"
            title_with_label = f"{title_base} {label_suffix}" if label_suffix else title_base
            ax.set_title(title_with_label, fontsize=14, pad=20)
            ax.grid(axis='x', alpha=0.3)
            plot_created = True
    
    if not plot_created or fig is None:
        # Clean up any orphaned figures we created before raising error
        current_fig_nums = set(plt.get_fignums())
        for fig_num in current_fig_nums - initial_fig_nums:
            plt.close(fig_num)
        raise ValueError("Failed to create SHAP plot with any method")
    
    # Ensure the plot has content by checking figure size
    if fig.get_size_inches()[0] == 0 or fig.get_size_inches()[1] == 0:
        plt.close(fig)  # Close the invalid figure
        raise ValueError("SHAP plot figure has zero size")
    
    plt.tight_layout()
    
    # Save to file
    plot_filename = f"shap_summary{file_suffix}.png"
    plot_path = VIS_DIR / plot_filename
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    logger.debug(f"SHAP plot saved to {plot_path}")
    
    # Verify file was created and has content
    if not plot_path.exists():
        plt.close(fig)  # Close figure before raising error
        raise FileNotFoundError(f"SHAP plot file was not created at {plot_path}")
    if plot_path.stat().st_size == 0:
        plt.close(fig)  # Close figure before raising error
        raise ValueError(f"SHAP plot file is empty at {plot_path}")
    
    # Save to PDF
    pdf_pages.savefig(bbox_inches='tight', facecolor='white')
    
    # Bug 1 Fix: Close the specific figure we created
    plt.close(fig)
    
    # Clean up any remaining orphaned figures that may have been created during the process
    # (e.g., if shap.summary_plot created additional internal figures)
    current_fig_nums = set(plt.get_fignums())
    for fig_num in current_fig_nums - initial_fig_nums:
        plt.close(fig_num)
    
    return shap_values, explainer  # Return both shap_values and explainer
