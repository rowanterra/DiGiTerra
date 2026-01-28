#!/usr/bin/env python
# coding: utf-8

"""
Partial Dependence Plot (PDP) and Individual Conditional Expectation (ICE) plotting
using scikit-learn's PartialDependenceDisplay.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from sklearn.inspection import PartialDependenceDisplay
from python_scripts.config import VIS_DIR


def plot_partial_dependence(estimator, X, features, model_name, pdf_pages=None,
                           sample_weight=None, categorical_features=None,
                           feature_names=None, target=None, response_method='auto',
                           n_cols=3, grid_resolution=100, percentiles=(0.05, 0.95),
                           method='auto', n_jobs=None, verbose=0,
                           kind='average', centered=False, subsample=1000,
                           random_state=None, file_suffix='', label_suffix='',
                           **kwargs):
    """
    Plot partial dependence plots using scikit-learn's PartialDependenceDisplay.from_estimator.
    
    Args:
        estimator: Fitted estimator object
        X: Input data (used to generate grid and complement features)
        features: List of feature indices or names for PDPs
        model_name: Name of the model
        pdf_pages: PdfPages object for saving to PDF (optional)
        sample_weight: Sample weights
        categorical_features: Indicates categorical features
        feature_names: Name of each feature
        target: In multiclass/multioutput, specifies class/task
        response_method: {'auto', 'predict_proba', 'decision_function'}
        n_cols: Maximum number of columns in grid plot
        grid_resolution: Number of equally spaced points on axes
        percentiles: Lower and upper percentile for PDP axes
        method: {'auto', 'recursion', 'brute'}
        n_jobs: Number of CPUs to use
        verbose: Verbose output
        kind: {'average', 'individual', 'both'}
        centered: Whether to center ICE and PD lines at origin
        subsample: Sampling for ICE curves
        random_state: Random state
        file_suffix: Suffix for filename
        label_suffix: Suffix for plot title
        **kwargs: Additional keyword arguments
        
    Returns:
        PartialDependenceDisplay object
    """
    # Create display from estimator
    disp = PartialDependenceDisplay.from_estimator(
        estimator, X, features,
        sample_weight=sample_weight,
        categorical_features=categorical_features,
        feature_names=feature_names,
        target=target,
        response_method=response_method,
        n_cols=n_cols,
        grid_resolution=grid_resolution,
        percentiles=percentiles,
        method=method,
        n_jobs=n_jobs,
        verbose=verbose,
        kind=kind,
        centered=centered,
        subsample=subsample,
        random_state=random_state,
        **kwargs
    )
    
    # Add title
    title_base = f"{model_name} | Partial Dependence Plot"
    title_with_label = f"{title_base} {label_suffix}" if label_suffix else title_base
    if disp.bounding_ax_ is not None:
        disp.bounding_ax_.set_title(title_with_label, fontsize=14, pad=20)
    elif disp.figure_ is not None:
        disp.figure_.suptitle(title_with_label, fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    # Save to file
    plot_filename = f"partial_dependence{file_suffix}.png"
    plot_path = VIS_DIR / plot_filename
    if disp.figure_ is not None:
        disp.figure_.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Partial dependence plot saved to {plot_path}")
        
        # Save to PDF if provided
        if pdf_pages is not None:
            pdf_pages.savefig(disp.figure_, bbox_inches='tight', facecolor='white')
    
    return disp
