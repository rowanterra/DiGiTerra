#!/usr/bin/env python
# coding: utf-8

"""
Permutation Importance plotting using scikit-learn's permutation_importance.
"""

import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import python_scripts.plotting.plot_style  # noqa: F401
from sklearn.inspection import permutation_importance
from python_scripts.config import VIS_DIR

logger = logging.getLogger(__name__)


def plot_permutation_importance(estimator, X, y, model_name, pdf_pages=None,
                                scoring=None, n_repeats=5, n_jobs=None,
                                random_state=None, sample_weight=None,
                                max_samples=1.0, feature_names=None,
                                file_suffix='', label_suffix=''):
    """
    Plot permutation importance using scikit-learn's permutation_importance.
    
    Args:
        estimator: Fitted estimator
        X: Data on which permutation importance will be computed
        y: Targets for supervised or None for unsupervised
        model_name: Name of the model
        pdf_pages: PdfPages object for saving to PDF (optional)
        scoring: Scorer to use
        n_repeats: Number of times to permute a feature
        n_jobs: Number of jobs to run in parallel
        random_state: Random state
        sample_weight: Sample weights
        max_samples: Number of samples to draw from X
        feature_names: Names of features
        file_suffix: Suffix for filename
        label_suffix: Suffix for plot title
        
    Returns:
        Result object from permutation_importance
    """
    # Compute permutation importance
    result = permutation_importance(
        estimator, X, y,
        scoring=scoring,
        n_repeats=n_repeats,
        n_jobs=n_jobs,
        random_state=random_state,
        sample_weight=sample_weight,
        max_samples=max_samples
    )
    
    # Handle multiple scorers
    if isinstance(result, dict):
        # Multiple scorers - create subplots
        n_scorers = len(result)
        fig, axes = plt.subplots(1, n_scorers, figsize=(6 * n_scorers, 6))
        if n_scorers == 1:
            axes = [axes]
        
        for idx, (scorer_name, scorer_result) in enumerate(result.items()):
            ax = axes[idx]
            _plot_single_importance(scorer_result, feature_names, ax, scorer_name)
        
        plt.suptitle(f"{model_name} | Permutation Importance {label_suffix}".strip(), 
                    fontsize=14, y=1.02)
    else:
        # Single scorer
        fig, ax = plt.subplots(figsize=(10, 6))
        _plot_single_importance(result, feature_names, ax, None)
        title_base = f"{model_name} | Permutation Importance"
        title_with_label = f"{title_base} {label_suffix}" if label_suffix else title_base
        ax.set_title(title_with_label, fontsize=14, pad=20)
    
    plt.tight_layout()
    
    # Save to file
    plot_filename = f"permutation_importance{file_suffix}.png"
    plot_path = VIS_DIR / plot_filename
    fig.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    logger.debug(f"Permutation importance plot saved to {plot_path}")
    
    # Save to PDF if provided
    if pdf_pages is not None:
        pdf_pages.savefig(fig, bbox_inches='tight', facecolor='white')
    
    return result


def _plot_single_importance(result, feature_names, ax, scorer_name=None):
    """
    Helper function to plot a single permutation importance result.
    Uses seaborn-style muted palette when available.
    """
    importances_mean = result.importances_mean
    importances_std = result.importances_std

    indices = np.argsort(importances_mean)[::-1]
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances_mean))]
    else:
        feature_names = list(feature_names)

    names = [feature_names[i] for i in indices]
    means = importances_mean[indices]
    stds = importances_std[indices]
    y_pos = np.arange(len(means))

    try:
        import seaborn as sns
        palette = sns.color_palette("muted", len(means))
        bars = ax.barh(y_pos, means, xerr=stds, capsize=4, color=palette, edgecolor="none")
    except Exception:
        bars = ax.barh(y_pos, means, xerr=stds, capsize=4, color=".65", edgecolor="none")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Mean importance", fontsize=11)
    title = "Permutation Importance"
    if scorer_name:
        title += f" ({scorer_name})"
    ax.set_title(title, fontsize=12)
    ax.grid(axis="x", alpha=0.35)
    ax.invert_yaxis()
