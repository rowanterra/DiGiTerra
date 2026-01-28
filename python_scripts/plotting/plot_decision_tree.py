#!/usr/bin/env python
# coding: utf-8

"""
Decision Tree plotting using scikit-learn's plot_tree.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from sklearn.tree import plot_tree
from python_scripts.config import VIS_DIR


def plot_decision_tree(decision_tree, model_name, pdf_pages=None,
                      max_depth=None, feature_names=None, class_names=None,
                      label='all', filled=False, impurity=True, node_ids=False,
                      proportion=False, rounded=False, precision=3, ax=None,
                      fontsize=None, file_suffix='', label_suffix=''):
    """
    Plot a decision tree using scikit-learn's plot_tree.
    
    Args:
        decision_tree: Decision tree regressor or classifier
        model_name: Name of the model
        pdf_pages: PdfPages object for saving to PDF (optional)
        max_depth: Maximum depth of representation
        feature_names: Names of each feature
        class_names: Names of each target class
        label: {'all', 'root', 'none'}
        filled: Whether to paint nodes
        impurity: Whether to show impurity at each node
        node_ids: Whether to show ID number on each node
        proportion: Whether to show proportions/percentages
        rounded: Whether to use rounded corners and Helvetica fonts
        precision: Number of digits of precision
        ax: Matplotlib axis to plot to
        fontsize: Size of text font
        file_suffix: Suffix for filename
        label_suffix: Suffix for plot title
        
    Returns:
        List of artists for annotation boxes
    """
    # Create figure if ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))
    else:
        fig = ax.figure
    
    # Plot tree
    annotations = plot_tree(
        decision_tree,
        max_depth=max_depth,
        feature_names=feature_names,
        class_names=class_names,
        label=label,
        filled=filled,
        impurity=impurity,
        node_ids=node_ids,
        proportion=proportion,
        rounded=rounded,
        precision=precision,
        ax=ax,
        fontsize=fontsize
    )
    
    # Add title
    title_base = f"{model_name} | Decision Tree"
    title_with_label = f"{title_base} {label_suffix}" if label_suffix else title_base
    ax.set_title(title_with_label, fontsize=14, pad=20)
    
    plt.tight_layout()
    
    # Save to file
    plot_filename = f"decision_tree{file_suffix}.png"
    plot_path = VIS_DIR / plot_filename
    fig.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"Decision tree plot saved to {plot_path}")
    
    # Save to PDF if provided
    if pdf_pages is not None:
        pdf_pages.savefig(fig, bbox_inches='tight', facecolor='white')
    
    return annotations
