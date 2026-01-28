import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# import scipy.stats as stats
import numpy as np

import seaborn as sns

from python_scripts.config import VIS_DIR

def visualize_predictions(model_name, 
                          y_train, y_train_pred, 
                          y_test, y_test_pred, 
                          target_names, 
                          units, sigfig, pdf_pages,
                          train_results=None, test_results=None,
                          file_suffix='', label_suffix=''):


    # Convert to arrays and handle index alignment for DataFrames
    if isinstance(y_train_pred, pd.DataFrame):
        y_train_pred_array = y_train_pred.values
        train_pred_index = y_train_pred.index
    else:
        y_train_pred_array = np.array(y_train_pred)
        train_pred_index = None
    
    if isinstance(y_test_pred, pd.DataFrame):
        y_test_pred_array = y_test_pred.values
        test_pred_index = y_test_pred.index
    else:
        y_test_pred_array = np.array(y_test_pred)
        test_pred_index = None

    if y_train_pred_array.ndim == 1:
        y_train_pred_array = y_train_pred_array.reshape(-1, 1)
        y_test_pred_array = y_test_pred_array.reshape(-1, 1)

    for i, target in enumerate(target_names):
        if train_results is not None and test_results is not None:
            fig, axs = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={'width_ratios': [1.3, 1.3, 1]})
        else:
            fig, axs = plt.subplots(1, 2, figsize=(13, 5))

        if not units:
            unitstr = 'units'
        else:
            unitstr = units

        # === Scatter Plot ===
        # Align y_train with y_train_pred indices if they're DataFrames
        if isinstance(y_train, pd.DataFrame) and train_pred_index is not None:
            y_train_aligned = y_train.loc[train_pred_index, target]
        elif isinstance(y_train, pd.DataFrame):
            y_train_aligned = y_train[target]
        else:
            y_train_aligned = y_train if y_train.ndim == 1 else y_train[:, i]
        
        # Align y_test with y_test_pred indices if they're DataFrames
        if isinstance(y_test, pd.DataFrame) and test_pred_index is not None:
            y_test_aligned = y_test.loc[test_pred_index, target]
        elif isinstance(y_test, pd.DataFrame):
            y_test_aligned = y_test[target]
        else:
            y_test_aligned = y_test if y_test.ndim == 1 else y_test[:, i]
        
        # Ensure arrays have the same length
        if len(y_train_aligned) != len(y_train_pred_array[:, i]):
            raise ValueError(f"Size mismatch: y_train has {len(y_train_aligned)} samples but y_train_pred has {len(y_train_pred_array[:, i])} samples. "
                           f"This may be due to outlier removal. Ensure y_train matches the filtered predictions.")
        if len(y_test_aligned) != len(y_test_pred_array[:, i]):
            raise ValueError(f"Size mismatch: y_test has {len(y_test_aligned)} samples but y_test_pred has {len(y_test_pred_array[:, i])} samples.")
        
        axs[0].scatter(y_train_aligned, y_train_pred_array[:, i], alpha=0.6, label='Train', color='royalblue', edgecolor='k')
        axs[0].scatter(y_test_aligned, y_test_pred_array[:, i], alpha=0.6, label='Test', color='forestgreen', edgecolor='k')
        min_val = min(y_train_aligned.min(), y_test_aligned.min())
        max_val = max(y_train_aligned.max(), y_test_aligned.max())
        axs[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        title_base = f"{model_name} | {target}\nPredicted vs Actual"
        title_with_label = f"{title_base} {label_suffix}" if label_suffix else title_base
        axs[0].set_title(title_with_label)
        axs[0].set_xlabel(f"Actual '{unitstr}'")
        axs[0].set_ylabel(f"Predicted '{unitstr}'")
        axs[0].legend()
        axs[0].grid(True)

        # === Residuals ===
        residuals_test = y_test_aligned - y_test_pred_array[:, i]
        sns.histplot(residuals_test, bins=15, ax=axs[1], kde=True, 
                     color="mediumseagreen", edgecolor="black")
        axs[1].axvline(0, color='red', linestyle='--')
        residuals_title_base = f"{model_name} | {target}\nTest Residuals"
        residuals_title_with_label = f"{residuals_title_base} {label_suffix}" if label_suffix else residuals_title_base
        axs[1].set_title(residuals_title_with_label)
        axs[1].set_xlabel(f"Residual '{unitstr}'")
        axs[1].set_ylabel("Count")
        # === Metrics Table (only if present) ===
        if train_results is not None and test_results is not None:
            axs[2].axis('off')
            metrics_df = pd.DataFrame({
                'Metric': ['R²', 'MSE', 'RMSE', 'MAE'],
                'Train': train_results.iloc[i][1:].round(sigfig).values,
                'Test': test_results.iloc[i][1:].round(sigfig).values
            })
            table = axs[2].table(cellText=metrics_df.values,
                                 colLabels=metrics_df.columns,
                                 loc='center',
                                 cellLoc='center')
            table.scale(1.2, 1.5)
            axs[2].set_title("Summary Metrics", pad=20)
        plt.tight_layout()
        plot_filename = f"target_plot_{i + 1}{file_suffix}.png"
        plot_path = VIS_DIR / plot_filename
        plt.savefig(plot_path)
        pdf_pages.savefig()
        plt.close()
