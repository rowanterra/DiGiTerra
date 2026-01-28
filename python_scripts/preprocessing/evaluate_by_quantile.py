#!/usr/bin/env python
# coding: utf-8

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)

#Rowan 10/13
def evaluate_by_quantile(model_name, y_true, y_pred, metadata, target_col, sigfig, pdf_pages, quantile_col):

    # If a list is passed for target_col, extract the only item
    if isinstance(target_col, list):
        target_col = target_col[0]

    # Ensure y_true and y_pred are 1D arrays
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true[target_col].values
    elif isinstance(y_true, pd.Series):
        y_true = y_true.values

    if isinstance(y_pred, pd.DataFrame) or isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
        y_pred = y_pred[:, 0]
    logger.debug(f"Evaluating by quantile column: {quantile_col}")
    # Get quantile info from metadata
    quantiles = metadata.loc[:, quantile_col].reindex(metadata.index)

    # Align quantiles with y_true's index
    quantiles = quantiles.loc[metadata.index.intersection(metadata.index)]
    logger.debug("Constructing evaluation dataframe")
    # Construct clean DataFrame
    #BREAKING WHEN MAKING DATAFRAME: Error: Data must be 1-dimensional, got ndarray of shape (55, 2) instead
    df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'quantile': metadata.loc[metadata.index.intersection(metadata.index), quantile_col].values
    }, index=metadata.index.intersection(metadata.index))
    logger.debug("Grouping by quantile")
    # Group by quantile
    results = []
    for q in sorted(df['quantile'].unique()):
        group = df[df['quantile'] == q]
        r2 = r2_score(group['Actual'], group['Predicted'])
        mse = mean_squared_error(group['Actual'], group['Predicted'])
        mae = mean_absolute_error(group['Actual'], group['Predicted'])
        rmse = np.sqrt(mse)
        results.append({
            'quantile': q,
            'R²': r2,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'n': len(group)
        })
    logger.debug("Computing quantile metrics")
    results_df = pd.DataFrame(results)

    # Log results
    logger.info(f"{model_name} Performance by quantile:")
    logger.info(f"\n{results_df.round(sigfig)}")

    #Bar plot of R² by quantile
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x='quantile', y='R²', palette='viridis')
    plt.title(f"{model_name} | R² Score by Quantile")
    plt.ylabel("R² Score")
    plt.xlabel("Quantile")
    plt.ylim(-1, 1)
    plt.grid(True, axis='y')
    plt.tight_layout()
    pdf_pages.savefig()
    plt.close()

    return results_df

