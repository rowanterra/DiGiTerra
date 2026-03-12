"""Data exploration handlers for DiGiTerra.

Auto-detect transformers, auto-detect NaN/zeros, correlation matrices, and pairplot.
Invoked by app.py route handlers. Receives store and request data to avoid circular imports.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from werkzeug.utils import secure_filename

from python_scripts.helpers import preprocess_data

logger = logging.getLogger(__name__)

IMAGE_DPI = 150
CORRELATION_HEATMAP_LINEWIDTH = 0.5
CORRELATION_CBAR_SHRINK = 0.8


def handle_auto_detect_transformers(store: dict, data: dict):
    """Return (response_dict, status_code). On error, response_dict has 'error' key."""
    if "data" not in store:
        return ({"error": "No data uploaded. Please upload a file first."}, 400)
    if not data or "indicators" not in data:
        return ({"error": "No indicators provided"}, 400)
    try:
        df = store["data"]
        selected_indicators = data["indicators"]
        if isinstance(selected_indicators, (int, np.integer)):
            selected_indicators = [selected_indicators]
        try:
            indicator_names = df.columns.take(selected_indicators).tolist()
        except (KeyError, IndexError, TypeError):
            indicator_names = df.columns[selected_indicators].tolist()
        categorical_indices = []
        max_cardinality = 50
        for idx, col_name in enumerate(indicator_names):
            col_idx = selected_indicators[idx]
            col_data = df[col_name]
            is_object_type = not pd.api.types.is_numeric_dtype(col_data)
            unique_count = col_data.nunique(dropna=True)
            is_low_cardinality = unique_count <= max_cardinality
            is_integer_like = False
            if pd.api.types.is_numeric_dtype(col_data):
                if col_data.dropna().dtype in [np.int64, np.int32, np.int16, np.int8]:
                    if unique_count <= max_cardinality and col_data.min() >= 0:
                        is_integer_like = True
            if is_object_type or (is_low_cardinality and is_integer_like):
                categorical_indices.append(col_idx)
        return (
            {
                "transformer_indices": categorical_indices,
                "message": f"Found {len(categorical_indices)} categorical column(s) to transform.",
            },
            200,
        )
    except Exception as e:
        logger.error("Error auto-detecting transformers: %s", e, exc_info=True)
        return ({"error": f"Error detecting categorical columns: {str(e)}"}, 500)


def handle_auto_detect_nan_zeros(store: dict, data: dict):
    """Return (response_dict, status_code). On error, response_dict has 'error' key."""
    if "data" not in store:
        return ({"error": "No data uploaded. Please upload a file first."}, 400)
    if not data or "indicators" not in data or "predictors" not in data:
        return ({"error": "Indicators and predictors are required"}, 400)
    try:
        df = store["data"]
        inds = data["indicators"]
        preds = data["predictors"]
        if isinstance(inds, (int, np.integer)):
            inds = [inds]
        if isinstance(preds, (int, np.integer)):
            preds = [preds]
        indices = list(dict.fromkeys(list(inds) + list(preds)))
        try:
            col_names = df.columns.take(indices).tolist()
        except (KeyError, IndexError, TypeError):
            col_names = [df.columns[i] for i in indices]
        cols_df = df[col_names]
        needs_missing_handling = cols_df.isna().any().any()
        numeric_cols = cols_df.select_dtypes(include=[np.number])
        needs_zero_handling = (numeric_cols == 0).any().any() if not numeric_cols.empty else False
        if not needs_missing_handling and not needs_zero_handling:
            message = "No missing values or zeros detected in indicator/target columns. You can leave cleaning as \"No Columns\"."
        else:
            parts = []
            if needs_missing_handling:
                parts.append("missing values")
            if needs_zero_handling:
                parts.append("zeros")
            message = f"Detected {' and '.join(parts)} in indicator/target columns. Please choose how to handle them below."
        return (
            {
                "needs_missing_handling": bool(needs_missing_handling),
                "needs_zero_handling": bool(needs_zero_handling),
                "message": message,
                "ok": not (needs_missing_handling or needs_zero_handling),
            },
            200,
        )
    except Exception as e:
        logger.error("Error auto-detecting NaN/zeros: %s", e, exc_info=True)
        return ({"error": str(e)}, 500)


def handle_corr(
    store: dict,
    data: dict,
    user_vis_dir: Path,
    with_prefix_fn,
):
    """Return (response_dict, status_code). On error, response_dict has 'error' key."""
    if not data:
        return ({"error": "No data provided"}, 400)
    if "data" not in store:
        return ({"error": "No data uploaded. Please upload a file first."}, 400)
    df = store["data"]
    drop_missing = data.get("dropMissing", "none")
    impute_strategy = data.get("imputeStrategy", "none")
    drop_zero = data.get("dropZero", "none")
    if "colsIgnore" not in data:
        return ({"error": "Missing required field: colsIgnore"}, 400)
    if data["colsIgnore"] == "all":
        cols_names = df.columns
    else:
        try:
            cols_indices = data["colsIgnore"]
            cols_names = df.columns[cols_indices]
        except (KeyError, IndexError, TypeError) as e:
            logger.error("Error accessing column indices: %s", e, exc_info=True)
            return ({"error": "Invalid column indices provided."}, 400)
    df_corr = df[cols_names].select_dtypes(include="number")
    if df_corr.empty:
        return ({"error": "No numeric columns available for correlation matrices."}, 400)
    if impute_strategy in {"0", "0.01"}:
        impute_strategy = float(impute_strategy)
    df_corr = preprocess_data(
        df=df_corr,
        target_cols=df_corr.columns.tolist(),
        indicator_cols=df_corr.columns.tolist(),
        drop_missing=drop_missing,
        impute_strategy=impute_strategy,
        drop_zero=drop_zero,
    )
    if df_corr.empty:
        return ({"error": "No rows available after preprocessing for correlation matrices."}, 400)
    user_vis_dir.mkdir(parents=True, exist_ok=True)
    corr_methods = [
        ("Pearson Correlation", "pearson"),
        ("Spearman Correlation", "spearman"),
        ("Kendall Correlation", "kendall"),
    ]
    correlation_images = {}
    for title, method in corr_methods:
        correlation_image_path = user_vis_dir / f"correlation_{method}.png"
        corr_matrix = df_corr.corr(method=method)
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(
            corr_matrix,
            ax=ax,
            cmap="vlag",
            center=0,
            annot=False,
            linewidths=CORRELATION_HEATMAP_LINEWIDTH,
            linecolor="white",
            cbar_kws={"shrink": CORRELATION_CBAR_SHRINK},
        )
        ax.set_title(title)
        plt.tight_layout()
        fig.savefig(correlation_image_path, dpi=IMAGE_DPI)
        plt.close(fig)
        correlation_images[method] = with_prefix_fn(f"/user-visualizations/correlation_{method}.png")
    pdf_path = user_vis_dir / "correlation_matrices.pdf"
    pairplot_image = None
    with PdfPages(pdf_path) as pdf_pages:
        for title, method in corr_methods:
            corr_matrix = df_corr.corr(method=method)
            fig, ax = plt.subplots(figsize=(7, 7))
            sns.heatmap(
                corr_matrix,
                ax=ax,
                cmap="vlag",
                center=0,
                annot=True,
                fmt=".2f",
                linewidths=CORRELATION_HEATMAP_LINEWIDTH,
                linecolor="white",
                cbar_kws={"shrink": 0.8},
            )
            ax.set_title(title)
            plt.tight_layout()
            pdf_pages.savefig(fig)
            plt.close(fig)
        numeric_columns = df_corr.columns.tolist()
        if len(numeric_columns) >= 2:
            x_col, y_col = numeric_columns[0], numeric_columns[1]
            safe_x = secure_filename(str(x_col))
            safe_y = secure_filename(str(y_col))
            pairplot_path = user_vis_dir / f"pairplot_{safe_x}_{safe_y}.png"
            grid = sns.pairplot(df_corr[[x_col, y_col]], diag_kind="hist")
            grid.savefig(pairplot_path, dpi=IMAGE_DPI)
            pdf_pages.savefig(grid.fig)
            plt.close("all")
            pairplot_image = with_prefix_fn(f"/user-visualizations/pairplot_{safe_x}_{safe_y}.png")
    xlsx_path = user_vis_dir / "correlation_matrices.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_corr.describe().T.to_excel(writer, sheet_name="Descriptive Statistics")
        for title, method in corr_methods:
            df_corr.corr(method=method).to_excel(writer, sheet_name=title)
    selected_columns = list(df_corr.columns)
    descriptive_stats = []
    if selected_columns:
        stats_df = df_corr[selected_columns].describe(percentiles=[0.25, 0.5, 0.75, 1.0]).T
        stats_df = stats_df.rename(columns={
            "count": "n", "min": "min", "max": "max", "mean": "mean", "std": "std",
            "25%": "25", "50%": "50", "75%": "75", "100%": "100",
        }).reset_index().rename(columns={"index": "column"})
        stats_df["column"] = stats_df["column"].astype(str).str.slice(0, 10)
        stats_df = stats_df[["column", "n", "min", "max", "mean", "std", "25", "50", "75", "100"]].round(4)
        descriptive_stats = stats_df.to_dict(orient="records")
    numeric_columns = df_corr.columns.tolist()
    return (
        {
            "correlation_image": correlation_images.get("pearson"),
            "correlation_images": correlation_images,
            "descriptive_stats": descriptive_stats,
            "numeric_columns": numeric_columns,
            "pairplot_image": pairplot_image,
        },
        200,
    )


def handle_pairplot(
    store: dict,
    data: dict,
    user_vis_dir: Path,
    with_prefix_fn,
):
    """Return (response_dict, status_code). On error, response_dict has 'error' key."""
    if not data:
        return ({"error": "No data provided"}, 400)
    if "data" not in store:
        return ({"error": "No data uploaded. Please upload a file first."}, 400)
    if "colsIgnore" not in data:
        return ({"error": "Missing required field: colsIgnore"}, 400)
    df = store["data"]
    x_col = data.get("x")
    y_col = data.get("y")
    drop_missing = data.get("dropMissing", "none")
    impute_strategy = data.get("imputeStrategy", "none")
    drop_zero = data.get("dropZero", "none")
    if not x_col or not y_col:
        return ({"error": "Missing required fields: x, y"}, 400)
    df_numeric = df.select_dtypes(include="number")
    if df_numeric.empty:
        return ({"error": "No numeric columns available for pairplot."}, 400)
    if impute_strategy in {"0", "0.01"}:
        impute_strategy = float(impute_strategy)
    df_numeric = preprocess_data(
        df=df_numeric,
        target_cols=df_numeric.columns.tolist(),
        indicator_cols=df_numeric.columns.tolist(),
        drop_missing=drop_missing,
        impute_strategy=impute_strategy,
        drop_zero=drop_zero,
    )
    if df_numeric.empty:
        return ({"error": "No rows available after preprocessing for pairplot."}, 400)
    if x_col not in df_numeric.columns or y_col not in df_numeric.columns:
        return ({"error": "Selected columns are not available for pairplot."}, 400)
    safe_x = secure_filename(str(x_col))
    safe_y = secure_filename(str(y_col))
    pairplot_path = user_vis_dir / f"pairplot_{safe_x}_{safe_y}.png"
    grid = sns.pairplot(df_numeric[[x_col, y_col]], diag_kind="hist")
    grid.savefig(pairplot_path, dpi=150)
    plt.close("all")
    return (
        {"pairplot_image": with_prefix_fn(f"/user-visualizations/pairplot_{safe_x}_{safe_y}.png")},
        200,
    )
