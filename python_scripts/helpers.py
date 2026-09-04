"""
Shared helpers for preprocessing, prediction, and Excel export. VIS_DIR (from config)
is the directory for predictions.csv and similar output. Paths are set in python_scripts/config.py.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xlsxwriter
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from python_scripts.config import VIS_DIR
from python_scripts.preprocessing.utilities import _get_scaler, make_preprocessor

logger = logging.getLogger(__name__)


def unpack_classification_result(result_tuple):
    """
    Split a classification result tuple into named parts, accepting the older shorter tuple.
    Handles old format (8 or 9 values) and new format (11 values with additional_metrics, feature_selection_info,
    outlier_info).

    Returns:
        tuple: (report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order,
                additional_metrics, feature_selection_info, outlier_info)
    """
    n = len(result_tuple)
    if n >= 11:
        return tuple(result_tuple[:11])
    if n >= 9:
        return result_tuple[:9] + (None,) * (11 - 9)
    # Old format - pad with None
    return result_tuple + (None,) * (11 - n)


def preprocess_data(
    df,
    target_cols=None,
    indicator_cols=None,
    drop_missing="all",  # 'all', 'indicator', 'target' should be a check box or something maybe?
    impute_strategy=None,  # None, 'mean', 'median', 'knn', or 0 or 0.01
    drop_zero="all",
    zero_handle="drop",  # 'none', 'drop', 'mean', 'median', or numeric constant for replacement
    zero_replace_values=None,  # list of exact values to convert to 0
    zero_replace_patterns=None,  # list of lambda or callable to match patterns (e.g., lambda x: str(x).startswith('<'))
    fill_values=None,  # {column: value} to reuse instead of recomputing (inference reuses training fills)
    fill_values_out=None,  # mutable dict the function writes the fills it used into
    zero_fill_values=None,  # {column: value} replacing zeros, reused at inference
    fill_plan_out=None,  # a FillPlan; when given, mean/median fills are recorded here and deferred to the split
):
    df_clean = df.copy()
    indicator_cols = list(indicator_cols or [])
    target_cols = list(target_cols or [])

    # No supervised targets (e.g. clustering): target-only scopes are invalid; downgrade safely.
    if not target_cols:
        if drop_missing == "target":
            drop_missing = "none"
        elif drop_missing == "indicatorAndTarget":
            drop_missing = "indicator"
        if drop_zero == "target":
            drop_zero = "none"
        elif drop_zero == "indicatorAndTarget":
            drop_zero = "indicator"

    if drop_zero == "none" or (isinstance(zero_handle, str) and str(zero_handle).lower() == "none"):
        zero_handle_norm = "none"
    else:
        zero_handle_norm = zero_handle if zero_handle is not None else "drop"

    # no impute strategy: drop the rows the user selected
    if impute_strategy == "none":
        if drop_missing == "all":
            df_clean.dropna(inplace=True)
        elif drop_missing == "indicator":
            # one missing indicator drops the row
            df_clean.dropna(subset=indicator_cols, how="any", inplace=True)
        elif drop_missing == "target" and target_cols:
            # one missing target drops the row
            df_clean.dropna(subset=target_cols, inplace=True, how="any")
        elif drop_missing == "indicatorAndTarget":
            subset = indicator_cols + target_cols
            df_clean.dropna(subset=subset, inplace=True, how="any")

    # impute strategy: replace missing values
    else:
        if drop_missing == "all":
            impute_scope = list(df_clean.select_dtypes(include=[np.number]).columns)
        elif drop_missing == "indicator":
            impute_scope = list(df_clean[indicator_cols].select_dtypes(include=[np.number]).columns)
        elif drop_missing == "target" and target_cols:
            impute_scope = list(df_clean[target_cols].select_dtypes(include=[np.number]).columns)
        elif drop_missing == "indicatorAndTarget":
            subset_cols = indicator_cols + target_cols
            impute_scope = list(df_clean[subset_cols].select_dtypes(include=[np.number]).columns)
        else:
            impute_scope = []

        supplied = dict(fill_values or {})
        deferred = fill_plan_out is not None and impute_strategy in ("mean", "median")
        if deferred:
            # The pipeline fits these on its training split; see preprocessing/split_fill.py.
            fill_plan_out.nan_strategy = impute_strategy
            fill_plan_out.nan_columns = [c for c in impute_scope if c not in supplied]
        for col in impute_scope:
            if col in supplied:
                fill = supplied[col]
            elif deferred:
                continue
            elif impute_strategy == "mean":
                fill = df_clean[col].mean()
            elif impute_strategy == "median":
                fill = df_clean[col].median()
            elif isinstance(impute_strategy, (int, float)):
                fill = impute_strategy
            else:
                continue
            if pd.isna(fill):
                continue
            # Direct assignment, not fillna(inplace=True) on a column selection:
            # the latter is chained assignment and becomes a no-op under copy-on-write.
            df_clean[col] = df_clean[col].fillna(fill)
            if fill_values_out is not None:
                fill_values_out[col] = float(fill)

    def _numeric_columns_for_zero_scope(frame):
        if drop_zero == "all":
            return list(frame.select_dtypes(include=[np.number]).columns)
        if drop_zero == "indicator":
            cols = [c for c in indicator_cols if c in frame.columns]
            return list(frame[cols].select_dtypes(include=[np.number]).columns) if cols else []
        if drop_zero == "target":
            if not target_cols:
                return []
            cols = [c for c in target_cols if c in frame.columns]
            return list(frame[cols].select_dtypes(include=[np.number]).columns) if cols else []
        if drop_zero == "indicatorAndTarget":
            cols = list(
                dict.fromkeys(
                    [c for c in indicator_cols if c in frame.columns] + [c for c in target_cols if c in frame.columns]
                )
            )
            return list(frame[cols].select_dtypes(include=[np.number]).columns) if cols else []
        return []

    # Zeros: either drop rows (legacy) or replace zeros in scoped numeric columns
    if zero_handle_norm == "none":
        pass
    elif zero_handle_norm == "drop":
        if drop_zero == "all":
            df_clean = df_clean[(df_clean != 0).any(axis=1)]
        elif drop_zero == "indicator":
            df_clean = df_clean[~(df_clean[indicator_cols].eq(0).all(axis=1))]
        elif drop_zero == "target" and target_cols:
            df_clean = df_clean[~(df_clean[target_cols].eq(0).all(axis=1))]
        elif drop_zero == "indicatorAndTarget":
            subset = indicator_cols + target_cols
            df_clean = df_clean[~(df_clean[subset].eq(0).all(axis=1))]
    else:
        zcols = _numeric_columns_for_zero_scope(df_clean)
        zsupplied = dict(zero_fill_values or {})
        zdeferred = fill_plan_out is not None and zero_handle_norm in ("mean", "median")
        if zdeferred:
            fill_plan_out.zero_strategy = zero_handle_norm
            fill_plan_out.zero_columns = [
                c for c in zcols if c not in zsupplied and pd.api.types.is_numeric_dtype(df_clean[c])
            ]
        for col in zcols:
            if col not in df_clean.columns or not pd.api.types.is_numeric_dtype(df_clean[col]):
                continue
            mask = df_clean[col] == 0
            if not mask.any():
                continue
            non_zero = df_clean.loc[~mask, col]
            if col in zsupplied:
                repl = zsupplied[col]
            elif zdeferred:
                continue
            elif zero_handle_norm == "mean":
                repl = non_zero.mean()
            elif zero_handle_norm == "median":
                repl = non_zero.median()
            elif isinstance(zero_handle_norm, (int, float, np.floating, np.integer)):
                repl = float(zero_handle_norm)
            else:
                continue
            if pd.isna(repl):
                repl = 0.0
            df_clean.loc[mask, col] = repl

    # ---- Step 2: Replace values with 0 ----
    if zero_replace_values:
        df_clean.replace(zero_replace_values, 0, inplace=True)

    if zero_replace_patterns:
        for col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(
                lambda x: 0 if any([pattern(x) for pattern in zero_replace_patterns]) else x
            )

    return df_clean


def write_to_excelRegression(
    data,
    indicator_names,
    predictor_names,
    stratify_name,
    modelName,
    params,
    units,
    trainOverall,
    testOverall,
    train_results,
    test_results,
    scaler,
    seed,
    shapes,
    quantileBin_results,
    cross_validation_summary=None,
    feature_selection_info=None,
    outlier_info=None,
):
    workbook = xlsxwriter.Workbook(str(VIS_DIR / "model_performance.xlsx"), {"nan_inf_to_errors": True})
    worksheet = workbook.add_worksheet("Preprocessing")
    worksheet.write("A1", "Filename")
    worksheet.write("B1", data["filename"])
    worksheet.write("A2", "Indicators")
    indicator_list = indicator_names.tolist() if hasattr(indicator_names, "tolist") else list(indicator_names)
    worksheet.write("B2", f"{[str(i)[:10] for i in indicator_list]}")
    worksheet.write("A3", "Predictors")
    predictor_list = predictor_names.tolist() if hasattr(predictor_names, "tolist") else list(predictor_names)
    worksheet.write("B3", f"{[str(p)[:10] for p in predictor_list]}")
    worksheet.write("A4", "Stratify Variable")
    worksheet.write("B4", stratify_name)
    worksheet.write("A5", "Scaling")
    worksheet.write("B5", scaler)
    worksheet.write("A6", "Seed")
    worksheet.write("B6", seed)

    worksheet.write("A7", "X_train shape:")
    worksheet.write("B7", str(shapes["X_train"]))
    worksheet.write("A8", "X_test shape:")
    worksheet.write("B8", str(shapes["X_test"]))
    worksheet.write("A9", "y_train shape:")
    worksheet.write("B9", str(shapes["y_train"]))
    worksheet.write("A10", "y_test shape:")
    worksheet.write("B10", str(shapes["y_test"]))
    worksheet.write("A11", "train_groups")
    # train_groups.to_frame()
    # with pd.ExcelWriter('static/performance.xlsx', engine='xlsxwriter') as writer:
    #     train_groups.to_excel(writer, sheet_name='quantileBins', startrow=11, index=True, header=True)

    worksheet2 = workbook.add_worksheet("Model Selection")
    worksheet2.write("A1", "Model")
    worksheet2.write("B1", f"{modelName}")
    worksheet2.write("A2", "Hyperparameters")

    row = 3  # B2 means row 1 (0-indexed) and column 1 (also 0-indexed)
    col = 1  # Column B

    worksheet2.write(row - 1, col, "Parameter")  # Header
    worksheet2.write(row - 1, col + 1, "Value")  # Header

    for key, value in params.items():
        worksheet2.write(row, col, key)
        worksheet2.write(row, col + 1, str(value))  # Convert to string just in case
        row += 1

    worksheet3 = workbook.add_worksheet("Overall Performance")

    header_format = workbook.add_format({"bold": True, "bg_color": "#D7E4BC"})
    worksheet3.write(0, 0, "Units:", header_format)
    worksheet3.write(0, 1, units)

    worksheet3.write("A2", "Training Score")
    worksheet3.write("B2", trainOverall["R²"])
    worksheet3.write("A3", "Validation Score")
    worksheet3.write("B3", testOverall["R²"])

    worksheet3.write("A4", "Training RMSE")
    worksheet3.write("B4", trainOverall["RMSE"])
    worksheet3.write("A5", "Validation RMSE")
    worksheet3.write("B5", testOverall["RMSE"])

    worksheet3.write("A6", "Training MAE")
    worksheet3.write("B6", trainOverall["MAE"])
    worksheet3.write("A7", "Validation MAE")
    worksheet3.write("B7", testOverall["MAE"])

    # Add additional regression metrics if available
    row = 8
    if "Explained Variance" in trainOverall.index:
        worksheet3.write(f"A{row}", "Training Explained Variance")
        worksheet3.write(f"B{row}", trainOverall["Explained Variance"])
        row += 1
        worksheet3.write(f"A{row}", "Validation Explained Variance")
        worksheet3.write(f"B{row}", testOverall["Explained Variance"])
        row += 1

    if "Median AE" in trainOverall.index:
        worksheet3.write(f"A{row}", "Training Median AE")
        worksheet3.write(f"B{row}", trainOverall["Median AE"])
        row += 1
        worksheet3.write(f"A{row}", "Validation Median AE")
        worksheet3.write(f"B{row}", testOverall["Median AE"])
        row += 1

    if "Max Error" in trainOverall.index:
        worksheet3.write(f"A{row}", "Training Max Error")
        worksheet3.write(f"B{row}", trainOverall["Max Error"])
        row += 1
        worksheet3.write(f"A{row}", "Validation Max Error")
        worksheet3.write(f"B{row}", testOverall["Max Error"])
        row += 1

    if "MAPE" in trainOverall.index:
        worksheet3.write(f"A{row}", "Training MAPE")
        worksheet3.write(f"B{row}", trainOverall["MAPE"])
        row += 1
        worksheet3.write(f"A{row}", "Validation MAPE")
        worksheet3.write(f"B{row}", testOverall["MAPE"])

    workbook.close()
    file_path = VIS_DIR / "model_performance.xlsx"

    with pd.ExcelWriter(file_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        train_results.to_excel(writer, sheet_name="Train Results", index=False)
        test_results.to_excel(writer, sheet_name="Test Results", index=False)

        if isinstance(quantileBin_results, pd.DataFrame):
            logger.debug(f"Quantile bin results:\n{quantileBin_results}")
            quantileBin_results.to_excel(writer, sheet_name="quantile Bin Results", index=False)

        # Add Cross Validation sheet if available
        if cross_validation_summary and len(cross_validation_summary) > 0:
            cv_df = pd.DataFrame(cross_validation_summary)
            cv_df.to_excel(writer, sheet_name="Cross Validation", index=False)

        # Add Feature Selection sheet if available
        if feature_selection_info:
            fs_data = {
                "Property": [
                    "Method",
                    "K Requested",
                    "Original Features",
                    "Selected Features",
                    "Selected Feature Names",
                ],
                "Value": [
                    feature_selection_info.get("method", "N/A"),
                    feature_selection_info.get("k_requested", "N/A"),
                    feature_selection_info.get("original_count", "N/A"),
                    feature_selection_info.get("selected_count", "N/A"),
                    ", ".join(feature_selection_info.get("selected_features", []))
                    if feature_selection_info.get("selected_features")
                    else "N/A",
                ],
            }
            fs_df = pd.DataFrame(fs_data)
            fs_df.to_excel(writer, sheet_name="Feature Selection", index=False)

        # Add Outlier Handling sheet if available
        if outlier_info:
            oi_data = {
                "Property": ["Method", "Action", "Outliers Detected", "Original Samples", "Remaining Samples"],
                "Value": [
                    outlier_info.get("method", "N/A"),
                    outlier_info.get("action", "N/A"),
                    outlier_info.get("n_outliers", 0),
                    outlier_info.get("original_samples", "N/A"),
                    outlier_info.get("remaining_samples", "N/A"),
                ],
            }
            oi_df = pd.DataFrame(oi_data)
            oi_df.to_excel(writer, sheet_name="Outlier Handling", index=False)
            # If any training rows were removed as outliers, list their indices on a separate sheet
            removed = outlier_info.get("removed_row_indices")
            if removed and isinstance(removed, list) and len(removed) > 0:
                removed_df = pd.DataFrame({"Removed sample index": removed})
                removed_df.to_excel(writer, sheet_name="Outlier Samples Removed", index=False)


# Backward compatibility: base name no longer assumed to mean regression
write_to_excel = write_to_excelRegression


def write_to_excelClassifier(
    data,
    indicator_names,
    predictor_names,
    stratify_name,
    scaler,
    seed,
    modelName,
    params,
    units,
    report,
    cm,
    additional_metrics=None,
    outlier_info=None,
):
    """
    Write classification results and metrics to Excel.

    Args:
        additional_metrics: Dictionary with additional metrics like ROC AUC, Average Precision, etc.
                          from calculate_classification_metrics()
        outlier_info: Optional dict with method, action, n_outliers, original_samples, remaining_samples,
                      and optionally removed_row_indices (list of training row indices removed as outliers).
    """
    workbook = xlsxwriter.Workbook(str(VIS_DIR / "model_performance.xlsx"), {"nan_inf_to_errors": True})
    worksheet = workbook.add_worksheet("Preprocessing")
    worksheet.write("A1", "Filename")
    worksheet.write("B1", data["filename"])
    worksheet.write("A2", "Indicators")
    indicator_list = indicator_names.tolist() if hasattr(indicator_names, "tolist") else list(indicator_names)
    worksheet.write("B2", f"{[str(i)[:10] for i in indicator_list]}")
    worksheet.write("A3", "Predictors")
    predictor_list = predictor_names.tolist() if hasattr(predictor_names, "tolist") else list(predictor_names)
    worksheet.write("B3", f"{[str(p)[:10] for p in predictor_list]}")
    worksheet.write("A4", "Stratify Variable")
    worksheet.write("B4", stratify_name)
    worksheet.write("A5", "Scaling")
    worksheet.write("B5", scaler)
    worksheet.write("A6", "Seed")
    worksheet.write("B6", seed)

    worksheet2 = workbook.add_worksheet("Model Selection")
    worksheet2.write("A1", "Model")
    worksheet2.write("B1", f"{modelName}")
    worksheet2.write("A2", "Hyperparameters")

    row = 3
    col = 1

    worksheet2.write(row - 1, col, "Parameter")
    worksheet2.write(row - 1, col + 1, "Value")

    for key, value in params.items():
        worksheet2.write(row, col, key)
        worksheet2.write(row, col + 1, str(value))
        row += 1

    worksheet3 = workbook.add_worksheet("Overall Performance")
    header_format = workbook.add_format({"bold": True, "bg_color": "#D7E4BC"})
    worksheet3.write(0, 0, "Units:", header_format)
    worksheet3.write(0, 1, units)

    row = 1
    col = 0
    logger.debug("Writing classification results to Excel")
    for key, value in report.items():
        worksheet3.write(row, col, key)
        if key == "accuracy":
            worksheet3.write(row, col + 1, value)
        else:
            for key2, value2 in value.items():
                worksheet3.write(row, col + 1, key2)
                worksheet3.write(row, col + 2, value2)
                row += 1
        row += 1

    # Add confusion matrix
    col = 1
    worksheet3.write(row, 0, "Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            worksheet3.write(row + i + 1, col + j, cm[i, j])

    row += cm.shape[0] + 2

    # Add additional metrics if provided
    if additional_metrics:
        worksheet3.write(row, 0, "Additional Metrics")
        row += 1

        # Basic metrics
        if "accuracy" in additional_metrics:
            worksheet3.write(row, 0, "Accuracy")
            worksheet3.write(row, 1, additional_metrics["accuracy"])
            row += 1

        # Precision, Recall, F1 (macro, micro, weighted)
        for metric_type in ["precision", "recall", "f1"]:
            for avg_type in ["macro", "micro", "weighted"]:
                key = f"{metric_type}_{avg_type}"
                if key in additional_metrics:
                    worksheet3.write(row, 0, f"{metric_type.capitalize()} ({avg_type})")
                    worksheet3.write(row, 1, additional_metrics[key])
                    row += 1

        # ROC AUC
        if "roc_auc" in additional_metrics:
            worksheet3.write(row, 0, "ROC AUC")
            worksheet3.write(row, 1, additional_metrics["roc_auc"])
            row += 1
        for avg_type in ["macro", "micro", "weighted"]:
            key = f"roc_auc_{avg_type}"
            if key in additional_metrics:
                worksheet3.write(row, 0, f"ROC AUC ({avg_type})")
                worksheet3.write(row, 1, additional_metrics[key])
                row += 1

        # Average Precision
        if "average_precision" in additional_metrics:
            worksheet3.write(row, 0, "Average Precision")
            worksheet3.write(row, 1, additional_metrics["average_precision"])
            row += 1
        for avg_type in ["macro", "micro", "weighted"]:
            key = f"average_precision_{avg_type}"
            if key in additional_metrics:
                worksheet3.write(row, 0, f"Average Precision ({avg_type})")
                worksheet3.write(row, 1, additional_metrics[key])
                row += 1

    workbook.close()
    file_path = VIS_DIR / "model_performance.xlsx"

    # Use openpyxl to add additional sheets with detailed metrics
    with pd.ExcelWriter(file_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        # Classification Report as DataFrame
        if isinstance(report, dict):
            report_data = []
            for key, value in report.items():
                if key == "accuracy":
                    report_data.append({"Metric": "accuracy", "Value": value})
                else:
                    for key2, value2 in value.items():
                        report_data.append({"Metric": f"{key}_{key2}", "Value": value2})
            if report_data:
                report_df = pd.DataFrame(report_data)
                report_df.to_excel(writer, sheet_name="Classification Report", index=False)

        # Confusion Matrix as DataFrame
        cm_df = pd.DataFrame(cm)
        cm_df.to_excel(writer, sheet_name="Confusion Matrix", index=False)

        # Additional Metrics as DataFrame
        if additional_metrics:
            metrics_data = []
            # Basic metrics
            for key in [
                "accuracy",
                "precision_macro",
                "precision_micro",
                "precision_weighted",
                "recall_macro",
                "recall_micro",
                "recall_weighted",
                "f1_macro",
                "f1_micro",
                "f1_weighted",
            ]:
                if key in additional_metrics:
                    metrics_data.append({"Metric": key.replace("_", " ").title(), "Value": additional_metrics[key]})

            # ROC AUC metrics
            for key in ["roc_auc", "roc_auc_macro", "roc_auc_micro", "roc_auc_weighted"]:
                if key in additional_metrics:
                    metrics_data.append({"Metric": key.replace("_", " ").upper(), "Value": additional_metrics[key]})

            # Average Precision metrics
            for key in [
                "average_precision",
                "average_precision_macro",
                "average_precision_micro",
                "average_precision_weighted",
            ]:
                if key in additional_metrics:
                    metrics_data.append({"Metric": key.replace("_", " ").title(), "Value": additional_metrics[key]})

            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                metrics_df.to_excel(writer, sheet_name="Additional Metrics", index=False)

            # Per-class metrics
            if "per_class" in additional_metrics:
                per_class_data = []
                for key, value in additional_metrics["per_class"].items():
                    per_class_data.append({"Metric": key.replace("_", " ").title(), "Value": value})
                if per_class_data:
                    per_class_df = pd.DataFrame(per_class_data)
                    per_class_df.to_excel(writer, sheet_name="Per-Class Metrics", index=False)

            if "per_class_roc_auc" in additional_metrics:
                roc_auc_data = []
                for key, value in additional_metrics["per_class_roc_auc"].items():
                    roc_auc_data.append({"Metric": key.replace("_", " ").upper(), "Value": value})
                if roc_auc_data:
                    roc_auc_df = pd.DataFrame(roc_auc_data)
                    roc_auc_df.to_excel(writer, sheet_name="Per-Class ROC AUC", index=False)

            if "per_class_ap" in additional_metrics:
                ap_data = []
                for key, value in additional_metrics["per_class_ap"].items():
                    ap_data.append({"Metric": key.replace("_", " ").title(), "Value": value})
                if ap_data:
                    ap_df = pd.DataFrame(ap_data)
                    ap_df.to_excel(writer, sheet_name="Per-Class Average Precision", index=False)

        # Outlier Handling sheet (same as regression)
        if outlier_info:
            oi_data = {
                "Property": ["Method", "Action", "Outliers Detected", "Original Samples", "Remaining Samples"],
                "Value": [
                    outlier_info.get("method", "N/A"),
                    outlier_info.get("action", "N/A"),
                    outlier_info.get("n_outliers", 0),
                    outlier_info.get("original_samples", "N/A"),
                    outlier_info.get("remaining_samples", "N/A"),
                ],
            }
            oi_df = pd.DataFrame(oi_data)
            oi_df.to_excel(writer, sheet_name="Outlier Handling", index=False)
            removed = outlier_info.get("removed_row_indices")
            if removed and isinstance(removed, list) and len(removed) > 0:
                removed_df = pd.DataFrame({"Removed sample index": removed})
                removed_df.to_excel(writer, sheet_name="Outlier Samples Removed", index=False)


def _xlsx_safe_metric(value):
    """xlsxwriter with nan_inf_to_errors can raise on NaN/Inf; use a safe cell value."""
    try:
        if value is None:
            return "N/A"
        fv = float(value)
        if fv != fv or fv in (float("inf"), float("-inf")):  # NaN or inf
            return "N/A"
        return fv
    except (TypeError, ValueError):
        return value


def write_to_excelCluster(
    data,
    indicator_names,
    stratify_name,
    scaler,
    seed,
    modelName,
    params,
    units,
    train_silhouette,
    train_calinski_harabasz,
    train_davies_bouldin,
    test_silhouette,
    test_calinski_harabasz,
    test_davies_bouldin,
    best_k,
    centers,
    silhouette_grid,
):

    logger.debug(f"Cluster centers:\n{centers}")
    logger.debug(f"Silhouette grid:\n{silhouette_grid}")
    workbook = xlsxwriter.Workbook(str(VIS_DIR / "model_performance.xlsx"), {"nan_inf_to_errors": True})
    worksheet = workbook.add_worksheet("Preprocessing")
    worksheet.write("A1", "Filename")
    worksheet.write("B1", data["filename"])
    worksheet.write("A2", "Indicators")
    indicator_list = indicator_names.tolist() if hasattr(indicator_names, "tolist") else list(indicator_names)
    worksheet.write("B2", f"{[str(i)[:10] for i in indicator_list]}")
    worksheet.write("A3", "Stratify Variable")
    worksheet.write("B3", stratify_name)
    worksheet.write("A4", "Scaling")
    worksheet.write("B4", scaler)
    worksheet.write("A5", "Seed")
    worksheet.write("B5", seed)

    worksheet2 = workbook.add_worksheet("Model Selection")
    worksheet2.write("A1", "Model")
    worksheet2.write("B1", f"{modelName}")
    worksheet2.write("A2", "Hyperparameters")

    row = 3  # B2 means row 1 (0-indexed) and column 1 (also 0-indexed)
    col = 1  # Column B

    worksheet2.write(row - 1, col, "Parameter")  # Header
    worksheet2.write(row - 1, col + 1, "Value")  # Header

    for key, value in params.items():
        worksheet2.write(row, col, key)
        worksheet2.write(row, col + 1, str(value))  # Convert to string just in case
        row += 1

    worksheet3 = workbook.add_worksheet("Overall Performance")
    header_format = workbook.add_format({"bold": True, "bg_color": "#D7E4BC"})
    worksheet3.write(0, 0, "Units:", header_format)
    worksheet3.write(0, 1, units)

    worksheet3.write("A2", "Train Silhouette")
    worksheet3.write("B2", _xlsx_safe_metric(train_silhouette))
    worksheet3.write("A3", "Train Calinski Harabasz")
    worksheet3.write("B3", _xlsx_safe_metric(train_calinski_harabasz))
    worksheet3.write("A4", "Train Davies Bouldin")
    worksheet3.write("B4", _xlsx_safe_metric(train_davies_bouldin))

    worksheet3.write("A5", "Test Silhouette")
    worksheet3.write("B5", _xlsx_safe_metric(test_silhouette))
    worksheet3.write("A6", "Test Calinski Harabasz")
    worksheet3.write("B6", _xlsx_safe_metric(test_calinski_harabasz))
    worksheet3.write("A7", "Test Davies Bouldin")
    worksheet3.write("B7", _xlsx_safe_metric(test_davies_bouldin))

    worksheet3.write("A8", "Best K")
    worksheet3.write("B8", int(best_k) if best_k is not None else "N/A")

    centersrow = 8
    centerscol = 1
    worksheet3.write("A9", "Centers")
    if centers is not None and hasattr(centers, "shape"):
        for i in range(centers.shape[0]):
            for j in range(centers.shape[1]):
                worksheet3.write(centersrow + i, centerscol + j, _xlsx_safe_metric(centers[i, j]))
        gridrow = centersrow + i + 1
    else:
        worksheet3.write(centersrow, centerscol, "N/A")
        gridrow = centersrow + 1

    gridcol = 1
    worksheet3.write(gridrow, 0, "Silhouette Grid")
    gridrow += 1
    for key, value in silhouette_grid.items():
        worksheet3.write(gridrow, gridcol, key)
        worksheet3.write(gridrow, gridcol + 1, str(value))  # Convert to string just in case
        gridrow += 1

    workbook.close()
    VIS_DIR / "model_performance.xlsx"


def _build_cv_summary(results: dict, scoring: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize cross_validate output.

    scoring maps the display name used as the cross_validate key to the sklearn
    scorer string. sklearn returns error metrics negated under a neg_ scorer, and
    the display name hides that prefix, so the scorer string decides the sign.
    """
    fold_df = pd.DataFrame(results)
    summary_rows = []
    scoring = scoring or {}
    for key, values in results.items():
        if not key.startswith("test_"):
            continue
        metric_name = key.replace("test_", "")
        metric_values = np.array(values, dtype=float)
        scorer = scoring.get(metric_name, metric_name)
        # A neg_ scorer returns negated error; flip it back for display.
        if isinstance(scorer, str) and scorer.startswith("neg_"):
            metric_values = -metric_values
        if metric_name.startswith("neg_"):
            metric_name = metric_name.replace("neg_", "")
            metric_values = -metric_values
        summary_rows.append(
            {
                "Metric": metric_name,
                "Mean": float(np.mean(metric_values)),
                "Std": float(np.std(metric_values)),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    return summary_df, fold_df


def write_cross_validation_results(
    summary_df: pd.DataFrame, fold_df: pd.DataFrame, cv_type: str, cv_folds: int
) -> Path:
    file_path = VIS_DIR / "cross_validation.xlsx"
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        fold_df.to_excel(writer, sheet_name="Fold Scores", index=False)
        meta_df = pd.DataFrame([{"CV Type": cv_type, "Folds": cv_folds}])
        meta_df.to_excel(writer, sheet_name="Settings", index=False)
    return file_path


def run_cross_validation(
    df: pd.DataFrame,
    indicator_names,
    predictor_names,
    model,
    scaler: str,
    cv_type: str,
    cv_folds: int,
    useTransformer: str,
    transformer_cols,
    seed: int,
    problem_type: str,
    y_scaler_type: str = "none",
):
    if cv_type == "None" or cv_folds is None:
        return None, None

    if problem_type not in {"regression", "classification"}:
        return None, None

    X_raw = df[indicator_names].copy()
    y_raw = df[predictor_names].copy()

    if problem_type == "classification" and y_raw.shape[1] != 1:
        return None, None

    numeric_cols = X_raw.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = []
    if useTransformer == "Yes" and transformer_cols is not None:
        categorical_cols = [c for c in transformer_cols if c in X_raw.columns]

    preprocessor = make_preprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols, cat_mode="onehot")
    steps = [("preprocess", preprocessor)]
    scaler_obj = _get_scaler(scaler)
    if scaler_obj is not None:
        steps.append(("scaler", scaler_obj))

    estimator = clone(model)
    if problem_type == "regression" and y_raw.shape[1] == 1 and y_scaler_type and y_scaler_type.lower() != "none":
        y_scaler = _get_scaler(y_scaler_type)
        if y_scaler is not None:
            estimator = TransformedTargetRegressor(regressor=estimator, transformer=y_scaler)

    steps.append(("model", estimator))
    pipeline = Pipeline(steps)

    folds = cv_folds if cv_folds and cv_folds >= 2 else 5
    if cv_type == "StratifiedKFold" and problem_type == "classification":
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    elif cv_type == "RepeatedKFold":
        cv = RepeatedKFold(n_splits=folds, n_repeats=2, random_state=seed)
    elif cv_type == "RepeatedStratifiedKFold" and problem_type == "classification":
        cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=2, random_state=seed)
    elif cv_type == "ShuffleSplit":
        cv = ShuffleSplit(n_splits=folds, test_size=0.2, random_state=seed)
    elif cv_type == "StratifiedShuffleSplit" and problem_type == "classification":
        cv = StratifiedShuffleSplit(n_splits=folds, test_size=0.2, random_state=seed)
    else:
        cv = KFold(n_splits=folds, shuffle=True, random_state=seed)

    if problem_type == "classification":
        scoring = {
            "accuracy": "accuracy",
            "precision_weighted": "precision_weighted",
            "recall_weighted": "recall_weighted",
            "f1_weighted": "f1_weighted",
        }
    else:
        scoring = {
            "r2": "r2",
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
        }

    y_values = y_raw.iloc[:, 0] if isinstance(y_raw, pd.DataFrame) else y_raw
    results = cross_validate(pipeline, X_raw, y_values, scoring=scoring, cv=cv)
    summary_df, fold_df = _build_cv_summary(results, scoring)
    file_path = write_cross_validation_results(summary_df, fold_df, cv_type, folds)
    # Return both file path and summary data for frontend display
    return file_path, summary_df


def _model_for_inference_attrs(model):
    """Return the model that has _digiterra_* attributes (wrapper or best_estimator_)."""
    if hasattr(model, "best_estimator_"):
        inner = model.best_estimator_
        if (
            getattr(inner, "_digiterra_raw_features", None) is not None
            or getattr(inner, "_digiterra_preprocessor", None) is not None
        ):
            return inner
    return model


def prediction(
    df_clean,
    best_model,
    training_features,
    X_scaler,
    y_scaler,
    feature_order,
    target_names=None,
    applicability=None,
    conformal=None,
    model_type=None,
):

    logger.info("Starting prediction function")

    # Use the object that actually has inference attributes (e.g. best_estimator_ when model is GridSearchCV)
    model_for_attrs = _model_for_inference_attrs(best_model)

    # --- 1. Align raw features and replay train-time transformations ---
    raw_features = getattr(model_for_attrs, "_digiterra_raw_features", None) or feature_order
    missing_raw = sorted(set(raw_features) - set(df_clean.columns))
    if missing_raw:
        raise ValueError(f"Missing features in prediction file: {missing_raw}")

    X_new_raw = df_clean[raw_features].copy()

    preprocessor = getattr(model_for_attrs, "_digiterra_preprocessor", None)
    transformed_feature_names = getattr(model_for_attrs, "_digiterra_feature_names", None)
    model_feature_order = getattr(model_for_attrs, "_digiterra_model_features", None)

    if preprocessor is not None:
        X_transformed = preprocessor.transform(X_new_raw)
        X_new = pd.DataFrame(X_transformed, index=X_new_raw.index)
        if transformed_feature_names and len(transformed_feature_names) == X_new.shape[1]:
            X_new.columns = transformed_feature_names
    else:
        X_new = X_new_raw.copy()

    # --- 2. Scale, then subset ------------------------------------------------
    # Training order is scale -> feature selection, so the scaler was fit on the
    # full transformed frame. Subsetting to the selected columns first would hand
    # the scaler fewer columns than it was fit with.
    # feature_names_in_ is an ndarray; `arr or []` raises for more than one column.
    _names = getattr(X_scaler, "feature_names_in_", None)
    scaler_features = [str(n) for n in _names] if _names is not None and len(_names) > 0 else []
    if X_scaler is not None:
        if scaler_features:
            missing_for_scaler = sorted(set(scaler_features) - set(X_new.columns))
            if missing_for_scaler:
                raise ValueError(
                    f"Prediction preprocessing mismatch. Missing transformed features: {missing_for_scaler}"
                )
            X_scaled_full = pd.DataFrame(
                X_scaler.transform(X_new[scaler_features]),
                index=X_new.index,
                columns=scaler_features,
            )
        else:
            X_scaled_full = pd.DataFrame(X_scaler.transform(X_new), index=X_new.index, columns=X_new.columns)
    else:
        X_scaled_full = X_new

    if model_feature_order:
        missing_model_features = sorted(set(model_feature_order) - set(X_scaled_full.columns))
        if missing_model_features:
            raise ValueError(
                f"Prediction preprocessing mismatch. Missing transformed features: {missing_model_features}"
            )
        X_new_scaled = X_scaled_full[model_feature_order].values
    elif feature_order and set(feature_order).issubset(set(X_scaled_full.columns)):
        X_new_scaled = X_scaled_full[feature_order].values
    else:
        X_new_scaled = X_scaled_full.values

    # --- 3. Predict ---
    y_pred = best_model.predict(X_new_scaled)

    # --- 4. Map ordinal class indices back to labels (classification with pre-made bins) ---
    int_to_label = getattr(model_for_attrs, "_digiterra_int_to_label", None)
    if int_to_label is not None:
        y_pred = np.asarray(y_pred)
        flat = y_pred.ravel()
        try:
            mapped = np.array([int_to_label[int(i)] for i in flat])
        except (KeyError, TypeError, ValueError):
            mapped = flat
        y_pred = mapped.reshape(y_pred.shape)

    # --- 5. Ensure 2D shape for inverse_transform ---
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # --- 6. Inverse-transform only for regression targets ---
    # Cluster labels and encoded class indices are integers too, so a numeric dtype
    # is not enough to identify a scaled regression target. The training run records
    # its problem type on the model; anything else leaves the prediction alone.
    problem_type = getattr(model_for_attrs, "_digiterra_problem_type", None)
    if y_scaler is not None and problem_type in (None, "regression") and int_to_label is None:
        if np.issubdtype(y_pred.dtype, np.floating) or np.issubdtype(y_pred.dtype, np.integer):
            y_pred = y_scaler.inverse_transform(y_pred)

    # --- 7. Handle single vs multiple targets ---
    num_targets = y_pred.shape[1] if y_pred.ndim > 1 else 1

    if num_targets > 1:
        # Multiple targets: create a column for each target
        if target_names is None:
            target_names = [f"Target_{i + 1}" for i in range(num_targets)]
        elif len(target_names) != num_targets:
            # Fallback if target_names count doesn't match
            target_names = [f"Target_{i + 1}" for i in range(num_targets)]

        for i, target_name in enumerate(target_names):
            pred_col = f"Predicted_{target_name}"
            df_clean[pred_col] = y_pred[:, i]
    else:
        # Single target: maintain backward compatibility
        y_pred_flat = y_pred.ravel()
        df_clean["Predicted_Target"] = y_pred_flat

    # --- 8. How much to trust each prediction ---
    #
    # Regression carries a 90 percent interval from the residuals on the validation
    # rows, in target units. Classification carries the probability of each class,
    # which is the per-row statement a classifier already makes. A classifier without
    # predict_proba carries neither, and the model card says why.
    #
    # The branch is the task, not whichever record happens to be present. A conformal
    # record on a classification run is a leak from an earlier run, and taking it
    # would write NaN intervals over a string class and suppress the probabilities.
    task = str(model_type or "").strip().lower()
    if task == "classification" and conformal is not None:
        logger.warning("Ignoring a conformal record on a classification run; an interval is for a numeric target.")
        conformal = None
    try:
        if task != "classification" and conformal is not None and conformal.half_width:
            pred_columns = [c for c in df_clean.columns if c.startswith("Predicted_")]
            for column in pred_columns:
                name = column[len("Predicted_") :]
                width = conformal.half_width.get(name)
                if width is None and len(conformal.half_width) == 1:
                    width = next(iter(conformal.half_width.values()))
                if width is None:
                    continue
                values = pd.to_numeric(df_clean[column], errors="coerce")
                suffix = "" if len(pred_columns) == 1 else f"_{name}"
                df_clean[f"lower_90{suffix}"] = values - width
                df_clean[f"upper_90{suffix}"] = values + width
        elif task != "regression" and hasattr(model_for_attrs, "predict_proba"):
            probabilities = model_for_attrs.predict_proba(X_new_scaled)
            classes = list(getattr(model_for_attrs, "classes_", []))
            if int_to_label is not None:
                classes = [int_to_label.get(int(c), c) for c in classes]
            for index, label in enumerate(classes):
                df_clean[f"probability_{label}"] = probabilities[:, index]
            df_clean["max_probability"] = probabilities.max(axis=1)
    except Exception as e:
        logger.warning("Uncertainty not written: %s", e)

    # --- 9. How far each row sits from the rows the model was trained on ---
    #
    # A model returns a prediction for any row whose columns parse. These three
    # columns say which of those predictions rest on data the model has seen.
    # novelty_score is higher for a row more like the training rows; novel_row is
    # true below the threshold recorded at training time, which is not recomputed
    # from the new file because a file of unusual rows would then look ordinary.
    if applicability is not None:
        try:
            from python_scripts.preprocessing.applicability import score_rows

            scored = score_rows(applicability, pd.DataFrame(X_new_scaled, columns=applicability.feature_order))
            df_clean["novelty_score"] = scored["novelty_score"]
            df_clean["novel_row"] = scored["novel_row"]
            df_clean["out_of_range_columns"] = scored["out_of_range_columns"]
        except Exception as e:
            logger.warning("Novelty not scored: %s", e)

    # --- 10. Save ---
    df_clean.to_csv(VIS_DIR / "predictions.csv", index=False)

    logger.info("Predictions complete. Saved to predictions.csv")
