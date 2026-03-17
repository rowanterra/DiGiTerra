from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any, Sequence
import re
import math
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from python_scripts.plotting.plot_style import apply_plot_style

logger = logging.getLogger(__name__)

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, QuantileTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    precision_recall_curve, average_precision_score, roc_curve, auc, roc_auc_score,
    silhouette_samples
)
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import label_binarize
from python_scripts.plotting.plot_shap_summary_graphic import plot_shap_summary
from python_scripts.plotting.plot_roc_curve import plot_roc_curve_from_estimator
from python_scripts.plotting.plot_precision_recall_curve import plot_precision_recall_curve_from_estimator
from python_scripts.plotting.plot_partial_dependence import plot_partial_dependence
from python_scripts.config import VIS_DIR
from python_scripts.plotting.bundles import export_plots


# -------------------------
# Utilities
# -------------------------

def sort_class_labels_numeric_bins(labels: Sequence[Any]) -> List[Any]:
    """
    Sort class labels so numeric bins appear in logical numeric order (e.g. <50, 50-150, 150-500, >500).
    Labels that look like ranges or thresholds are ordered by their lower bound. When no number is
    detected in a label (plain categories), those labels are sorted alphabetically and appear after
    any numeric-style labels.
    """
    if labels is None or len(labels) == 0:
        return list(labels) if labels is not None else []
    labels = list(labels)
    inf = float("inf")
    neg_inf = float("-inf")

    def sort_key(lab):
        s = str(lab).strip()
        # Plain number
        try:
            n = float(s.replace(",", ""))
            return (n, n, "")
        except ValueError:
            pass
        # > N or >N
        m = re.match(r"^>\s*([-\d.eE+]+)\s*$", s)
        if m:
            try:
                n = float(m.group(1))
                return (n, inf, "")
            except ValueError:
                pass
        # < N or <N
        m = re.match(r"^<\s*([-\d.eE+]+)\s*$", s)
        if m:
            try:
                n = float(m.group(1))
                return (neg_inf, n, "")
            except ValueError:
                pass
        # N-M or N - M (range)
        m = re.match(r"^([-\d.eE+]+)\s*[-–]\s*([-\d.eE+]+)\s*$", s)
        if m:
            try:
                a, b = float(m.group(1)), float(m.group(2))
                return (a, b, "")
            except ValueError:
                pass
        # No number detected: sort after numeric-style labels, then alphabetically by label
        return (inf, inf, s)

    return sorted(labels, key=sort_key)


def choose_columns_from_df(df: pd.DataFrame,
                           target_col: Optional[str],
                           max_ohe_cardinality: int = 50) -> Tuple[List[str], List[str], List[str]]:
    """
    If predictors are not specified, infer numeric vs categorical. Text cols left empty by default.
    Returns (numeric_predictors, categorical_predictors, text_cols).
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    obj_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    numeric_predictors = [c for c in num_cols if c != target_col]
    categorical_predictors = [c for c in obj_cols if c != target_col and df[c].nunique(dropna=True) <= max_ohe_cardinality]
    text_cols: List[str] = []
    return numeric_predictors, categorical_predictors, text_cols


def resolve_predictors(df: pd.DataFrame, numeric_predictors, categorical_predictors, target_col) -> Tuple[list, list]:
    """
    Use user-specified predictor lists when provided; otherwise infer only the missing parts.
    Returns (numeric_predictors, categorical_predictors, text_cols).
    """

    need_infer_num = numeric_predictors is None
    need_infer_cat = categorical_predictors is None
    #need_infer_txt = text_cols is None

    inf_num, inf_cat, inf_txt = ([], [], [])
    if need_infer_num or need_infer_cat: #or need_infer_txt:
        inf_num, inf_cat, inf_txt = choose_columns_from_df(df, target_col)

    nump = numeric_predictors if not need_infer_num else inf_num
    catp = categorical_predictors if not need_infer_cat else inf_cat
    #textp = text_cols if not need_infer_txt else inf_txt
    return nump, catp#, textp



def _get_scaler(kind: str):
    kind = (kind or "none").lower()
    if kind == "standard": return StandardScaler()
    if kind == "robust":   return RobustScaler()
    if kind == "quantile": return QuantileTransformer()
    if kind == "minmax":   return MinMaxScaler()
    if kind == "none":     return None
    raise ValueError("Unknown scaler kind")


def _group_by_map(series: pd.Series, mapping: Dict[str, str], regex: bool = True) -> pd.Series:
    def _map_one(s: str):
        if pd.isna(s): return s
        for pat, label in mapping.items():
            if (re.search(pat, s) if regex else pat == s):
                return label
        return s
    return series.astype(str).apply(_map_one)


def make_preprocessor(numeric_cols: List[str],
                      categorical_cols: Optional[List[str]] = None,
                      *, cat_mode: str = "onehot",
                      group_maps: Optional[Dict[str, Dict[str,str]]] = None) -> Pipeline:
    def apply_group_map(Xdf: pd.DataFrame):
        Xdf = Xdf.copy()
        if group_maps:
            for col, m in group_maps.items():
                if col in Xdf.columns:
                    Xdf[col] = _group_by_map(Xdf[col], m, regex=True)
        return Xdf

    transformers = []
    if numeric_cols:
        transformers.append(("num", "passthrough", numeric_cols))
    if categorical_cols:
        if cat_mode == "onehot":
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols))
        elif cat_mode == "passthrough":
            transformers.append(("cat", "passthrough", categorical_cols))
        else:
            raise ValueError("cat_mode must be 'onehot' or 'passthrough'")

    if not transformers:
        raise ValueError("No columns were provided to the preprocessor. Check your predictor lists.")

    coltx = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)
    return Pipeline(steps=[("group_map", FunctionTransformer(apply_group_map)),
                          ("columns", coltx)])


def get_feature_names(preprocessor: Pipeline, input_df: pd.DataFrame) -> List[str]:
    names: List[str] = []
    coltx = preprocessor.named_steps["columns"]
    for name, trans, cols in coltx.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if hasattr(trans, "get_feature_names_out"):
            base = cols if isinstance(cols, list) else [cols]
            try:
                fn = trans.get_feature_names_out(base).tolist()
            except Exception:
                fn = trans.get_feature_names_out().tolist()
            names.extend(fn)
        else:
            if isinstance(cols, list):
                names.extend(cols)
            elif isinstance(cols, str):
                names.append(cols)
    return names

#Rowan 10/13
def make_strat_labels_robust(ddf: pd.DataFrame,
                              target_col: str,
                              problem_type: str,
                              test_size: float = 0.2,
                              max_bins: int | Sequence[float] = 5,
                              min_bins: int = 2) -> Tuple[pd.DataFrame, bool, Dict]:
    """
    Creates quantile stratification column for train/test splitting. Returns (df_with_stratify_col, use_stratified_split, counts).
    Regression: quantile bins with decreasing bin count until valid; falls back to no stratify.
    Classification: uses label strings if each class has >=2 samples and at least one test sample.
    """
    pt = problem_type.strip().lower()
    y = ddf[target_col]
    counts = {}

    if pt == "regression":
        if isinstance(max_bins, (list, tuple, np.ndarray, pd.Series, pd.Index)):
            bins = list(max_bins)
            if len(bins) < 2:
                return ddf.assign(BIN_LABEL=None), False, {}
            try:
                lab = pd.cut(y, bins=bins, labels=False, include_lowest=True)
            except ValueError:
                return ddf.assign(BIN_LABEL=None), False, {}
            vc = lab.value_counts()
            ok = (vc >= 2).all() and ((vc * test_size) >= 1).all()
            if ok and lab.notna().all():
                counts = vc.sort_index().to_dict()
                return ddf.assign(BIN_LABEL=lab), True, counts
            return ddf.assign(BIN_LABEL=None), False, {}
        for q in range(max_bins, min_bins - 1, -1):
            try:
                lab = pd.qcut(y, q=q, labels=False, duplicates="drop")
            except ValueError:
                ranks = y.rank(method="average") / (len(y) + 1e-9)
                lab = pd.qcut(ranks, q=q, labels=False)
            vc = lab.value_counts()
            ok = (vc >= 2).all() and ((vc * test_size) >= 1).all()
            if ok and lab.notna().all():
                counts = vc.sort_index().to_dict()
                return ddf.assign(QUANTILE_STRATIFY=lab), True, counts
        return ddf.assign(QUANTILE_STRATIFY=None), False, {}

    elif pt == "classification":
        if isinstance(max_bins, (list, tuple, np.ndarray, pd.Series, pd.Index)):
            bins = list(max_bins)
            if len(bins) < 2:
                return ddf.assign(BIN_LABEL=None), False, {}
            try:
                lab = pd.cut(y, bins=bins, labels=False, include_lowest=True)
            except ValueError:
                return ddf.assign(BIN_LABEL=None), False, {}
            vc = lab.value_counts()
            ok = (vc >= 2).all() and ((vc * test_size) >= 1).all()
            if ok and lab.notna().all():
                counts = vc.sort_index().to_dict()
                return ddf.assign(BIN_LABEL=lab), True, counts
            return ddf.assign(BIN_LABEL=None), False, {}
        lab = y.astype(str)
        vc = lab.value_counts()
        ok = (vc >= 2).all() and ((vc * test_size) >= 1).all()
        if ok:
            counts = vc.sort_index().to_dict()
            return ddf.assign(QUANTILE_STRATIFY=lab), True, counts
        else:
            return ddf.assign(QUANTILE_STRATIFY=None), False, {}

    else:
        raise ValueError("problem_type must be 'regression' or 'classification' or 'cluster'")


def _scale_pairs(X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: Optional[pd.DataFrame], y_test: Optional[pd.DataFrame],
                 scale_X: str, scale_y: str):
    X_scaler = _get_scaler(scale_X)
    y_scaler = _get_scaler(scale_y) if y_train is not None else None

    if X_train.shape[1] == 0:
        raise ValueError("X_train has 0 columns before scaling. Did your transformer drop everything?")

    if X_scaler:
        X_train_s = pd.DataFrame(X_scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        if X_test is not None and len(X_test) > 0:
            X_test_s = pd.DataFrame(X_scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
        else:
            # Create empty DataFrame with same structure as X_train_s
            X_test_s = pd.DataFrame(columns=X_train.columns, dtype=X_train.dtypes)
    else:
        X_train_s = X_train.copy()
        X_test_s = X_test.copy() if X_test is not None and len(X_test) > 0 else pd.DataFrame(columns=X_train.columns, dtype=X_train.dtypes)

    if y_train is not None:
        if y_scaler:
            y_train_s = pd.DataFrame(y_scaler.fit_transform(y_train), columns=y_train.columns, index=y_train.index)
            y_test_s  = pd.DataFrame(y_scaler.transform(y_test),      columns=y_test.columns,  index=y_test.index) if y_test is not None else None
        else:
            y_train_s, y_test_s = y_train.copy(), (y_test.copy() if y_test is not None else None)
    else:
        y_train_s = y_test_s = None

    return X_train_s, X_test_s, y_train_s, y_test_s, X_scaler, y_scaler


def regression_report(y_true, y_pred, target_names=None, sigfig=3) -> pd.DataFrame:
    """
    Calculate comprehensive regression metrics using the enhanced calculate_all_metrics module.
    """
    from python_scripts.preprocessing.calculate_all_metrics import calculate_regression_metrics
    return calculate_regression_metrics(y_true, y_pred, target_names=target_names, sigfig=sigfig)


def visualize_regression(y_train: pd.DataFrame, y_train_pred: pd.DataFrame,
                         y_test: pd.DataFrame, y_test_pred: pd.DataFrame, units: str = ""):
    tname = y_test.columns[0] if isinstance(y_test, pd.DataFrame) else "y"
    yt = y_test[tname] if isinstance(y_test, pd.DataFrame) else pd.Series(np.ravel(y_test), name="y")
    yp = y_test_pred[tname] if isinstance(y_test_pred, pd.DataFrame) else pd.Series(np.ravel(y_test_pred), name="ŷ")

    # Pred vs Actual
    plt.figure(figsize=(6,6))
    plt.scatter(yt, yp, alpha=0.6, edgecolors="k")
    lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel(f"Actual {units}".strip()); plt.ylabel(f"Predicted {units}".strip())
    plt.title(f"Predicted vs Actual: {tname}")
    plt.tight_layout(); plt.show()

    # Residuals
    res = yt.values - yp.values
    plt.figure(figsize=(6,4))
    plt.hist(res, bins=30, edgecolor="black")
    plt.axvline(0, linestyle="--")
    plt.xlabel("Residual (y - ŷ)"); plt.ylabel("Count"); plt.title(f"Residuals: {tname}")
    plt.tight_layout(); plt.show()


def _ensure_numeric_series(s: pd.Series) -> pd.Series:
    if not np.issubdtype(s.dtype, np.number):
        return pd.to_numeric(s, errors="coerce")
    return s
