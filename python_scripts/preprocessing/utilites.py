from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any, Sequence
import re
import math
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
from python_scripts.preprocessing.visualize_predictions import visualize_predictions
from python_scripts.plotting.plot_shap_summary_graphic import plot_shap_summary
from python_scripts.plotting.plot_roc_curve import plot_roc_curve_from_estimator
from python_scripts.plotting.plot_precision_recall_curve import plot_precision_recall_curve_from_estimator
from python_scripts.plotting.plot_partial_dependence import plot_partial_dependence
from python_scripts.config import VIS_DIR


# -------------------------
# Utilities
# -------------------------

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


def _fig_save_all_to_pdf(pdf_pages):
    """Save all open matplotlib figures to a PdfPages object.
    
    Args:
        pdf_pages: PdfPages object from matplotlib.backends.backend_pdf
    """
    for num in plt.get_fignums():
        fig = plt.figure(num)
        pdf_pages.savefig(fig)
        plt.close(fig)

def _plot_feature_importance(model, feature_names, title="Feature importance (model-based)"):
    # Works for linear (coef_) and tree-based (feature_importances_); otherwise skip
    vals = None
    try:
        if hasattr(model, "feature_importances_"):
            vals = model.feature_importances_
        elif hasattr(model, "coef_"):
            co = model.coef_
            import numpy as np
            vals = np.mean(np.abs(co), axis=0) if getattr(co, "ndim", 1) > 1 else np.abs(co)
    except Exception:
        vals = None
    if vals is None:
        return False
    import numpy as np, matplotlib.pyplot as plt
    order = np.argsort(vals)[::-1][:30]  # top 30
    plt.figure(figsize=(7, min(12, 0.35*len(order)+2)))
    plt.barh([feature_names[i] for i in order][::-1], vals[order][::-1])
    plt.title(title)
    plt.tight_layout()
    return True

def _plot_permutation_importance(model, X, y, feature_names, title="Permutation importance", n_repeats=10, random_state=42):
    try:
        res = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=None)
    except Exception:
        return False
    import numpy as np, matplotlib.pyplot as plt
    order = res.importances_mean.argsort()[::-1][:30]
    plt.figure(figsize=(7, min(12, 0.35*len(order)+2)))
    plt.barh([feature_names[i] for i in order][::-1], res.importances_mean[order][::-1])
    plt.title(title)
    plt.tight_layout()
    return True

def plot_regression_bundle(art: dict, units: str = ""):
    # Pred vs Actual, Residuals hist, Residuals vs Fitted, 2D density
    import numpy as np, pandas as pd, matplotlib.pyplot as plt
    y_test = art["splits"]["y_test"]
    y_pred = art["predictions"]["y_test_pred"]
    tname = y_test.columns[0] if isinstance(y_test, pd.DataFrame) else "y"
    yt = y_test[tname] if isinstance(y_test, pd.DataFrame) else pd.Series(np.ravel(y_test), name="y")
    yp = y_pred[tname] if isinstance(y_pred, pd.DataFrame) else pd.Series(np.ravel(y_pred), name="ŷ")
    def save_plot(filename: str) -> None:
        plt.savefig(VIS_DIR / filename)

    # 1) Pred vs Actual
    plt.figure(figsize=(6,6))
    plt.scatter(yt, yp, alpha=0.6, edgecolors="k")
    lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel(f"Actual {units}".strip()); plt.ylabel(f"Predicted {units}".strip())
    plt.title(f"Predicted vs Actual: {tname}")
    plt.tight_layout()
    save_plot("regression_predicted_vs_actual.png")

    # 2) Residuals histogram
    res = yt.values - yp.values
    plt.figure(figsize=(6,4))
    plt.hist(res, bins=30, edgecolor="black")
    plt.axvline(0, linestyle="--")
    plt.xlabel("Residual (y - ŷ)"); plt.ylabel("Count"); plt.title(f"Residuals: {tname}")
    plt.tight_layout()
    save_plot("regression_residuals_hist.png")

    # 3) Residuals vs Fitted
    plt.figure(figsize=(6,4))
    plt.scatter(yp, res, alpha=0.6, edgecolors="k")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Fitted (ŷ)"); plt.ylabel("Residual"); plt.title("Residuals vs Fitted")
    plt.tight_layout()
    save_plot("regression_residuals_vs_fitted.png")

    # 4) Actual vs Predicted density
    plt.figure(figsize=(6,6))
    plt.hist2d(yt.values, yp.values, bins=40)
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("Actual vs Predicted (density)")
    plt.tight_layout()
    save_plot("regression_density.png")

    # 5) Importance (model-based or permutation)
    model = art["model"]
    feat_names = art.get("feature_names", [])
    X_test = art["splits"]["X_test"]
    ok = _plot_feature_importance(model, feat_names, title="Feature importance (model)")
    if ok:
        save_plot("regression_feature_importance.png")
    else:
        ok_perm = _plot_permutation_importance(
            model,
            X_test.values,
            yt.values if hasattr(yt, "values") else yt,
            feat_names,
            title="Permutation importance",
        )
        if ok_perm:
            save_plot("regression_permutation_importance.png")
    
    # 6) Partial Dependence Plots (for tree-based models)
    try:
        # Only generate PDP for tree-based models that support it
        if hasattr(model, 'feature_importances_') and len(feat_names) > 0:
            # Plot top 6 most important features
            importances = model.feature_importances_
            top_features_idx = np.argsort(importances)[::-1][:6]
            top_features = [feat_names[i] if i < len(feat_names) else i for i in top_features_idx]
            
            if top_features:
                plot_partial_dependence(
                    model, X_test, features=top_features,
                    model_name=art.get("model_name", "Model"),
                    pdf_pages=None,
                    feature_names=feat_names,
                    n_cols=3,
                    grid_resolution=50
                )
                save_plot("regression_partial_dependence.png")
    except Exception as e:
        logger.debug(f"Could not generate partial dependence plots: {e}")

def plot_classification_bundle(art: dict, svctrue: bool):
    # Confusion matrix, ROC (per class + micro/macro), PR (per class + micro/macro), Calibration
    import numpy as np, matplotlib.pyplot as plt
    y_test = art["splits"]["y_test"]
    classes = art["metrics"]["classes"]
    y_pred = art["predictions"]["y_test_pred"]
    model = art["model"]
    X_test = art["splits"]["X_test"]

    # 1) Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()
    plot_path = VIS_DIR / "confusion_matrix.png"
    plt.savefig(plot_path)

    # prepare scores
    # Binarize y for multi-class
    y_bin = label_binarize(y_test, classes=classes)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        dfc = model.decision_function(X_test)
        if getattr(dfc, "ndim", 1) == 1:
            import numpy as np
            y_score = np.c_[-dfc, dfc]
        else:
            y_score = dfc
    else:
        y_score = None

    # 2) ROC curves using scikit-learn's RocCurveDisplay
    if y_score is not None:
        try:
            # Use scikit-learn's Display class for better formatting
            plot_roc_curve_from_estimator(
                model, X_test, y_test,
                model_name=art.get("model_name", "Model"),
                pdf_pages=None,
                plot_chance_level=True,
                despine=True
            )
        except Exception as e:
            logger.warning(f"Could not generate ROC curve using Display class: {e}")
            # Fallback to manual plotting for per-class curves
            import numpy as np
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                auc_i = auc(fpr, tpr)
                plt.figure()
                plt.plot(fpr, tpr, linewidth=2)
                plt.plot([0,1],[0,1], linestyle="--")
                plt.title(f"ROC: {cls} (AUC={auc_i:.3f})")
                plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
                plt.tight_layout()

        # 3) PR curves using scikit-learn's PrecisionRecallDisplay
        try:
            plot_precision_recall_curve_from_estimator(
                model, X_test, y_test,
                model_name=art.get("model_name", "Model"),
                pdf_pages=None,
                plot_chance_level=True,
                despine=True
            )
        except Exception as e:
            logger.warning(f"Could not generate PR curve using Display class: {e}")
            # Fallback to manual plotting
            from sklearn.metrics import precision_recall_curve, average_precision_score
            for i, cls in enumerate(classes):
                rec, prec, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
                ap = average_precision_score(y_bin[:, i], y_score[:, i])
                plt.figure()
                plt.plot(rec, prec, linewidth=2)
                plt.title(f"Precision-Recall: {cls} (AP={ap:.3f})")
                plt.xlabel("Recall"); plt.ylabel("Precision")
                plt.tight_layout()

        # 4) Calibration curves (require predict_proba; decision_function scores not in [0,1])
        from sklearn.calibration import CalibrationDisplay
        if not svctrue and hasattr(model, "predict_proba"):
            for i, cls in enumerate(classes):
                plt.figure()
                CalibrationDisplay.from_predictions(y_true=y_bin[:, i], y_prob=y_score[:, i], n_bins=10)
                plt.title(f"Calibration curve: class {cls}")
                plt.tight_layout()

    # 5) Feature importance / permutation
    feat_names = art.get("feature_names", [])
    ok = _plot_feature_importance(model, feat_names, title="Feature importance (model)")
    if not ok:
        _ = _plot_permutation_importance(model, X_test.values, y_test.values, feat_names, title="Permutation importance")
    
    # 6) Partial Dependence Plots (for tree-based models)
    try:
        # Only generate PDP for tree-based models that support it
        if hasattr(model, 'feature_importances_') and len(feat_names) > 0:
            # Plot top 6 most important features
            if ok and hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                top_features_idx = np.argsort(importances)[::-1][:6]
                top_features = [feat_names[i] if i < len(feat_names) else i for i in top_features_idx]
            else:
                # Use first few features if no importance available
                top_features = feat_names[:6] if len(feat_names) >= 6 else feat_names
            
            if top_features:
                plot_partial_dependence(
                    model, X_test, features=top_features,
                    model_name=art.get("model_name", "Model"),
                    pdf_pages=None,
                    feature_names=feat_names,
                    n_cols=3,
                    grid_resolution=50
                )
    except Exception as e:
        logger.debug(f"Could not generate partial dependence plots: {e}")



def plot_clustering_bundle(art: dict):
    # Silhouette plot, PCA scatter (train/test), cluster size bar
    import numpy as np, matplotlib.pyplot as plt
    X_train = art["splits"]["X_train"].values
    labels_train = art["clusters"]["labels_train"]
    centers = art["clusters"]["centers"]
    best_k = art["clusters"]["best_k"]
    X_test = art["splits"]["X_test"].values if len(art["splits"]["X_test"]) else None
    labels_test = art["clusters"]["labels_test"] if len(art["splits"]["X_test"]) else None

    # 1) Silhouette plot
    if len(np.unique(labels_train)) > 1:
        sample_sil = silhouette_samples(X_train, labels_train)
        y_lower = 10
        plt.figure(figsize=(7,5))
        for i in range(best_k):
            ith = sample_sil[labels_train==i]
            ith.sort()
            size_i = ith.shape[0]
            y_upper = y_lower + size_i
            import numpy as np
            plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith, alpha=0.6)
            plt.text(0, y_lower + 0.5*size_i, str(i))
            y_lower = y_upper + 10
        plt.axvline(np.mean(sample_sil), linestyle="--")
        plt.xlabel("Silhouette coefficient"); plt.ylabel("Samples")
        plt.title("Silhouette plot (train)")
        plt.tight_layout()

    # 2) PCA scatter
    from sklearn.decomposition import PCA
    if X_train.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=42).fit(X_train)
        Z = pca.transform(X_train)
        plt.figure(figsize=(6.5,5.5))
        plt.scatter(Z[:,0], Z[:,1], c=labels_train, s=25, edgecolor="k", alpha=0.8)
        Cz = pca.transform(centers) if centers is not None else None
        if Cz is not None:
            plt.scatter(Cz[:,0], Cz[:,1], s=120, marker="X", label="centers")
            plt.legend()
        plt.title(f"Clusters (PCA 2D): k={best_k} (train)"); plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.tight_layout()
        plot_path = VIS_DIR / "cluster_pca_train.png"
        plt.savefig(plot_path)

        if X_test is not None and labels_test is not None and len(X_test):
            Zt = pca.transform(X_test)
            plt.figure(figsize=(6.5,5.5))
            plt.scatter(Zt[:,0], Zt[:,1], c=labels_test, s=25, edgecolor="k", alpha=0.8)
            if Cz is not None:
                plt.scatter(Cz[:,0], Cz[:,1], s=120, marker="X", label="centers")
                plt.legend()
            plt.title(f"Clusters (PCA 2D): k={best_k} (test)"); plt.xlabel("PC1"); plt.ylabel("PC2")
            plt.tight_layout()
            plot_path = VIS_DIR / "cluster_pca_test.png"
            plt.savefig(plot_path)

    # 3) Cluster sizes
    import pandas as pd
    plt.figure(figsize=(6,4))
    pd.Series(labels_train).value_counts().sort_index().plot(kind="bar")
    plt.title("Cluster sizes (train)"); plt.xlabel("cluster"); plt.ylabel("count")
    plt.tight_layout()

def export_plots(art: dict, pdf_pages, units: str = "", svctrue=False) -> Optional[str]:
    """
    Auto-detects problem type from art['config'].problem_type and generates a bundle of plots.
    If out_pdf_path is provided, saves all current figures to that PDF and returns the path.
    """
    ptype = art["type"]
    if ptype == "regression":
        plot_regression_bundle(art, units=units)
    elif ptype == "classification":
        plot_classification_bundle(art, svctrue=svctrue)
    elif ptype == "cluster":
        plot_clustering_bundle(art)
    else:
        raise ValueError("Unknown problem type for export_plots.")

    _fig_save_all_to_pdf(pdf_pages)
    return pdf_pages



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
