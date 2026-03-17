"""Plot bundles for regression, classification, and clustering (export_plots and helpers)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from python_scripts.plotting.plot_style import apply_plot_style
from python_scripts.config import VIS_DIR

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    roc_auc_score,
    silhouette_samples,
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA

from python_scripts.plotting.plot_roc_curve import plot_roc_curve_from_estimator
from python_scripts.plotting.plot_precision_recall_curve import plot_precision_recall_curve_from_estimator
from python_scripts.plotting.plot_partial_dependence import plot_partial_dependence
from python_scripts.plotting.plot_shap_summary_graphic import plot_shap_summary

logger = logging.getLogger(__name__)


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
    import numpy as np
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except ImportError:
        sns = None
    order = np.argsort(vals)[::-1][:30]  # top 30
    names = [feature_names[i] for i in order][::-1]
    values = vals[order][::-1]
    fig, ax = plt.subplots(figsize=(7, min(12, 0.35 * len(order) + 2)))
    if sns is not None:
        sns.barplot(x=values, y=names, ax=ax, palette="muted", orient="h")
        ax.set_xlabel("Importance", fontsize=11)
    else:
        ax.barh(names, values, color=".7", edgecolor="none")
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    return True

def _plot_permutation_importance(model, X, y, feature_names, title="Permutation importance", n_repeats=10, random_state=42):
    try:
        res = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=None)
    except Exception:
        return False
    import numpy as np
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except ImportError:
        sns = None
    order = res.importances_mean.argsort()[::-1][:30]
    names = [feature_names[i] for i in order][::-1]
    values = res.importances_mean[order][::-1]
    fig, ax = plt.subplots(figsize=(7, min(12, 0.35 * len(order) + 2)))
    if sns is not None:
        sns.barplot(x=values, y=names, ax=ax, palette="muted", orient="h")
        ax.set_xlabel("Mean importance", fontsize=11)
    else:
        ax.barh(names, values, color=".7", edgecolor="none")
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    return True

def plot_regression_bundle(art: dict, units: str = ""):
    apply_plot_style()
    # Pred vs Actual, Residuals hist, Residuals vs Fitted, 2D density
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except ImportError:
        sns = None
    y_test = art["splits"]["y_test"]
    y_pred = art["predictions"]["y_test_pred"]
    tname = y_test.columns[0] if isinstance(y_test, pd.DataFrame) else "y"
    yt = y_test[tname] if isinstance(y_test, pd.DataFrame) else pd.Series(np.ravel(y_test), name="y")
    yp = y_pred[tname] if isinstance(y_pred, pd.DataFrame) else pd.Series(np.ravel(y_pred), name="ŷ")
    def save_plot(filename: str) -> None:
        plt.savefig(VIS_DIR / filename, dpi=150, bbox_inches="tight", facecolor="white")

    lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
    res = yt.values - yp.values

    # 1) Pred vs Actual
    fig, ax = plt.subplots(figsize=(6, 6))
    if sns is not None:
        sns.scatterplot(x=yt, y=yp, alpha=0.65, s=28, ax=ax, color=".4")
    else:
        ax.scatter(yt, yp, alpha=0.65, edgecolors="none", s=28)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color=".45", linewidth=1.5)
    ax.set_xlabel(f"Actual {units}".strip())
    ax.set_ylabel(f"Predicted {units}".strip())
    ax.set_title(f"Predicted vs Actual: {tname}")
    plt.tight_layout()
    save_plot("regression_predicted_vs_actual.png")
    plt.close(fig)

    # 2) Residuals histogram (seaborn histplot with KDE when available)
    fig, ax = plt.subplots(figsize=(6, 4))
    if sns is not None:
        sns.histplot(res, bins=30, kde=True, ax=ax, color=".5", edgecolor="white", linewidth=0.8)
    else:
        ax.hist(res, bins=30, edgecolor="white", linewidth=0.8)
    ax.axvline(0, linestyle="--", color=".5")
    ax.set_xlabel("Residual (y - ŷ)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residuals: {tname}")
    plt.tight_layout()
    save_plot("regression_residuals_hist.png")
    plt.close(fig)

    # 3) Residuals vs Fitted
    fig, ax = plt.subplots(figsize=(6, 4))
    if sns is not None:
        sns.scatterplot(x=yp, y=res, alpha=0.65, s=28, ax=ax, color=".4")
    else:
        ax.scatter(yp, res, alpha=0.65, edgecolors="none", s=28)
    ax.axhline(0, linestyle="--", color=".5")
    ax.set_xlabel("Fitted (ŷ)")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs Fitted")
    plt.tight_layout()
    save_plot("regression_residuals_vs_fitted.png")
    plt.close(fig)

    # 4) Actual vs Predicted density
    fig, ax = plt.subplots(figsize=(6, 6))
    if sns is not None:
        try:
            sns.histplot(x=yt.values, y=yp.values, bins=40, cmap="Blues", cbar=True, ax=ax, pthresh=0.05)
        except TypeError:
            ax.hist2d(yt.values, yp.values, bins=40, cmap="Blues")
    else:
        ax.hist2d(yt.values, yp.values, bins=40, cmap="Blues")
    ax.plot([lo, hi], [lo, hi], linestyle="--", color=".45", linewidth=1.5)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted (density)")
    plt.tight_layout()
    save_plot("regression_density.png")
    plt.close(fig)

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

def _plot_single_confusion_matrix(y_true, y_pred, classes, plot_path: Path, title_suffix: str = ""):
    """Draw and save one confusion matrix to plot_path. Uses consistent string labels and clear grid."""
    import numpy as np
    apply_plot_style()
    classes = list(classes)
    class_list = [str(c).strip() for c in classes]
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_true_str = np.array([str(x).strip() for x in y_true])
    y_pred_str = np.array([str(x).strip() for x in y_pred])
    cm = confusion_matrix(y_true_str, y_pred_str, labels=class_list)
    short_classes = [
        label if len(label) <= 12 else label[:10] + "…"
        for label in class_list
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=short_classes)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
    ax.grid(False)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title("Confusion Matrix" + (f" ({title_suffix})" if title_suffix else ""), fontsize=13)
    plt.xticks(rotation=45, ha="right")
    for text in disp.text_.ravel():
        text.set_fontsize(11)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)


def plot_classification_bundle(art: dict, svctrue: bool):
    # Confusion matrix (train + test), ROC, PR, Calibration
    import numpy as np, matplotlib.pyplot as plt
    y_test = art["splits"]["y_test"]
    y_train = art["splits"]["y_train"]
    X_train = art["splits"]["X_train"]
    X_test = art["splits"]["X_test"]
    classes = art["metrics"]["classes"]
    y_pred = art["predictions"]["y_test_pred"]
    model = art["model"]

    # Ordinal: model may predict integers; map back to labels for display
    int_to_label = getattr(model, "_digiterra_int_to_label", None)
    def _to_labels(arr):
        if int_to_label is None:
            return np.asarray(arr).ravel()
        flat = np.asarray(arr).ravel()
        return np.array([int_to_label[int(x)] for x in flat])

    # 1) Confusion matrix – test set only (train matrix removed; no separate train confusion matrix)
    y_test_1d = np.asarray(y_test).ravel()
    y_pred_labels = _to_labels(y_pred)
    _plot_single_confusion_matrix(y_test_1d, y_pred_labels, classes, VIS_DIR / "confusion_matrix.png", "Test")

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

    # 2) ROC curves using scikit-learn's RocCurveDisplay (micro-averaged -> roc_curve.png)
    if y_score is not None:
        try:
            plot_roc_curve_from_estimator(
                model, X_test, y_test,
                model_name=art.get("model_name", "Model"),
                pdf_pages=None,
                plot_chance_level=True,
                despine=True
            )
        except Exception as e:
            logger.warning(f"Could not generate ROC curve using Display class: {e}")
        # Per-class ROC (one curve per class) -> roc_curve_per_class.png
        try:
            fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                auc_i = roc_auc_score(y_bin[:, i], y_score[:, i])
                ax_roc.plot(fpr, tpr, linewidth=2, label=f"{cls} (AUC={auc_i:.3f})")
            ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title(f"{art.get('model_name', 'Model')} | ROC (per class)")
            ax_roc.legend(loc="lower right", fontsize=9)
            ax_roc.set_xlim([0, 1])
            ax_roc.set_ylim([0, 1.05])
            ax_roc.grid(True, alpha=0.7)
            plt.tight_layout()
            plt.savefig(VIS_DIR / "roc_curve_per_class.png", dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig_roc)
        except Exception as e2:
            logger.warning(f"Could not generate per-class ROC: {e2}")

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
            # Fallback: save micro-averaged PR curve to the expected filename
            from sklearn.metrics import precision_recall_curve, average_precision_score
            try:
                y_bin_flat = y_bin.ravel()
                y_score_flat = y_score.reshape(y_bin.shape[0], -1)
                n_classes = y_score_flat.shape[1]
                prec_list, rec_list = [], []
                for i in range(n_classes):
                    rec, prec, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
                    prec_list.append(prec)
                    rec_list.append(rec)
                fig, ax = plt.subplots(figsize=(6, 5))
                for i, cls in enumerate(classes):
                    ap = average_precision_score(y_bin[:, i], y_score[:, i])
                    ax.plot(rec_list[i], prec_list[i], linewidth=2, label=f"{cls} (AP={ap:.3f})")
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title(f"{art.get('model_name', 'Model')} | Precision-Recall Curve")
                ax.legend(loc="best", fontsize=9)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1.05])
                ax.grid(True, alpha=0.7)
                plt.tight_layout()
                plt.savefig(VIS_DIR / "precision_recall_curve.png", dpi=150, bbox_inches="tight", facecolor="white")
                plt.close(fig)
            except Exception as e2:
                logger.warning(f"Could not generate fallback PR curve: {e2}")

        # 4) Calibration curves (require predict_proba; decision_function scores not in [0,1])
        from sklearn.calibration import CalibrationDisplay
        if not svctrue and hasattr(model, "predict_proba"):
            for i, cls in enumerate(classes):
                fig_cal, ax_cal = plt.subplots(figsize=(6, 5))
                CalibrationDisplay.from_predictions(y_true=y_bin[:, i], y_prob=y_score[:, i], n_bins=10, ax=ax_cal)
                ax_cal.set_title(f"{art.get('model_name', 'Model')} | Calibration: {cls}")
                ax_cal.grid(False)
                plt.tight_layout()
                if i == 0:
                    fig_cal.savefig(VIS_DIR / "calibration_curve.png", dpi=150, bbox_inches="tight", facecolor="white")
                plt.close(fig_cal)

    # 4b) SHAP feature importance for classifiers
    try:
        effective_model = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        X_train_art = art["splits"]["X_train"]
        shap_feat_names = list(art.get("feature_names", []))
        if not shap_feat_names and hasattr(X_train_art, "columns"):
            shap_feat_names = X_train_art.columns.tolist()
        if shap_feat_names and len(X_train_art) > 0:
            plot_shap_summary(
                effective_model, X_train_art, shap_feat_names,
                art.get("model_name", "Model"), pdf_pages=None, file_suffix=""
            )
    except Exception as e:
        logger.warning("Could not generate SHAP for classifier: %s", e)

    # 5) Feature importance / permutation (fallback when SHAP not used)
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
    apply_plot_style()
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except ImportError:
        sns = None
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
        fig, ax = plt.subplots(figsize=(7, 5))
        palette = sns.color_palette("muted", best_k) if sns else None
        for i in range(best_k):
            ith = sample_sil[labels_train == i]
            ith.sort()
            size_i = ith.shape[0]
            y_upper = y_lower + size_i
            color = palette[i] if palette is not None else f"C{i}"
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith, alpha=0.7, color=color)
            ax.text(-0.02, y_lower + 0.5 * size_i, str(i), fontsize=10, va="center")
            y_lower = y_upper + 10
        ax.axvline(np.mean(sample_sil), linestyle="--", color=".4")
        ax.set_xlabel("Silhouette coefficient")
        ax.set_ylabel("Samples")
        ax.set_title("Silhouette plot (train)")
        ax.grid(False)
        plt.tight_layout()
        plt.savefig(VIS_DIR / "cluster_silhouette.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # 2) PCA scatter
    from sklearn.decomposition import PCA
    if X_train.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=42).fit(X_train)
        Z = pca.transform(X_train)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        if sns is not None:
            sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=labels_train, palette="muted", s=35, alpha=0.85, ax=ax, legend="brief")
        else:
            ax.scatter(Z[:, 0], Z[:, 1], c=labels_train, s=25, edgecolor="k", alpha=0.8, cmap="tab10")
        Cz = pca.transform(centers) if centers is not None else None
        if Cz is not None:
            ax.scatter(Cz[:, 0], Cz[:, 1], s=120, marker="X", c="black", label="centers", zorder=5)
            ax.legend()
        ax.set_title(f"Clusters (PCA 2D): k={best_k} (train)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.tight_layout()
        plt.savefig(VIS_DIR / "cluster_pca_train.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        if X_test is not None and labels_test is not None and len(X_test):
            Zt = pca.transform(X_test)
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            if sns is not None:
                sns.scatterplot(x=Zt[:, 0], y=Zt[:, 1], hue=labels_test, palette="muted", s=35, alpha=0.85, ax=ax, legend="brief")
            else:
                ax.scatter(Zt[:, 0], Zt[:, 1], c=labels_test, s=25, edgecolor="k", alpha=0.8, cmap="tab10")
            if Cz is not None:
                ax.scatter(Cz[:, 0], Cz[:, 1], s=120, marker="X", c="black", label="centers", zorder=5)
                ax.legend()
            ax.set_title(f"Clusters (PCA 2D): k={best_k} (test)")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            plt.tight_layout()
            plt.savefig(VIS_DIR / "cluster_pca_test.png", dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)

    # 3) Cluster sizes (seaborn barplot for cleaner look)
    counts = pd.Series(labels_train).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    if sns is not None:
        sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, palette="muted")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Count")
    else:
        counts.plot(kind="bar", ax=ax, color=".7", edgecolor="none")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Count")
    ax.set_title("Cluster sizes (train)")
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "cluster_sizes.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

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
