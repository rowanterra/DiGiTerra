from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

import python_scripts.plotting.plot_style  # noqa: F401
from python_scripts.config import VIS_DIR
from python_scripts.plotting.bundles import export_plots
from python_scripts.preprocessing.outlier_handling import apply_outlier_handling
from python_scripts.preprocessing.utilities import _scale_pairs, get_feature_names, make_preprocessor

logger = logging.getLogger(__name__)


def _choose_k_silhouette(X: np.ndarray, k_min=2, k_max=10, algo="kmeans", seed=42):
    """Silhouette score for each k in the range. Returns the best k and the full grid.

    This is a suggestion shown next to the result. It does not change the model,
    which is fit with the number of clusters the user chose.
    """
    scores = {}
    for k in range(max(k_min, 2), max(k_max, 2) + 1):
        if algo == "kmeans":
            mdl = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
            labels = mdl.labels_
        elif algo == "gmm":
            mdl = GaussianMixture(n_components=k, covariance_type="full", random_state=seed).fit(X)
            labels = mdl.predict(X)
        else:
            mdl = AgglomerativeClustering(n_clusters=k).fit(X)
            labels = mdl.labels_
        if len(np.unique(labels)) > 1:
            scores[k] = silhouette_score(X, labels)
    if not scores:
        return 2, {}
    best_k = max(scores, key=scores.get)
    return best_k, scores


def _fit_clusterer(model, algo, Xtr: np.ndarray, n_clusters):
    if algo == "kmeans":
        # mdl = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(Xtr)
        mdl = model.fit(Xtr)
        centers = mdl.cluster_centers_
        predict = mdl.predict
        labels_tr = mdl.labels_
    elif algo == "gmm":
        # mdl = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=42).fit(Xtr)
        mdl = model.fit(Xtr)
        centers = mdl.means_
        predict = mdl.predict
        labels_tr = predict(Xtr)
    else:  # agglo
        # mdl = AgglomerativeClustering(n_clusters=n_clusters).fit(Xtr)
        mdl = model.fit(Xtr)
        labels_tr = mdl.labels_
        # Build one centroid per cluster that actually exists. Ranging over a
        # requested n_clusters instead averages empty slices into NaN rows when the
        # fitted model has fewer clusters than the number passed in.
        present = np.unique(labels_tr)
        centers = np.vstack([Xtr[labels_tr == i].mean(axis=0) for i in present])

        def predict(Xv):
            try:
                from scipy.spatial.distance import cdist
            except Exception:
                # naive nearest-center w/o scipy
                def _cdist(A, B):
                    # squared Euclidean
                    AA = (A**2).sum(1)[:, None]
                    BB = (B**2).sum(1)[None, :]
                    return np.sqrt(AA + BB - 2 * A @ B.T)

                D = _cdist(Xv, centers)
                return D.argmin(axis=1)
            else:
                return cdist(Xv, centers).argmin(axis=1)

    return mdl, predict, labels_tr, centers


def _assign_labels_by_centers(Xv: np.ndarray, centers: np.ndarray | None) -> np.ndarray:
    """Nearest cluster for each row when the estimator has no .predict (e.g. SpectralClustering)."""
    if centers is None or len(Xv) == 0:
        return np.array([], dtype=int)
    C = np.asarray(centers, dtype=float)
    if C.ndim != 2 or C.shape[0] < 1:
        return np.array([], dtype=int)
    try:
        from scipy.spatial.distance import cdist

        return cdist(Xv, C).argmin(axis=1).astype(int)
    except Exception:
        AA = (Xv.astype(float) ** 2).sum(1)[:, None]
        BB = (C**2).sum(1)[None, :]
        D = np.sqrt(np.maximum(AA + BB - 2 * Xv.astype(float) @ C.T, 0.0))
        return D.argmin(axis=1).astype(int)


# Advanced-option keys a cluster train_* function forwards from its **kwargs.
CLUSTERING_OPTION_KEYS = ("outlier_method", "outlier_action", "progress_tracker", "modeling_mode")


def clustering_options(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pick the advanced options out of a train_* function's **kwargs.

    The remaining keys are estimator hyperparameters and stay with the caller.
    """
    return {k: kwargs[k] for k in CLUSTERING_OPTION_KEYS if k in kwargs}


def run_clustering(
    model,
    model_name,
    train_data,
    X,
    units,
    X_scaler_type,
    seed,
    sigfig,
    quantileBinDict,
    useTransformer,
    transformer_cols,
    test_size,
    k_min,
    k_max,
    outlier_method="none",
    outlier_action="remove",
    progress_tracker=None,
    modeling_mode="simple",
    **kwargs,
) -> tuple[Any, ...]:
    """Fit a clusterer and score it. quantileBinDict is accepted for call-site
    symmetry with the supervised pipelines; clustering has no target to bin.

    Returns a 12-tuple: the 11 legacy items followed by ``outlier_info`` (None
    when outlier_method is 'none').
    """
    if progress_tracker:
        progress_tracker.update_stage("model_training", "running", 5, f"Preparing data for {model_name}...")

    # Build modeling frame (no target)
    # Ensure train_data is a DataFrame
    if not isinstance(train_data, pd.DataFrame):
        if isinstance(train_data, dict):
            train_data = pd.DataFrame(train_data)
        else:
            raise TypeError(f"train_data must be a DataFrame, got {type(train_data)}")

    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        if isinstance(X, dict):
            X = pd.DataFrame(X)
        elif hasattr(X, "__array__"):
            # If it's array-like, try to convert to DataFrame
            X = pd.DataFrame(X)
        else:
            raise TypeError(f"X must be a DataFrame, got {type(X)}")

    # nump = cfg.numeric_predictors or df.select_dtypes(include=np.number).columns.tolist()
    # catp = cfg.categorical_predictors or [c for c in df.select_dtypes(exclude=np.number).columns.tolist() if
    # df[c].nunique(dropna=True) <= 50]
    # textp = cfg.text_cols or []
    nump = X.select_dtypes(include=np.number).columns.tolist()
    catp = []
    if useTransformer == "Yes":
        catp = transformer_cols
    textp = []

    indicator_cols = nump + catp + textp
    # Ensure we're selecting columns from a DataFrame, not a dict
    existing_cols = []
    if isinstance(train_data, pd.DataFrame):
        # Filter to only existing columns
        existing_cols = [col for col in indicator_cols if col in train_data.columns]
        if not existing_cols:
            raise ValueError(
                f"None of the indicator columns {indicator_cols} exist in train_data. Available columns: "
                f"{list(train_data.columns)}"
            )
        # Use .loc to ensure we get a DataFrame, not a Series or dict
        if len(existing_cols) == 1:
            # Single column - still return as DataFrame
            ddf = train_data.loc[:, existing_cols].dropna().copy()
        else:
            ddf = train_data[existing_cols].dropna().copy()

        # Double-check ddf is a DataFrame
        if not isinstance(ddf, pd.DataFrame):
            # If somehow it's not a DataFrame, force conversion
            ddf = (
                pd.DataFrame(ddf) if hasattr(ddf, "__array__") or isinstance(ddf, (dict, list)) else pd.DataFrame([ddf])
            )
    else:
        raise TypeError(f"train_data must be a DataFrame after type check, got {type(train_data)}")

    # Ensure ddf is a DataFrame
    if not isinstance(ddf, pd.DataFrame):
        raise TypeError(
            f"ddf must be a DataFrame after column selection, got {type(ddf)}. indicator_cols={indicator_cols}, "
            f"existing_cols={existing_cols}"
        )

    # Optional split to assess generalization/stability
    DO_SPLIT = True
    if DO_SPLIT:
        X_train_raw, X_test_raw = train_test_split(ddf, test_size=test_size, random_state=seed)
        # Ensure both are DataFrames
        if not isinstance(X_train_raw, pd.DataFrame):
            X_train_raw = (
                pd.DataFrame(X_train_raw) if hasattr(X_train_raw, "__array__") else pd.DataFrame([X_train_raw])
            )
        if not isinstance(X_test_raw, pd.DataFrame):
            X_test_raw = pd.DataFrame(X_test_raw) if hasattr(X_test_raw, "__array__") else pd.DataFrame([X_test_raw])
    else:
        X_train_raw = ddf.copy()
        X_test_raw = ddf.iloc[0:0].copy() if isinstance(ddf, pd.DataFrame) else pd.DataFrame(columns=existing_cols)

    # Transform
    preproc = make_preprocessor(numeric_cols=nump, categorical_cols=catp, cat_mode="onehot")
    X_train_t = pd.DataFrame(preproc.fit_transform(X_train_raw), index=X_train_raw.index)
    # Ensure X_test_t is always a DataFrame, even if empty
    if len(X_test_raw) > 0:
        X_test_t = pd.DataFrame(preproc.transform(X_test_raw), index=X_test_raw.index)
    else:
        # Create empty DataFrame with same structure as X_train_t
        X_test_t = pd.DataFrame(columns=X_train_t.columns, dtype=X_train_t.dtypes)
    feat_names = get_feature_names(preproc, X_train_raw)
    if len(feat_names) == X_train_t.shape[1]:
        X_train_t.columns = feat_names
        if len(X_test_t) > 0:
            X_test_t.columns = feat_names

    # Scale X only
    # Ensure X_test_t is a DataFrame before scaling
    if not isinstance(X_test_t, pd.DataFrame):
        X_test_t = (
            pd.DataFrame(X_test_t)
            if X_test_t is not None
            else pd.DataFrame(columns=X_train_t.columns, dtype=X_train_t.dtypes)
        )
    X_train_s, X_test_s, _, _, X_scaler, _ = _scale_pairs(X_train_t, X_test_t, None, None, X_scaler_type, "none")

    # Outlier handling on X only; clustering has no target, so this is the whole step.
    outlier_info = None
    if outlier_method and outlier_method != "none":
        if progress_tracker:
            progress_tracker.update_stage(
                "outlier_handling", "running", 0, f"Detecting outliers using {outlier_method}..."
            )
        train_index_before = X_train_s.index
        X_train_s, X_test_s, outlier_mask = apply_outlier_handling(
            X_train_s, X_test_s, None, outlier_method, outlier_action, seed=seed
        )
        outlier_mask = np.asarray(outlier_mask, dtype=bool)
        n_outliers = int((~outlier_mask).sum())
        outlier_info = {
            "method": outlier_method,
            "action": outlier_action,
            "n_outliers": n_outliers,
            "original_samples": int(len(outlier_mask)),
            "remaining_samples": int(outlier_mask.sum()),
        }
        if outlier_action == "remove" and n_outliers > 0:
            outlier_info["removed_row_indices"] = train_index_before[~outlier_mask].tolist()
        if progress_tracker:
            progress_tracker.update_stage(
                "outlier_handling",
                "completed",
                100,
                f"Outlier handling complete ({n_outliers} outliers "
                f"{'removed' if outlier_action == 'remove' else 'capped'})",
            )
        if len(X_train_s) < 2:
            raise ValueError(
                "Outlier removal left fewer than 2 training rows; choose a less aggressive method or 'cap'."
            )

    if progress_tracker:
        progress_tracker.update_stage("model_training", "running", 30, f"Fitting {model_name}...")

    # Choose k & fit
    sil_scores = {}
    centers = None
    # Models that don't follow the standard kmeans/gmm/agglo pattern
    special_models = {
        "dbscan",
        "birch",
        "spectral",
        "hdbscan",
        "affinity_propagation",
        "meanshift",
        "optics",
        "minibatch_kmeans",
        "bisecting_kmeans",
    }
    if model_name in special_models:
        logger.debug(f"Using clustering model: {model_name}")
        logger.debug(f"Model parameters: {model}")
        if model_name == "spectral":
            labels_train = model.fit_predict(X_train_s.values)
            predict_fn = None
            centers = None
        elif model_name in {"minibatch_kmeans", "bisecting_kmeans"}:
            # These models have cluster_centers_ and predict method
            model.fit(X_train_s.values)
            labels_train = model.labels_
            centers = model.cluster_centers_
            predict_fn = model.predict
        else:
            model.fit(X_train_s.values)
            labels_train = getattr(model, "labels_", None)
            if labels_train is None:
                if hasattr(model, "predict"):
                    labels_train = model.predict(X_train_s.values)
                elif hasattr(model, "fit_predict"):
                    labels_train = model.fit_predict(X_train_s.values)
                else:
                    # Fallback: assign all to one cluster
                    labels_train = np.zeros(X_train_s.shape[0], dtype=int)
            predict_fn = model.predict if hasattr(model, "predict") else None
            # Try to get centers for models that have them
            if hasattr(model, "cluster_centers_"):
                centers = model.cluster_centers_
            elif hasattr(model, "means_"):
                centers = model.means_
            else:
                centers = None

        # Determine number of clusters
        if model_name in {"dbscan", "hdbscan", "optics"}:
            unique_labels = set(labels_train) - {-1}  # -1 is noise/outlier label
            best_k = len(unique_labels) if unique_labels else 1
        elif model_name in {"affinity_propagation", "meanshift"}:
            # These models determine clusters automatically
            unique_labels = set(labels_train)
            best_k = len(unique_labels) if unique_labels else 1
        elif model_name in {"minibatch_kmeans", "bisecting_kmeans"}:
            # These models have n_clusters parameter
            best_k = getattr(model, "n_clusters", len(np.unique(labels_train)))
        else:
            # birch, spectral
            best_k = getattr(model, "n_clusters", None) or len(np.unique(labels_train))
    else:
        # Handle standard models (kmeans, gmm, agglo)
        algo = "gmm" if model_name == "gmm" else model_name
        suggested_k, sil_scores = _choose_k_silhouette(X_train_s.values, k_min, k_max, algo, seed)
        logger.debug(f"Using clustering model: {model_name}")
        logger.debug(f"Model parameters: {model}")
        model, predict_fn, labels_train, centers = _fit_clusterer(model, algo, X_train_s.values, suggested_k)
        # The sweep is advisory. What gets reported is the number of clusters the
        # fitted model produced, which is what every downstream chart is drawn from.
        best_k = int(len(np.unique(labels_train)))
        if best_k != suggested_k:
            logger.info(
                "Silhouette sweep suggested k=%s; model was fit with k=%s and is reported as such.",
                suggested_k,
                best_k,
            )

    logger.debug("Computing cluster labels for test set")
    # Ensure X_test_s is a DataFrame before accessing .values
    if not isinstance(X_test_s, pd.DataFrame):
        if X_test_s is not None:
            X_test_s = pd.DataFrame(X_test_s)
        else:
            X_test_s = pd.DataFrame(columns=X_train_s.columns, dtype=X_train_s.dtypes)

    if len(X_test_s) > 0:
        if predict_fn is not None:
            labels_test = predict_fn(X_test_s.values)
        else:
            labels_test = np.array([], dtype=int)
        if len(labels_test) != len(X_test_s):
            labels_test = _assign_labels_by_centers(X_test_s.values, centers)
    else:
        labels_test = np.array([], dtype=int)
    logger.debug("Computing cluster scores")

    # Scores
    def _cluster_scores(Xa, labels):
        if Xa is None or len(labels) == 0 or len(np.unique(labels)) < 2:
            return {}
        return {
            "silhouette": silhouette_score(Xa, labels),
            "calinski_harabasz": calinski_harabasz_score(Xa, labels),
            "davies_bouldin": davies_bouldin_score(Xa, labels),
        }

    clusters = {
        "labels_train": labels_train,
        "labels_test": labels_test,
        "centers": centers,
        "best_k": best_k,
        "silhouette_grid": sil_scores,
    }

    def _ensure_scores(scores):
        if scores:
            return scores
        return {
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "davies_bouldin": float("nan"),
        }

    train_scores = _ensure_scores(_cluster_scores(X_train_s.values, labels_train))
    # Ensure X_test_s is a DataFrame before accessing .values
    if isinstance(X_test_s, pd.DataFrame) and len(labels_test) > 0:
        test_scores = _ensure_scores(_cluster_scores(X_test_s.values, labels_test))
    else:
        test_scores = _ensure_scores({})
    logger.debug("Cluster scores computed")

    art = {
        "type": "cluster",
        "preprocessor": preproc,
        "X_scaler": X_scaler,
        "model": model,
        "feature_names": list(X_train_t.columns),
        "splits": {"X_train": X_train_s, "X_test": X_test_s},
        "clusters": {
            "labels_train": labels_train,
            "labels_test": labels_test,
            "centers": centers,
            "best_k": best_k,
            "silhouette_grid": sil_scores,
        },
        "metrics": {"train": train_scores, "test": test_scores},
    }

    pdf_pages = PdfPages(VIS_DIR / "visualizations.pdf")
    export_plots(art, pdf_pages)

    pdf_pages.close()
    logger.info("Clustering pipeline completed")
    logger.debug(f"Train scores: {train_scores}")
    logger.debug(f"Test scores: {test_scores}")

    quantileBinResults = ""
    try:
        model._digiterra_problem_type = "cluster"
        model._digiterra_preprocessor = preproc
        model._digiterra_raw_features = X_train_raw.columns.tolist()
        model._digiterra_feature_names = X_train_t.columns.tolist()
        model._digiterra_model_features = X_train_s.columns.tolist()
    except Exception:
        logger.debug("Could not attach inference artifacts to clustering model", exc_info=True)

    return (
        train_scores,
        test_scores,
        model.get_params(),
        {"X_train": X_train_s.shape, "X_test": X_test_s.shape},
        model,
        X_scaler,
        quantileBinResults,
        X_train_raw.columns.tolist(),
        clusters["best_k"],
        clusters["centers"],
        clusters["silhouette_grid"],
        outlier_info,
    )
