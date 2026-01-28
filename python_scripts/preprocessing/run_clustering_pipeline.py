
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
import re
import math
import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    precision_recall_curve, average_precision_score, roc_curve, auc, roc_auc_score
)

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture


from matplotlib.backends.backend_pdf import PdfPages
from python_scripts.preprocessing.visualize_predictions import visualize_predictions
from python_scripts.plotting.plot_shap_summary_graphic import plot_shap_summary
from python_scripts.preprocessing.utilites import make_preprocessor
from python_scripts.preprocessing.utilites import get_feature_names
from python_scripts.preprocessing.utilites import _scale_pairs
from python_scripts.preprocessing.utilites import export_plots

from python_scripts.config import VIS_DIR


def _choose_k_silhouette(X: np.ndarray, k_min=2, k_max=10, algo="kmeans"):
    scores = {}
    for k in range(max(k_min, 2), max(k_max, 2) + 1):
        if algo == "kmeans":
            mdl = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
            labels = mdl.labels_
        elif algo == "gmm":
            mdl = GaussianMixture(n_components=k, covariance_type="full", random_state=42).fit(X)
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
        #mdl = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(Xtr)
        mdl = model.fit(Xtr)
        centers = mdl.cluster_centers_
        predict = mdl.predict
        labels_tr = mdl.labels_
    elif algo == "gmm":
        #mdl = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=42).fit(Xtr)
        mdl = model.fit(Xtr)
        centers = mdl.means_
        predict = mdl.predict
        labels_tr = predict(Xtr)
    else:  # agglo
        #mdl = AgglomerativeClustering(n_clusters=n_clusters).fit(Xtr)
        mdl = model.fit(Xtr)
        labels_tr = mdl.labels_
        centers = np.vstack([Xtr[labels_tr==i].mean(axis=0) for i in range(n_clusters)])
        def predict(Xv):
            try:
                from scipy.spatial.distance import cdist
            except Exception:
                # naive nearest-center w/o scipy
                def _cdist(A, B):
                    # squared Euclidean
                    AA = (A**2).sum(1)[:,None]
                    BB = (B**2).sum(1)[None,:]
                    return np.sqrt(AA + BB - 2*A@B.T)
                D = _cdist(Xv, centers)
                return D.argmin(axis=1)
            else:
                return cdist(Xv, centers).argmin(axis=1)
    return mdl, predict, labels_tr, centers


def run_clustering(model, model_name,
                    train_data,
                    X,
                    units, X_scaler_type,
                    seed, sigfig, quantileBinDict, useTransformer, transformer_cols, test_size, k_min, k_max) -> Dict[str, Any]:
    

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
        elif hasattr(X, '__array__'):
            # If it's array-like, try to convert to DataFrame
            X = pd.DataFrame(X)
        else:
            raise TypeError(f"X must be a DataFrame, got {type(X)}")
    
    # nump = cfg.numeric_predictors or df.select_dtypes(include=np.number).columns.tolist()
    # catp = cfg.categorical_predictors or [c for c in df.select_dtypes(exclude=np.number).columns.tolist() if df[c].nunique(dropna=True) <= 50]
    # textp = cfg.text_cols or []
    nump = X.select_dtypes(include=np.number).columns.tolist()
    catp = []
    if useTransformer == 'Yes':
        catp=transformer_cols.tolist()
    textp=[]

    indicator_cols = nump + catp + textp
    # Ensure we're selecting columns from a DataFrame, not a dict
    existing_cols = []
    if isinstance(train_data, pd.DataFrame):
        # Filter to only existing columns
        existing_cols = [col for col in indicator_cols if col in train_data.columns]
        if not existing_cols:
            raise ValueError(f"None of the indicator columns {indicator_cols} exist in train_data. Available columns: {list(train_data.columns)}")
        # Use .loc to ensure we get a DataFrame, not a Series or dict
        if len(existing_cols) == 1:
            # Single column - still return as DataFrame
            ddf = train_data.loc[:, existing_cols].dropna().copy()
        else:
            ddf = train_data[existing_cols].dropna().copy()
        
        # Double-check ddf is a DataFrame
        if not isinstance(ddf, pd.DataFrame):
            # If somehow it's not a DataFrame, force conversion
            ddf = pd.DataFrame(ddf) if hasattr(ddf, '__array__') or isinstance(ddf, (dict, list)) else pd.DataFrame([ddf])
    else:
        raise TypeError(f"train_data must be a DataFrame after type check, got {type(train_data)}")
    
    # Ensure ddf is a DataFrame
    if not isinstance(ddf, pd.DataFrame):
        raise TypeError(f"ddf must be a DataFrame after column selection, got {type(ddf)}. indicator_cols={indicator_cols}, existing_cols={existing_cols}")

    # Optional split to assess generalization/stability
    DO_SPLIT = True
    if DO_SPLIT:
        X_train_raw, X_test_raw = train_test_split(ddf, test_size=test_size, random_state=seed)
        # Ensure both are DataFrames
        if not isinstance(X_train_raw, pd.DataFrame):
            X_train_raw = pd.DataFrame(X_train_raw) if hasattr(X_train_raw, '__array__') else pd.DataFrame([X_train_raw])
        if not isinstance(X_test_raw, pd.DataFrame):
            X_test_raw = pd.DataFrame(X_test_raw) if hasattr(X_test_raw, '__array__') else pd.DataFrame([X_test_raw])
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
        X_test_t = pd.DataFrame(X_test_t) if X_test_t is not None else pd.DataFrame(columns=X_train_t.columns, dtype=X_train_t.dtypes)
    X_train_s, X_test_s, _, _, X_scaler, _ = _scale_pairs(X_train_t, X_test_t, None, None, X_scaler_type, "none")

    # Choose k & fit
    sil_scores = {}
    centers = None
    # Models that don't follow the standard kmeans/gmm/agglo pattern
    special_models = {"dbscan", "birch", "spectral", "hdbscan", "affinity_propagation", "meanshift", "optics", "minibatch_kmeans", "bisecting_kmeans"}
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
        if model_name == "gmm":
            # GMM uses different parameter name (n_components instead of n_clusters)
            best_k, sil_scores = _choose_k_silhouette(X_train_s.values, k_min, k_max, "gmm")
            logger.debug(f"Using clustering model: {model_name}")
            logger.debug(f"Model parameters: {model}")
            model, predict_fn, labels_train, centers = _fit_clusterer(model, "gmm", X_train_s.values, best_k)
        else:
            best_k, sil_scores = _choose_k_silhouette(X_train_s.values, k_min, k_max, model_name)
            logger.debug(f"Using clustering model: {model_name}")
            logger.debug(f"Model parameters: {model}")
            model, predict_fn, labels_train, centers = _fit_clusterer(model, model_name, X_train_s.values, best_k)

    logger.debug("Computing cluster labels for test set")
    # Ensure X_test_s is a DataFrame before accessing .values
    if not isinstance(X_test_s, pd.DataFrame):
        if X_test_s is not None:
            X_test_s = pd.DataFrame(X_test_s)
        else:
            X_test_s = pd.DataFrame(columns=X_train_s.columns, dtype=X_train_s.dtypes)
    
    if predict_fn is not None and len(X_test_s) > 0:
        labels_test = predict_fn(X_test_s.values)
    else:
        labels_test = np.array([])
    logger.debug("Computing cluster scores")
    # Scores
    def _cluster_scores(Xa, labels):
        if Xa is None or len(labels)==0 or len(np.unique(labels))<2:
            return {}
        return {
            "silhouette": silhouette_score(Xa, labels),
            "calinski_harabasz": calinski_harabasz_score(Xa, labels),
            "davies_bouldin": davies_bouldin_score(Xa, labels)
        }
    clusters = {
            "labels_train": labels_train,
            "labels_test": labels_test,
            "centers": centers,
            "best_k": best_k,
            "silhouette_grid": sil_scores
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

    art= {
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
            "silhouette_grid": sil_scores
        },
        "metrics": {"train": train_scores, "test": test_scores}
    }

    pdf_pages = PdfPages(VIS_DIR / "visualizations.pdf")
    export_plots(
       art, pdf_pages)
    
    pdf_pages.close()
    logger.info("Clustering pipeline completed")
    logger.debug(f"Train scores: {train_scores}")
    logger.debug(f"Test scores: {test_scores}")
    
    quantileBinResults = ''
    return train_scores, test_scores, model.get_params(), {
                'X_train': X_train_s.shape,
                'X_test': X_test_s.shape
            }, model, X_scaler, quantileBinResults, X_train_s.columns.tolist(), clusters['best_k'], clusters['centers'], clusters['silhouette_grid']
