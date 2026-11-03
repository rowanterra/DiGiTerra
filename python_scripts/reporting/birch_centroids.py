"""A data-only stand-in for a fitted Birch model, so a Birch run can be written with skops.

skops refuses the Birch type itself: the fitted model holds a tree of _CFNode and
_CFSubcluster objects linked in both directions. Birch.predict reads none of that
tree. It assigns each row the label of the nearest subcluster centroid, using
subcluster_centers_, the squared norms of those centers and subcluster_labels_.
BirchCentroids keeps exactly those arrays, the fitted labels_ and the Birch
parameters, and predicts with the same call Birch makes, so a loaded model returns
the labels the fitted Birch returned. What it cannot do is partial_fit: the
subcluster tree is not kept. fit exists because scikit-learn's checks require it; it
fits a Birch with the same parameters and keeps that model's prediction state.
"""

from __future__ import annotations

import numpy as np
from sklearn import config_context
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils.validation import check_is_fitted, validate_data


class BirchCentroids(ClusterMixin, BaseEstimator):
    """The prediction state of a fitted Birch model, as plain arrays."""

    def __init__(self, *, threshold=0.5, branching_factor=50, n_clusters=3, compute_labels=True):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.compute_labels = compute_labels

    @classmethod
    def from_birch(cls, birch) -> BirchCentroids:
        """Copy the parameters and the prediction arrays of a fitted Birch."""
        params = {k: v for k, v in birch.get_params(deep=False).items() if k in cls().get_params(deep=False)}
        out = cls(**params)
        out.subcluster_centers_ = np.asarray(birch.subcluster_centers_)
        out.subcluster_labels_ = np.asarray(birch.subcluster_labels_)
        out._subcluster_norms = np.asarray(birch._subcluster_norms)
        out.n_features_in_ = birch.n_features_in_
        if hasattr(birch, "feature_names_in_"):
            out.feature_names_in_ = np.asarray(birch.feature_names_in_)
        if hasattr(birch, "labels_"):
            out.labels_ = np.asarray(birch.labels_)
        return out

    def fit(self, X, y=None):
        """Fit a Birch with these parameters and keep its prediction state."""
        from sklearn.cluster import Birch

        fitted = BirchCentroids.from_birch(Birch(**self.get_params()).fit(X))
        self.__dict__.update(fitted.__dict__)
        return self

    def predict(self, X):
        """The label of the nearest subcluster centroid, as Birch.predict computes it."""
        check_is_fitted(self, "subcluster_centers_")
        X = validate_data(self, X, accept_sparse="csr", reset=False)
        with config_context(assume_finite=True):
            argmin = pairwise_distances_argmin(
                X, self.subcluster_centers_, metric_kwargs={"Y_norm_squared": self._subcluster_norms}
            )
        return self.subcluster_labels_[argmin]
