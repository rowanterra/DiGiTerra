"""Unit tests for python_scripts.model_registry."""
import pytest
from python_scripts.model_registry import MODEL_REGISTRY, get_model_kwargs, CLUSTER_MODELS


def test_registry_has_expected_models():
    assert "Linear" in MODEL_REGISTRY
    assert "Ridge" in MODEL_REGISTRY
    assert "kmeans" in MODEL_REGISTRY
    assert "Logistic_classifier" in MODEL_REGISTRY
    assert len(MODEL_REGISTRY) >= 69


def test_registry_problem_types():
    _, pt = MODEL_REGISTRY["Linear"]
    assert pt == "regression"
    _, pt = MODEL_REGISTRY["Logistic_classifier"]
    assert pt == "classification"
    _, pt = MODEL_REGISTRY["kmeans"]
    assert pt == "cluster"


def test_cluster_models_frozenset():
    assert "kmeans" in CLUSTER_MODELS
    assert "gmm" in CLUSTER_MODELS
    assert "Linear" not in CLUSTER_MODELS
    assert len(CLUSTER_MODELS) >= 12


def test_get_model_kwargs_ridge_simple():
    kw = get_model_kwargs("Ridge", {"alpha": "1.0"}, False, 42)
    assert "alpha" in kw
    assert kw.get("RidgeRandomState") == 42


def test_get_model_kwargs_kmeans_simple():
    kw = get_model_kwargs("kmeans", {"n_clusters": "5"}, False, 42)
    assert "n_clusters" in kw


def test_get_model_kwargs_linear_empty():
    kw = get_model_kwargs("Linear", {}, False, 42)
    assert kw == {}
