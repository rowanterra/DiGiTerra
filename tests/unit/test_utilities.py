"""Unit tests for python_scripts.preprocessing.utilities."""
import pytest
from python_scripts.preprocessing.utilities import sort_class_labels_numeric_bins


def test_sort_class_labels_numeric_bins_empty():
    assert sort_class_labels_numeric_bins([]) == []
    assert sort_class_labels_numeric_bins(None) == []


def test_sort_class_labels_numeric_bins_plain_numbers():
    assert sort_class_labels_numeric_bins([3, 1, 2]) == [1, 2, 3]
    assert sort_class_labels_numeric_bins(["10", "2", "1"]) == ["1", "2", "10"]


def test_sort_class_labels_numeric_bins_ranges():
    # Ranges ordered by lower bound
    labels = ["50-150", "<50", ">500", "150-500"]
    out = sort_class_labels_numeric_bins(labels)
    assert out[0] == "<50"
    assert out[-1] == ">500"
    assert "50-150" in out and "150-500" in out


def test_sort_class_labels_numeric_bins_categories():
    # No numbers -> alphabetical at end
    labels = ["B", "A", "C"]
    out = sort_class_labels_numeric_bins(labels)
    assert out == ["A", "B", "C"]
