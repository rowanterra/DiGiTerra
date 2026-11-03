"""Platform notes for reruns, added 2026-09-25.

RF, ExtraTrees and Bagging retrained on the Mac (arm64) and on Linux (x86_64) gave
different trees from bitwise-identical inputs, seeds and package versions, because
scikit-learn's compiled splitter resolves exact ties by the last bits of the
arithmetic (docs/tree_platform_difference_2026-09-25). scripts/rerun_bundle.py
named package drift but not a platform change, so such a rerun reported changed
metrics with no reason given.
"""

import platform

import pytest

from python_scripts.reporting import bundle
from scripts.compare_roundtrip_records import classify, max_relative_difference


@pytest.mark.parametrize(
    "text, expected",
    [
        ("macOS-15.5-arm64-arm-64bit", ("Darwin", "arm64")),
        ("macOS-14.6.1-x86_64-i386-64bit", ("Darwin", "x86_64")),
        ("Linux-6.18.44-fc-v42-x86_64-with-glibc2.39", ("Linux", "x86_64")),
        ("Linux-6.8.0-aarch64-with-glibc2.35", ("Linux", "arm64")),
        ("Windows-10-10.0.19045-SP0", ("Windows", None)),
    ],
)
def test_bundles_before_2026_09_25_give_their_platform_from_the_string(text, expected):
    assert bundle._recorded_platform({"environment": {"platform": text}}) == expected


def test_new_manifests_record_system_and_processor():
    rec = bundle.platform_record()
    assert rec["system"] == platform.system() and rec["machine"] and rec["platform"]


def _manifest(machine: str, estimator: str) -> dict:
    return {
        "model": {"estimator": estimator},
        "environment": {"system": platform.system(), "machine": machine, "platform": "x"},
    }


def test_no_note_on_the_same_platform():
    here = bundle.platform_record()["machine"]
    assert bundle.platform_change(_manifest(here, "RandomForestRegressor")) is None


def test_a_tree_model_from_another_processor_gets_the_tied_split_note():
    other = "x86_64" if bundle.platform_record()["machine"] != "x86_64" else "arm64"
    note = bundle.platform_change(_manifest(other, "RandomForestRegressor"))
    assert note and f"processor {other} to" in note and "tied splits" in note
    plain = bundle.platform_change(_manifest(other, "Ridge"))
    assert plain and "tied splits" not in plain and "last digits" in plain


def test_the_model_card_states_what_reproduces():
    manifest = {"model": {"name": "RF", "estimator": "RandomForestRegressor"}, "environment": {"packages": {}}}
    html = bundle.render_model_card(manifest)
    assert "same type of machine" in html and "tied splits" in html
    manifest["model"]["estimator"] = "Ridge"
    assert "tied splits" not in bundle.render_model_card(manifest)


def test_round_trip_records_are_classed_by_size_and_model():
    tree = {"estimator": "RandomForestRegressor", "output_before": [1.0, 2.0]}
    other_tree = {"estimator": "RandomForestRegressor", "output_before": [1.0, 2.1]}
    ridge = {"estimator": "Ridge", "output_before": [1.0, 2.0]}
    near = {"estimator": "Ridge", "output_before": [1.0, 2.0 * (1 + 1e-14)]}
    far = {"estimator": "Ridge", "output_before": [1.0, 2.1]}
    assert classify(ridge, ridge, True, 1e-9)[0] == "identical"
    assert classify(ridge, near, True, 1e-9)[0] == "rounding"
    assert classify(tree, other_tree, True, 1e-9)[0] == "tree, platform"
    assert classify(tree, other_tree, False, 1e-9)[0] == "unexplained"
    assert classify(ridge, far, True, 1e-9)[0] == "unexplained"
    assert max_relative_difference(["a"], ["b"]) is None
