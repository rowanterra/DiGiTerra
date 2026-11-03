"""DiGiTerra's fit equals plain scikit-learn's fit on the machine that runs the test.

This is the reproducibility standard adopted on 2026-09-25: on any machine,
a DiGiTerra run gives exactly what the official scikit-learn gives when called
directly on that machine with the same versions, data and seed. Results are not
compared between machines, because the official scikit-learn builds for x86_64
and arm64 resolve tied tree splits differently
(docs/tree_platform_difference_2026-09-25).

The test wraps the estimator's fit to capture the training matrix and target the
pipeline passes, fits a fresh estimator of the same class with the same parameters
on them, and requires identical tree node arrays and identical predictions.
"""

import uuid

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from tests.unit.test_bundle import _train_in_session

CASES = {
    "RF": RandomForestRegressor,
    "ExtraTrees": ExtraTreesRegressor,
    "Bagging": BaggingRegressor,
    "DecisionTree": DecisionTreeRegressor,
}


def _trees(model):
    return getattr(model, "estimators_", None) or [model]


@pytest.mark.parametrize("name", list(CASES))
def test_a_digiterra_tree_model_is_the_scikit_learn_fit_on_this_machine(name, monkeypatch):
    cls = CASES[name]
    seen = {}
    original = cls.fit

    def fit(self, X, y, *args, **kwargs):
        seen["X"], seen["y"], seen["model"] = np.array(X, dtype=float), np.array(y, dtype=float), self
        seen["extra"] = (args, kwargs)
        return original(self, X, y, *args, **kwargs)

    monkeypatch.setattr(cls, "fit", fit)
    _train_in_session(uuid.uuid4().hex, {"models": name})
    ours = seen["model"]
    args, kwargs = seen["extra"]
    monkeypatch.setattr(cls, "fit", original)
    plain = clone(ours).fit(seen["X"], seen["y"], *args, **kwargs)
    for a, b in zip(_trees(ours), _trees(plain), strict=True):
        for field in ("feature", "threshold", "children_left", "children_right", "value"):
            assert np.array_equal(getattr(a.tree_, field), getattr(b.tree_, field)), (name, field)
    assert np.array_equal(ours.predict(seen["X"]), plain.predict(seen["X"]))
