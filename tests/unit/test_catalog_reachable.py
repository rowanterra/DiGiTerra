"""Every estimator the sweep trained must be reachable from a screen.

An estimator can be in the registry, train correctly, and be offered by no select
at all. LassoLars was in that state: trainable, exercised by the sweep, present in
the hidden legacy selects, and absent from every select a person can reach. The
registry count and the interface count disagreed by one and no test noticed.

The expected set is read from docs/every_model_sweep_2026-09-03.json rather than
written out here, so an estimator added to the registry and swept, but never put on
a screen, fails this instead of reaching a reviewer.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "docs" / "every_model_sweep_2026-09-03.json"
TEMPLATE = ROOT / "templates" / "index.html"

# The selects a person can reach. The header* selects sit inside #headerModelSelection,
# which every code path hides, so an estimator offered only there is not reachable.
LIVE_SELECTS = (
    "simpleNModels",
    "simpleClassModels",
    "simpleClModels",
    "advancedNModels",
    "advancedClassModels",
    "advancedClModels",
    "automlNModels",
    "automlClassModels",
    "automlClModels",
)


def _options(select_id: str) -> set[str]:
    html = TEMPLATE.read_text(encoding="utf-8")
    m = re.search(rf'<select[^>]*id="{re.escape(select_id)}"[^>]*>(.*?)</select>', html, re.S)
    assert m, f"no select with id {select_id}"
    return {v for v in re.findall(r'<option[^>]*value="([^"]*)"', m.group(1)) if v}


@pytest.fixture(scope="module")
def offered() -> set[str]:
    seen: set[str] = set()
    for sid in LIVE_SELECTS:
        seen |= _options(sid)
    return seen


@pytest.fixture(scope="module")
def trained() -> list[dict]:
    return [r for r in json.loads(SWEEP.read_text(encoding="utf-8")) if r.get("status") == "pass"]


def test_every_trained_estimator_is_offered_somewhere(offered, trained):
    missing = sorted({r["model"] for r in trained} - offered)
    assert not missing, (
        f"{len(missing)} estimator(s) train but no select offers them: {', '.join(missing)}. "
        "Add each to the select for its task, or drop it from the registry."
    )


def test_every_offered_estimator_is_one_the_sweep_trained(offered, trained):
    """The other direction: a select must not offer something that does not train."""
    swept = {r["model"] for r in json.loads(SWEEP.read_text(encoding="utf-8"))}
    unknown = sorted(offered - swept)
    assert not unknown, f"selects offer estimators the sweep does not know: {', '.join(unknown)}"


def test_the_counts_the_documentation_states_are_the_counts_on_screen(offered, trained):
    """67 trainable, and 67 reachable, so the paper and the interface agree."""
    by_task = {}
    for r in trained:
        by_task.setdefault(r["task"], set()).add(r["model"])
    assert set(by_task) == {"regression", "classification", "cluster"}, sorted(by_task)
    assert len(by_task["regression"]) == 32, sorted(by_task["regression"])
    assert len(by_task["classification"]) == 23
    assert len(by_task["cluster"]) == 12
    assert sum(len(v) for v in by_task.values()) == 67
    assert len({r["model"] for r in trained} & offered) == 67


@pytest.mark.parametrize(
    "select_id, task",
    [
        ("simpleNModels", "regression"),
        ("advancedNModels", "regression"),
        ("automlNModels", "regression"),
        ("simpleClassModels", "classification"),
        ("advancedClassModels", "classification"),
        ("automlClassModels", "classification"),
        ("simpleClModels", "cluster"),
        ("advancedClModels", "cluster"),
        ("automlClModels", "cluster"),
    ],
)
def test_each_mode_offers_the_whole_task(select_id, task, trained):
    """A mode must not quietly offer fewer estimators than another mode for the same task."""
    expected = {r["model"] for r in trained if r["task"] == task}
    missing = sorted(expected - _options(select_id))
    assert not missing, f"{select_id} omits {', '.join(missing)}"
