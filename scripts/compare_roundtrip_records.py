"""Compare two round-trip records, for example one from Linux and one from a Mac.

Reads two JSON files written by scripts/check_artifact_roundtrip.py and compares
output_before model by model. A model is "identical" when every value is equal,
"rounding" when the largest relative difference is at most --rounding (1e-9 by
default), and "differs" otherwise. A tree model that differs between records made
on different processor types is marked "tree, platform": a tree grown to full depth
resolves tied splits by the last bits of the arithmetic, which differ between
x86_64 and arm64 (docs/tree_platform_difference_2026-09-25). Any other difference
is "unexplained" and sets exit code 1.

Records written before 2026-09-25 carry no processor type; pass it with
--machine-a and --machine-b.

Usage:
    python scripts/compare_roundtrip_records.py A.json B.json [--machine-a x86_64 --machine-b arm64]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from python_scripts.reporting.bundle import TREE_FAMILY  # noqa: E402


def _flat(values) -> list:
    if isinstance(values, list):
        return [x for v in values for x in _flat(v)]
    return [values]


def max_relative_difference(a, b) -> float | None:
    """Largest |a - b| / max(|a|, 1e-12) over paired numbers; None when a value is not a number."""
    fa, fb = _flat(a), _flat(b)
    if len(fa) != len(fb):
        return None
    worst = 0.0
    for x, y in zip(fa, fb, strict=True):
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (x, y))
        if not numeric:
            if x != y:
                return None
            continue
        worst = max(worst, abs(x - y) / max(abs(x), 1e-12))
    return worst


def classify(row_a: dict, row_b: dict, platforms_differ: bool, rounding: float) -> tuple[str, float | None]:
    a, b = row_a.get("output_before"), row_b.get("output_before")
    if a == b:
        return "identical", 0.0
    rel = max_relative_difference(a, b)
    if rel is not None and rel <= rounding:
        return "rounding", rel
    if platforms_differ and any(k in str(row_a.get("estimator") or "") for k in TREE_FAMILY):
        return "tree, platform", rel
    return "unexplained", rel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--machine-a")
    ap.add_argument("--machine-b")
    ap.add_argument("--rounding", type=float, default=1e-9)
    args = ap.parse_args()

    rec_a, rec_b = json.loads(Path(args.a).read_text()), json.loads(Path(args.b).read_text())
    ma = args.machine_a or (rec_a.get("environment") or {}).get("machine")
    mb = args.machine_b or (rec_b.get("environment") or {}).get("machine")
    if not ma or not mb:
        print("processor type missing from a record; pass --machine-a and --machine-b", file=sys.stderr)
        return 2
    platforms_differ = ma != mb
    rows_a = {r["model"]: r for r in rec_a.get("rows", []) if "output_before" in r}
    rows_b = {r["model"]: r for r in rec_b.get("rows", []) if "output_before" in r}
    counts: dict[str, int] = {}
    unexplained = []
    print(f"A: {args.a} ({ma})\nB: {args.b} ({mb})")
    for name in sorted(set(rows_a) & set(rows_b)):
        kind, rel = classify(rows_a[name], rows_b[name], platforms_differ, args.rounding)
        counts[kind] = counts.get(kind, 0) + 1
        if kind != "identical":
            shown = "not numeric" if rel is None else f"{rel:.2e}"
            print(f"  {name:24s} {kind:15s} max relative difference {shown}")
        if kind == "unexplained":
            unexplained.append(name)
    only = sorted(set(rows_a) ^ set(rows_b))
    if only:
        print(f"  in one record only: {', '.join(only)}")
    print(", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    return 1 if unexplained else 0


if __name__ == "__main__":
    raise SystemExit(main())
