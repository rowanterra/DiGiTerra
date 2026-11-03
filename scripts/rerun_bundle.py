"""Rerun the training recorded in a bundle's manifest and compare the metrics.

Given the bundle and the original data file, the script checks that the file's
SHA-256 matches the manifest, replays the recorded request through
run_model_training with the recorded seed, and prints the recorded and rerun
metrics side by side. Metrics that differ beyond --tol are listed and the exit
code is 1. Installed package versions and the operating system and processor type
are compared with the manifest, and a difference is printed as a note: a tree
model retrained on another processor type can resolve tied splits differently
(docs/tree_platform_difference_2026-09-25).

Usage:
    MPLBACKEND=Agg .venv/bin/python scripts/rerun_bundle.py BUNDLE.zip DATA.csv [--tol 1e-6] [--json out.json]
"""

import argparse
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_sup = Path(tempfile.mkdtemp(prefix="dgt_rerun_"))
os.environ.setdefault("DIGITERRA_APP_SUPPORT_DIR", str(_sup))
os.environ.setdefault("DIGITERRA_OUTPUT_DIR", str(_sup / "vis"))
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
warnings.filterwarnings("ignore")

from core.helpers import read_table_bounded  # noqa: E402
from python_scripts.app_model_training import run_model_training  # noqa: E402
from python_scripts.preprocessing.progress_tracker import get_result, remove_tracker  # noqa: E402
from python_scripts.reporting.bundle import package_drift, platform_change, read_manifest, sha256_file  # noqa: E402


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle")
    ap.add_argument("data")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--json")
    ap.add_argument("--skip-hash", action="store_true", help="rerun even when the file hash differs")
    args = ap.parse_args()

    manifest = read_manifest(args.bundle)
    recorded_sha = (manifest.get("data") or {}).get("sha256")
    actual_sha = sha256_file(args.data)
    if recorded_sha and recorded_sha != actual_sha and not args.skip_hash:
        print(f"hash mismatch: manifest {recorded_sha}, file {actual_sha}", file=sys.stderr)
        return 2
    drift = package_drift(manifest)
    if drift:
        print(f"note: installed versions differ from the bundle for {', '.join(drift)}", file=sys.stderr)
    moved = platform_change(manifest)
    if moved:
        print(f"note: {moved}", file=sys.stderr)

    request = dict(manifest.get("request") or {})
    request["seedValue"] = (manifest.get("model") or {}).get("seed") or request.get("seedValue")
    store = {"data": read_table_bounded(args.data, max_cells=0, max_frame_mb=0)}
    sid = "rerun"
    run_model_training(sid, request, "store", lambda _s: store)
    result = get_result(sid)
    remove_tracker(sid)
    if result is None or "error" in result:
        print(f"rerun failed: {(result or {}).get('error')}", file=sys.stderr)
        return 1

    recorded = manifest.get("metrics") or {}
    rows = []
    mismatches = []
    for key, rec in recorded.items():
        new = result.get(key)
        a, b = _num(rec), _num(new)
        same = (a is not None and b is not None and abs(a - b) <= args.tol) or (a is None and str(rec) == str(new))
        rows.append({"metric": key, "recorded": rec, "rerun": new, "match": same})
        if not same:
            mismatches.append(key)
    width = max(len(r["metric"]) for r in rows) if rows else 8
    for r in rows:
        flag = "ok" if r["match"] else "DIFF"
        print(f"{r['metric']:<{width}}  recorded {r['recorded']!s:>12}  rerun {r['rerun']!s:>12}  {flag}")
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "hash_match": recorded_sha == actual_sha,
                    "package_drift": drift,
                    "platform_change": moved,
                    "rows": rows,
                },
                indent=2,
            )
        )
    if mismatches:
        print(f"{len(mismatches)} metric(s) differ: {', '.join(mismatches)}", file=sys.stderr)
        if moved:
            print("the platform changed since the recorded run; see the note above", file=sys.stderr)
        return 1
    print("all recorded metrics reproduced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
