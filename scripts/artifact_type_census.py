"""Write the artifact allowlist from a census of every registered estimator.

Each registry entry is trained on its example file through the same function the
/process route calls, the fitted objects are written with skops, and every type
name the file asks for is collected. The union, one name per line, is written to
python_scripts/reporting/artifact_allowlist.txt, which artifact.py reads as the
exact set of types a DiGiTerra artifact may name. Run it after adding a model or
changing a pinned library version, and commit the file with that change.

Usage:
    MPLBACKEND=Agg .venv/bin/python scripts/artifact_type_census.py [--check]

--check exits 1 when the census names a type the committed file lacks.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("MPLBACKEND", "Agg")

OUT = ROOT / "python_scripts" / "reporting" / "artifact_allowlist.txt"
HEADER = (
    "# Exact type names a DiGiTerra artifact may ask skops to build.\n"
    "# Written by scripts/artifact_type_census.py from every registered estimator;\n"
    "# do not edit by hand. artifact.py reads this file and refuses any other name.\n"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    import check_artifact_roundtrip as rt
    import skops.io as sio

    from python_scripts.reporting import artifact

    seen: set[str] = set()

    def record(data: bytes) -> list[str]:
        seen.update(sio.get_untrusted_types(data=data))
        return []

    # The round-trip check is reused for the training and the writing; its own
    # allowlist test and loader are bypassed so the census sees every name.
    artifact.untrusted_types = record
    artifact.loads = lambda data: sio.loads(data, trusted=list(sio.get_untrusted_types(data=data)))

    for name, (trainer, task) in sorted(rt.MODEL_REGISTRY.items()):
        if trainer is None or task not in rt.DATASETS:
            continue
        try:
            row = rt.check(name, task)
            print(f"{row.get('status', '?'):18s} {name}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{'error':18s} {name}: {e}", flush=True)

    names = sorted(seen)
    if args.check:
        have = {line.strip() for line in OUT.read_text().splitlines() if line.strip() and not line.startswith("#")}
        missing = sorted(set(names) - have)
        if missing:
            print("Types the committed allowlist lacks:\n  " + "\n  ".join(missing))
            return 1
        print(f"allowlist covers all {len(names)} types the census produced")
        return 0
    OUT.write_text(HEADER + "\n".join(names) + "\n", encoding="utf-8")
    print(f"wrote {OUT} with {len(names)} types")
    return 0


if __name__ == "__main__":
    sys.exit(main())
