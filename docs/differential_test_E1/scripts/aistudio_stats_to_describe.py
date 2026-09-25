"""Reshape the AI Studio Aggregate output for D1 and D2 into the describe layout.

Input: reference_outputs/aistudio/aggregate_raw/D1_runN.csv and D2_runN.csv,
one row each, columns named like "average(age)". Output: D1_aistudio_runN.csv and
D2_aistudio_runN.csv with column,n,mean,sd,min,median,max (no q1, q3: Aggregate
has no quartile function). Values are copied as the text AI Studio wrote; nothing
is rounded here. Run from the E1 folder: python3 scripts/aistudio_stats_to_describe.py
"""
from pathlib import Path
import csv
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reference_outputs" / "aistudio"
MAP = {"count": "n", "average": "mean", "standard_deviation": "sd", "minimum": "min",
       "median": "median", "maximum": "max"}
ORDER = ["n", "mean", "sd", "min", "median", "max"]

for case in ("D1", "D2"):
    for run in (1, 2):
        src = OUT / "aggregate_raw" / f"{case}_run{run}.csv"
        if not src.exists():
            print("missing", src.name); continue
        with open(src, newline="") as f:
            rows = list(csv.reader(f))
        hdr, vals = rows[0], rows[1]
        table: dict[str, dict[str, str]] = {}
        for h, v in zip(hdr, vals):
            m = re.fullmatch(r"(\w+)\((.+)\)", h.strip())
            if not m or m.group(1) not in MAP:
                continue
            table.setdefault(m.group(2), {})[MAP[m.group(1)]] = v
        with open(OUT / f"{case}_aistudio_run{run}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["column"] + ORDER)
            for col, d in table.items():
                w.writerow([col] + [d.get(s, "") for s in ORDER])
        print(f"{case} run {run}: {len(table)} columns written from {src.name}")
