"""Read the Excel-saved workbooks and write the reference_outputs/excel files.

Input: reference_outputs/excel/E1_excel_run1.xlsx and E1_excel_run2.xlsx, each
saved by Excel from scripts/E1_excel_template.xlsx so that the cached cell
values are Excel's own. Output per run: D1_excel_runN.csv, D2_excel_runN.csv
(column,n,mean,sd,min,q1,median,q3,max), R1_excel_runN_coefficients_metrics.csv
(term,value) and R1_excel_runN_predictions.csv (order,target,predicted), in
the layout compare.py reads for r. Also versions_runN.txt from docProps/app.xml.
Values are written with repr() so nothing is rounded here. A workbook whose
cells still hold formulas without cached values (never opened in Excel) stops
the script. Run from the E1 folder: python3 scripts/extract_excel.py
"""
from pathlib import Path
import csv
import re
import sys
import zipfile
import openpyxl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reference_outputs" / "excel"
STATS = ["n", "mean", "sd", "min", "q1", "median", "q3", "max"]


def fmt(v) -> str:
    return repr(float(v)) if isinstance(v, (int, float)) else str(v)


def app_version(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        xml = z.read("docProps/app.xml").decode("utf-8", "replace")
    app = re.search(r"<Application>([^<]*)</Application>", xml)
    ver = re.search(r"<AppVersion>([^<]*)</AppVersion>", xml)
    return f"{app.group(1) if app else '?'} {ver.group(1) if ver else '?'}"


for run in (1, 2):
    src = OUT / f"E1_excel_run{run}.xlsx"
    if not src.exists():
        print(f"missing {src.name}; open the template in Excel and Save As this name")
        continue
    wb = openpyxl.load_workbook(src, data_only=True)
    for case in ("D1", "D2"):
        ws = wb[case]
        rows = list(ws.iter_rows(min_row=2, values_only=True))
        if any(v is None or isinstance(v, str) for r in rows for v in r[1:]):
            sys.exit(f"{src.name} sheet {case}: cells without cached numbers; was it saved by Excel?")
        with open(OUT / f"{case}_excel_run{run}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["column"] + STATS)
            for r in rows:
                w.writerow([r[0]] + [fmt(v) for v in r[1:9]])
    ws = wb["R1"]
    terms = {}
    for r in ws.iter_rows(min_row=2, max_row=12, values_only=True):
        terms[r[0]] = r[1]
    for r in ws.iter_rows(min_row=108, max_row=110, values_only=True):
        if r[0]:
            terms[r[0]] = r[1]
    if any(v is None or isinstance(v, str) for v in terms.values()):
        sys.exit(f"{src.name} sheet R1: cells without cached numbers; was it saved by Excel?")
    with open(OUT / f"R1_excel_run{run}_coefficients_metrics.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["term", "value"])
        for k, v in terms.items():
            w.writerow([k, fmt(v)])
    with open(OUT / f"R1_excel_run{run}_predictions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["order", "target", "predicted"])
        n = 0
        for r in ws.iter_rows(min_row=18, max_row=106, values_only=True):
            w.writerow([int(r[0]), fmt(r[1]), fmt(r[2])]); n += 1
    (OUT / f"versions_run{run}.txt").write_text(app_version(src) + "\n")
    print(f"run {run}: D1 D2 R1 written, {n} predictions, {len(terms)} terms, {app_version(src)}")
