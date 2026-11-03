"""Build the E1 Excel workbook (scripts/E1_excel_template.xlsx).

Input: data/diabetes_E1_base.csv, data/iris_E1_base.csv, data/diabetes_E1_train.csv,
data/diabetes_E1_test.csv (all numeric except iris species and the order column).
The workbook carries the data as values and every statistic as an Excel formula,
so the numbers are Excel's own once Excel opens and saves the file. D1 and D2 use (with the _xlfn. prefix the file format needs)
COUNT, AVERAGE, STDEV.S, MIN, QUARTILE.INC, MEDIAN, MAX. R1 uses LINEST with an
intercept, SUMPRODUCT for the 89 test predictions, SUMXMY2 and DEVSQ for the
metrics. Run from the E1 folder: python3 scripts/make_excel_template.py
Then in Excel: open the template, File > Save As E1_excel_run1.xlsx in
reference_outputs/excel/, close; open the template again, Save As
E1_excel_run2.xlsx. Then python3 scripts/extract_excel.py.
"""
from pathlib import Path
import csv
from openpyxl import Workbook
from openpyxl.utils import get_column_letter as L

ROOT = Path(__file__).resolve().parents[1]
wb = Workbook()
wb.remove(wb.active)


def load(name: str) -> tuple[list[str], list[list]]:
    with open(ROOT / "data" / name, newline="") as f:
        rows = list(csv.reader(f))
    hdr, body = rows[0], rows[1:]
    out = []
    for r in body:
        out.append([float(v) if i < len(r) and _isnum(v) else v for i, v in enumerate(r)])
    return hdr, out


def _isnum(v: str) -> bool:
    try:
        float(v); return True
    except ValueError:
        return False


def data_sheet(title: str, hdr: list[str], body: list[list]) -> None:
    ws = wb.create_sheet(title)
    ws.append(hdr)
    for r in body:
        ws.append(r)


def describe_sheet(title: str, src: str, hdr: list[str], n: int, cols: list[str]) -> None:
    ws = wb.create_sheet(title)
    ws.append(["column", "n", "mean", "sd", "min", "q1", "median", "q3", "max"])
    for c in cols:
        j = hdr.index(c) + 1
        rng = f"'{src}'!{L(j)}2:{L(j)}{n + 1}"
        ws.append([c, f"=COUNT({rng})", f"=AVERAGE({rng})", f"=_xlfn.STDEV.S({rng})", f"=MIN({rng})",
                   f"=_xlfn.QUARTILE.INC({rng},1)", f"=MEDIAN({rng})", f"=_xlfn.QUARTILE.INC({rng},3)", f"=MAX({rng})"])


# ---- D1, D2
h, b = load("diabetes_E1_base.csv"); data_sheet("diabetes_base", h, b)
describe_sheet("D1", "diabetes_base", h, len(b), [c for c in h if c != "order"])
h2, b2 = load("iris_E1_base.csv"); data_sheet("iris_base", h2, b2)
describe_sheet("D2", "iris_base", h2, len(b2), [c for c in h2 if c not in ("order", "species")])

# ---- R1
ht, bt = load("diabetes_E1_train.csv"); data_sheet("diabetes_train", ht, bt)
he, be = load("diabetes_E1_test.csv"); data_sheet("diabetes_test", he, be)
preds = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
ntr, nte = len(bt), len(be)
xs = f"'diabetes_train'!B2:K{ntr + 1}"           # age..s6 are columns B..K
ys = f"'diabetes_train'!L2:L{ntr + 1}"           # target is column L
ws = wb.create_sheet("R1")
ws["A1"] = "term"; ws["B1"] = "value"
# LINEST returns [m10 ... m1, b]; predictor j (1-based) sits at index 11-j, intercept at 11
ws.append(["intercept", f"=INDEX(LINEST({ys},{xs},TRUE,FALSE),1,11)"])
for j, p in enumerate(preds, start=1):
    ws.append([p, f"=INDEX(LINEST({ys},{xs},TRUE,FALSE),1,{11 - j})"])
# coefficient row for SUMPRODUCT: row 14, B..K = age..s6 in data order
ws["A14"] = "coef_row"
for j in range(10):
    ws.cell(14, 2 + j, f"=B{3 + j}")
# predictions block: row 17 header, rows 18.. one per test row
ws["A17"] = "order"; ws["B17"] = "target"; ws["C17"] = "predicted"
for i in range(nte):
    r = 18 + i; src = i + 2
    ws.cell(r, 1, f"='diabetes_test'!A{src}")
    ws.cell(r, 2, f"='diabetes_test'!L{src}")
    ws.cell(r, 3, f"=$B$2+SUMPRODUCT($B$14:$K$14,'diabetes_test'!B{src}:K{src})")
last = 17 + nte
ws.append([]); 
ws.append(["test_rmse", f"=SQRT(SUMXMY2(B18:B{last},C18:C{last})/{nte})"])
ws.append(["test_mae", f"=SUMPRODUCT(ABS(B18:B{last}-C18:C{last}))/{nte}"])
ws.append(["test_r2", f"=1-SUMXMY2(B18:B{last},C18:C{last})/DEVSQ(B18:B{last})"])
out = ROOT / "scripts" / "E1_excel_template.xlsx"
wb.save(out)
print("wrote", out, "train", ntr, "test", nte)
