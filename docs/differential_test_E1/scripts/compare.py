"""E1 comparison: every saved output against the oracle and against each other.

Purpose: read the files that exist under oracle_outputs/, reference_outputs/
<app>/ and digiterra_outputs/<case>/, line them up by case, and classify each
value: same, last digit, other. Predicted classes are compared row by row on
the "order" key.
Input: the file names and layouts given in PROTOCOL.txt. Files from Orange,
AI Studio, Excel and DiGiTerra are read through the adapters below; a file
the adapter cannot read is listed as "unparsed" and counted as missing,
never silently skipped.
Test: the oracle set must be complete or the script stops; each parsed
predictions file must have the expected number of rows (89 or 30) and every
"order" value of the test file, or the file is marked "row mismatch".
Output: RESULTS.csv (one row per app, case, term, value class), and
RESULT_<date>.txt with the counts, the "other" cells and the partial flag.
Run: python3 scripts/compare.py   (from the E1 folder)
"""
from __future__ import annotations

import datetime as dt
import math
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(".")
ORACLE = ROOT / "oracle_outputs"
REF = ROOT / "reference_outputs"
DGT = ROOT / "digiterra_outputs"
APPS = ["digiterra", "orange", "aistudio", "r", "excel"]
INDEPENDENT_TOL = 1e-6          # PLAN.txt: independent engines
N_TEST = {"R1": 89, "C1": 30, "C2": 30}
TERMS_R1 = ["intercept", "age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6",
            "test_rmse", "test_mae", "test_r2"]
STATS = ["n", "mean", "sd", "min", "q1", "median", "q3", "max"]


# ---------- value classes ----------
def classify(ref: float, val: float) -> str:
    if ref is None or val is None or (isinstance(val, float) and math.isnan(val)):
        return "missing"
    a, b = f"{ref:.6e}", f"{val:.6e}"
    if a == b:
        return "same"
    ma, ea = a.split("e"); mb, eb = b.split("e")
    if ea == eb and abs(round(float(ma) * 1e6) - round(float(mb) * 1e6)) == 1:
        return "last digit"
    if abs(ref) < 1e-10 and abs(val) < 1e-10:
        return "near zero"
    return "other"


def rounding_class(ref: float, val: float) -> str | None:
    """'rounded N dp when val equals ref rounded to 6, 4 or 3 decimals."""
    for d in (6, 4, 3):
        if abs(round(ref, d) - val) <= 1e-12:
            return f"rounded {d} dp"
    return None


def reldiff(ref: float, val: float) -> float:
    d = abs(ref - val)
    return d / abs(ref) if ref not in (0, None) else d


# ---------- adapters: each returns a dict term -> float, or a predictions frame ----------
def read_term_value(path: Path) -> dict[str, float]:
    """term,value files (oracle, r, excel coefficients_metrics)."""
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    tcol = "term" if "term" in cols else cols[0]
    vcol = "value" if "value" in cols else cols[1]
    alias = {"root_mean_squared_error": "test_rmse", "absolute_error": "test_mae"}  # AI Studio criteria
    out = {}
    for t, v in zip(df[tcol], df[vcol]):
        try:
            out[alias.get(str(t).strip().lower(), str(t).strip().lower())] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def read_describe(path: Path) -> dict[str, float]:
    """column, n, mean, sd, min, q1, median, q3, max -> {"<column>.<stat>": value}.
    Orange Feature Statistics and AI Studio Statistics use their own headers;
    the rename map below covers the ones seen so far and is extended when a
    new file arrives."""
    df = pd.read_csv(path)
    ren = {"feature": "column", "attribute": "column", "name": "column", "variable": "column",
           "average": "mean", "std": "sd", "std. dev.": "sd", "stdev": "sd", "standard deviation": "sd",
           "count": "n", "25%": "q1", "50%": "median", "75%": "q3", "first quartile": "q1",
           "third quartile": "q3", "minimum": "min", "maximum": "max",
           "25": "q1", "50": "median", "75": "q3", "100": "max"}
    cols = [str(c).strip().lower() for c in df.columns]
    if "100" in cols and "max" in cols:
        df = df.drop(columns=[df.columns[cols.index("100")]]); cols.remove("100")
    df.columns = [ren.get(c, c) for c in cols]
    if "column" not in df.columns:
        df = df.rename(columns={df.columns[0]: "column"})
    out = {}
    for _, r in df.iterrows():
        for s in STATS:
            if s in df.columns:
                try:
                    out[f"{str(r['column']).strip()}.{s}"] = float(r[s])
                except (TypeError, ValueError):
                    pass
    return out


def read_predictions(path: Path, case: str) -> pd.DataFrame:
    """Return a frame with columns order, predicted (and p_<class> when present)."""
    df = pd.read_csv(path)
    low = {c: str(c).strip().lower() for c in df.columns}
    order_col = next((c for c in df.columns if low[c] == "order"), df.columns[0])
    pred_col = None
    for c in df.columns:
        lc = low[c]
        if lc in ("predicted", "prediction", "prediction(target)", "prediction(species)"):
            pred_col = c; break
    if pred_col is None:
        # Orange names the prediction column after the learner
        for c in df.columns:
            if low[c] in ("linear regression", "knn", "logistic regression", "linear regression (1)"):
                pred_col = c; break
    if pred_col is None:
        for c in df.columns:
            if "predict" in low[c] and "confidence" not in low[c] and "(" not in low[c]:
                pred_col = c; break
    if pred_col is None:
        raise ValueError(f"no prediction column in {path.name}: {list(df.columns)}")
    out = pd.DataFrame({"order": pd.to_numeric(df[order_col], errors="coerce").astype("Int64"),
                        "predicted": df[pred_col]})
    for c in df.columns:
        m = re.match(r"^(?:p_|confidence\(|.*\()?(setosa|versicolor|virginica)\)?$", low[c])
        if m and c != pred_col:
            out[f"p_{m.group(1)}"] = pd.to_numeric(df[c], errors="coerce")
    return out


# ---------- gather ----------
def find(app: str, case: str, kind: str) -> list[Path]:
    """kind: coefficients_metrics | predictions | describe | metrics."""
    if app == "oracle":
        base = ORACLE
        pat = {"coefficients_metrics": f"{case}_oracle_coefficients_metrics.csv",
               "predictions": f"{case}_oracle_predictions.csv",
               "describe": f"{case}_oracle.csv"}[kind]
        return sorted(base.glob(pat))
    base = DGT if app == "digiterra" else REF / app
    if app == "digiterra":
        folders = {"R1": ["c000_base_a", "c001_base_b", "c999_base_c"], "C1": ["C1"], "C2": ["C2"],
                   "D1": ["D1"], "D2": ["D2"]}[case]
        names = {"coefficients_metrics": ["coefficients_metrics.csv", "metrics_from_predictions.csv"],
                 "predictions": ["predictions_test.csv"], "describe": [f"{case}_digiterra.csv"]}[kind]
        return sorted(p for fo in folders for nm in names for p in [base / fo / nm] if p.exists())
    pats = {"coefficients_metrics": [f"{case}_{app}_run*_coefficients_metrics.csv", f"{case}_{app}_run*_coefficients.csv",
                                     f"{case}*/**/*coef*.csv", f"{case}*/**/*manifest*.json"],
            "predictions": [f"{case}_{app}_run*_predictions.csv", f"{case}*/**/*predict*.csv"],
            "describe": [f"{case}_{app}_run*.csv", f"{case}/*.csv"]}[kind]
    hits = []
    for p in pats:
        hits += list(base.glob(p))
    return sorted(set(hits))


def main() -> None:
    rows, notes, unparsed = [], [], []
    missing_oracle = [k for k in ("D1", "D2") if not find("oracle", k, "describe")] + \
                     [k for k in ("R1",) if not find("oracle", k, "coefficients_metrics")] + \
                     [k for k in ("R1", "C1", "C2") if not find("oracle", k, "predictions")]
    if missing_oracle:
        sys.exit(f"oracle set incomplete: {missing_oracle}; run scripts/oracle.py first")

    oracle_R1 = read_term_value(find("oracle", "R1", "coefficients_metrics")[0])
    oracle_D = {k: read_describe(find("oracle", k, "describe")[0]) for k in ("D1", "D2")}
    oracle_P = {k: read_predictions(find("oracle", k, "predictions")[0], k) for k in ("R1", "C1", "C2")}
    test_orders = {k: set(oracle_P[k]["order"].dropna().astype(int)) for k in oracle_P}

    for app in APPS:
        tol_same_engine = app in ("digiterra", "orange")
        # R1 terms
        for f in find(app, "R1", "coefficients_metrics"):
            try:
                vals = read_term_value(f) if f.suffix == ".csv" else {}
            except Exception as e:  # noqa: BLE001
                unparsed.append((app, "R1", f.name, str(e))); continue
            absent = [t for t in TERMS_R1 if t not in vals]
            if absent:
                notes.append(f"{app} R1 {f.name}: terms not in file: {absent}")
            for t in TERMS_R1:
                if t not in vals:
                    continue
                ref = oracle_R1[t]; v = vals.get(t)
                cls = classify(ref, v) if v is not None else "missing"
                rd = reldiff(ref, v) if v is not None else float("nan")
                if not tol_same_engine and cls == "other" and rd < INDEPENDENT_TOL:
                    cls = "within 1e-6"
                if cls == "other" and rounding_class(ref, v):
                    cls = rounding_class(ref, v)
                rows.append(dict(app=app, case="R1", file=f"{f.parent.name}/{f.name}" if app == "digiterra" else f.name, term=t, oracle=ref, value=v, cls=cls, reldiff=rd))
        # descriptive
        for case in ("D1", "D2"):
            for f in find(app, case, "describe"):
                try:
                    vals = read_describe(f)
                except Exception as e:  # noqa: BLE001
                    unparsed.append((app, case, f.name, str(e))); continue
                absent = [t for t in oracle_D[case] if t not in vals]
                if absent:
                    notes.append(f"{app} {case} {f.name}: statistics not in file: {sorted(set(a.split('.')[1] for a in absent))}")
                for t, ref in oracle_D[case].items():
                    if t not in vals:
                        continue
                    v = vals.get(t)
                    cls = classify(ref, v) if v is not None else "missing"
                    rd = reldiff(ref, v) if v is not None else float("nan")
                    if not tol_same_engine and cls == "other" and rd < INDEPENDENT_TOL:
                        cls = "within 1e-6"
                    if cls == "other" and rounding_class(ref, v):
                        cls = rounding_class(ref, v)
                    rows.append(dict(app=app, case=case, file=f.name, term=t, oracle=ref, value=v, cls=cls, reldiff=rd))
        # predictions
        for case in ("R1", "C1", "C2"):
            for f in find(app, case, "predictions"):
                try:
                    p = read_predictions(f, case)
                except Exception as e:  # noqa: BLE001
                    unparsed.append((app, case, f.name, str(e))); continue
                got = set(p["order"].dropna().astype(int))
                if len(p) != N_TEST[case] or got != test_orders[case]:
                    notes.append(f"{app} {case} {f.name}: row mismatch, {len(p)} rows, "
                                 f"{len(got ^ test_orders[case])} order values differ")
                    rows.append(dict(app=app, case=case, file=f.name, term="rows", oracle=N_TEST[case],
                                     value=len(p), cls="row mismatch", reldiff=float("nan")))
                    continue
                m = oracle_P[case].merge(p, on="order", suffixes=("_oracle", "_app"))
                if case == "R1" and app == "aistudio":
                    yt = pd.read_csv(ROOT / "data" / "diabetes_E1_test.csv").set_index("order").loc[m["order"], "target"].to_numpy(float)
                    e = yt - m["predicted_app"].to_numpy(float)
                    r2 = 1 - (e ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
                    ref = oracle_R1["test_r2"]; cls = classify(ref, r2); rd = reldiff(ref, r2)
                    if cls == "other" and rd < INDEPENDENT_TOL:
                        cls = "within 1e-6"
                    rows.append(dict(app=app, case="R1", file=f.name + " (R2 from predictions)", term="test_r2",
                                     oracle=ref, value=r2, cls=cls, reldiff=rd))
                if case == "R1":
                    for _, r in m.iterrows():
                        ref, v = float(r["predicted_oracle"]), float(r["predicted_app"])
                        cls = classify(ref, v); rd = reldiff(ref, v)
                        if not tol_same_engine and cls == "other" and rd < INDEPENDENT_TOL:
                            cls = "within 1e-6"
                        rows.append(dict(app=app, case=case, file=f.name, term=f"pred.order{int(r['order'])}",
                                         oracle=ref, value=v, cls=cls, reldiff=rd))
                else:
                    mism = m[m["predicted_oracle"].astype(str).str.strip().str.lower()
                             != m["predicted_app"].astype(str).str.strip().str.lower()]
                    rows.append(dict(app=app, case=case, file=f.name, term="class_mismatches", oracle=0,
                                     value=len(mism), cls="same" if len(mism) == 0 else "other", reldiff=float("nan")))
                    for _, r in mism.iterrows():
                        notes.append(f"{app} {case} order {int(r['order'])}: oracle {r['predicted_oracle']} "
                                     f"vs {r['predicted_app']}")
                    for lab in ("setosa", "versicolor", "virginica"):
                        a, b = f"p_{lab}_oracle", f"p_{lab}_app"
                        if a in m.columns and b in m.columns:
                            for _, r in m.iterrows():
                                if pd.isna(r[b]):
                                    continue
                                ref, v = float(r[a]), float(r[b])
                                cls = classify(ref, v); rd = reldiff(ref, v)
                                rows.append(dict(app=app, case=case, file=f.name,
                                                 term=f"p_{lab}.order{int(r['order'])}", oracle=ref, value=v,
                                                 cls=cls, reldiff=rd))

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "RESULTS.csv", index=False, float_format="%.17g")
    today = dt.date.today().isoformat()
    expected = {(a, c) for a in APPS for c in ("D1", "D2", "R1", "C1", "C2")}
    # C2 and D/C cases that PLAN.txt excludes for an app (R and AI Studio fit a
    # different logistic estimator; Excel has no classifier)
    expected -= {("r", "C2"), ("aistudio", "C2"), ("excel", "C1"), ("excel", "C2")}
    present = set(zip(res["app"], res["case"])) if len(res) else set()
    partial = expected - present
    with open(ROOT / f"RESULT_{today}.txt", "w") as f:
        f.write(f"E1 comparison, {today}. {'PARTIAL' if partial else 'COMPLETE'} set.\n")
        if partial:
            f.write("Not yet present: " + ", ".join(f"{a}/{c}" for a, c in sorted(partial)) + "\n")
        f.write("\nCounts by app, case and class:\n")
        if len(res):
            f.write(res.groupby(["app", "case", "cls"]).size().to_string() + "\n")
        f.write("\nCells classed other (oracle, value, relative difference):\n")
        if len(res):
            oth = res[res["cls"].isin(["other", "row mismatch"])]
            for _, r in oth.iterrows():
                f.write(f"  {r['app']} {r['case']} {r['term']}: {r['oracle']!r} vs {r['value']!r} rel {r['reldiff']:.3e}\n")
        f.write("\nCells that equal the oracle rounded to 3 or 4 decimals (display rounding), by app and case:\n")
        if len(res):
            rr = res[res["cls"].str.startswith("rounded")]
            if len(rr):
                f.write(rr.groupby(["app", "case", "cls"]).size().to_string() + "\n")
        f.write("\nNotes:\n" + "".join(f"  {n}\n" for n in notes))
        f.write("\nUnparsed files:\n" + "".join(f"  {a} {c} {n}: {e}\n" for a, c, n, e in unparsed))
    print(open(ROOT / f"RESULT_{today}.txt").read())


if __name__ == "__main__":
    main()
