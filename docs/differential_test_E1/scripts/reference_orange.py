"""E1 reference run in Orange, through Orange's own scripting library.

Purpose: cases D1, D2, R1, C1, C2 computed by Orange's learners on the same
train and test files, with no split by Orange. Orange's scripting library is
the second of the two interfaces Demsar et al. (2013) describe, so this is a
run of Orange, not a re-implementation. The Orange widgets wrap these same
learner classes.
Input: data/<name>_E1_base.csv, _train.csv, _test.csv. "order" is moved to a
meta column so no learner reads it.
Test: stops if Orange cannot be imported, if the row counts are not 120/30
and 353/89, or if a test row's "order" is missing.
Output: reference_outputs/orange/<case>_orange_run<N>_*.csv and
versions_run<N>.txt. C1 is written twice: C1 with Orange's kNN preprocessors
turned off and C1b with Orange's default kNN preprocessors (HasClass,
Continuize, RemoveNaNColumns, SklImpute; measured 2026-09-24, there is no
Normalize). On numeric data with no missing values the two are identical.
Run, from the E1 folder, with Orange's bundled Python (run_all.sh finds it):
  <orange python> scripts/reference_orange.py 1
  <orange python> scripts/reference_orange.py 2
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

try:
    import Orange
    from Orange.classification import KNNLearner, LogisticRegressionLearner
    from Orange.data import Domain, Table
    from Orange.evaluation import CA, MAE, R2, RMSE, TestOnTestData
    from Orange.regression import LinearRegressionLearner
    import sklearn
except ImportError as e:  # noqa: BLE001
    sys.exit(f"Orange import failed: {e}. Run with Orange's bundled Python.")

RUN = sys.argv[1] if len(sys.argv) > 1 else "1"
OUT = Path("reference_outputs/orange")
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("data")
G = "%.17g"

(OUT / f"versions_run{RUN}.txt").write_text(
    f"Orange {Orange.version.version}\nscikit-learn {sklearn.__version__}\n"
    f"numpy {np.__version__}\npython {sys.version.split()[0]}\n")


def load_pair(name: str, target: str):
    tr = Table.from_file(str(DATA / f"{name}_E1_train.csv"))
    te = Table.from_file(str(DATA / f"{name}_E1_test.csv"))
    dom = tr.domain
    attrs = [v for v in dom.variables if v.name not in ("order", target)]
    new = Domain(attrs, dom[target], metas=[dom["order"]])
    tr, te = tr.transform(new), te.transform(new)
    return tr, te


def orders(t: Table):
    return [int(v) for v in t.metas[:, 0]]


def write_rows(path: Path, header: list[str], rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


# ---- D1, D2: Orange's own basic statistics (mean, sd n-1, min, quartiles linear)
from Orange.statistics.basic_stats import DomainBasicStats  # noqa: E402


def describe(name: str, cols: list[str], case: str) -> None:
    t = Table.from_file(str(DATA / f"{name}_E1_base.csv"))
    rows = []
    for c in cols:
        col = t.get_column(c).astype(float)
        st = DomainBasicStats(t)[t.domain.index(c)]
        # DomainBasicStats gives min, max, mean, var (population). Quartiles and
        # n-1 sd are from numpy on the same column so the file carries every
        # statistic PLAN.txt names; which are Orange's own is stated in the header.
        q1, med, q3 = np.percentile(col, [25, 50, 75])  # linear, same as pandas default
        rows.append([c, int(np.sum(~np.isnan(col))), G % st.mean, G % np.std(col, ddof=1), G % st.min,
                     G % q1, G % med, G % q3, G % st.max])
    write_rows(OUT / f"{case}_orange_run{RUN}.csv",
               ["column", "n", "mean", "sd", "min", "q1", "median", "q3", "max"], rows)
    with open(OUT / f"{case}_orange_run{RUN}_README.txt", "w") as f:
        f.write("mean, min, max: Orange.statistics.basic_stats.DomainBasicStats. "
                "sd (n-1), q1, median, q3: numpy on the Orange column (linear quantile). "
                "DomainBasicStats.var is the population variance and is not written.\n")


describe("diabetes", ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6", "target"], "D1")
describe("iris", ["sepal length", "sepal width", "petal length", "petal width"], "D2")

# ---- R1: linear regression on diabetes
tr, te = load_pair("diabetes", "target")
assert (len(tr), len(te)) == (353, 89), (len(tr), len(te))
lr = LinearRegressionLearner()          # fit_intercept=True, no regularization
model = lr(tr)
pred = np.asarray(model(te), dtype=float)
y = np.asarray(te.Y, dtype=float)
res = TestOnTestData()(tr, te, [lr])
rmse, mae, r2 = float(RMSE(res)[0]), float(MAE(res)[0]), float(R2(res)[0])
names = [a.name for a in tr.domain.attributes]
coef_rows = [["intercept", G % float(model.intercept)]] + \
            [[n, G % float(c)] for n, c in zip(names, model.coefficients)] + \
            [["test_rmse", G % rmse], ["test_mae", G % mae], ["test_r2", G % r2]]
write_rows(OUT / f"R1_orange_run{RUN}_coefficients_metrics.csv", ["term", "value"], coef_rows)
write_rows(OUT / f"R1_orange_run{RUN}_predictions.csv", ["order", "target", "predicted"],
           [[o, G % yy, G % p] for o, yy, p in zip(orders(te), y, pred)])

# ---- C1: kNN on iris, k=5 Euclidean uniform
tr, te = load_pair("iris", "species")
assert (len(tr), len(te)) == (120, 30), (len(tr), len(te))
labels = list(tr.domain.class_var.values)
true = [labels[int(v)] for v in te.Y]


def run_classifier(learner, case: str, with_probs: bool) -> None:
    m = learner(tr)
    cls = m(te)
    probs = m(te, m.Probs) if with_probs else None
    predicted = [labels[int(v)] for v in cls]
    header = ["order", "species", "predicted"] + ([f"p_{lab}" for lab in labels] if with_probs else [])
    rows = []
    for i, (o, t_, p) in enumerate(zip(orders(te), true, predicted)):
        row = [o, t_, p]
        if with_probs:
            row += [G % float(x) for x in probs[i]]
        rows.append(row)
    write_rows(OUT / f"{case}_orange_run{RUN}_predictions.csv", header, rows)
    r = TestOnTestData()(tr, te, [learner])
    acc = float(CA(r)[0])
    with open(OUT / f"{case}_orange_run{RUN}_metrics.txt", "w") as f:
        f.write(f"accuracy {G % acc}\nlabels {labels}\npreprocessors {[type(p).__name__ for p in learner.preprocessors]}\n")


knn_raw = KNNLearner(n_neighbors=5, metric="euclidean", weights="uniform", preprocessors=[])
run_classifier(knn_raw, "C1", with_probs=False)
knn_default = KNNLearner(n_neighbors=5, metric="euclidean", weights="uniform")
run_classifier(knn_default, "C1b", with_probs=False)

# ---- C2: logistic regression, L2, C=1 (Orange passes these to scikit-learn)
logit = LogisticRegressionLearner(penalty="l2", C=1.0)
run_classifier(logit, "C2", with_probs=True)

print(f"Orange run {RUN} written to {OUT}. R1 test_rmse {G % rmse}")
