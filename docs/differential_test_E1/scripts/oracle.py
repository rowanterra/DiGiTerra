"""E1 oracle: what scikit-learn 1.9.0 itself returns on the E1 case files.

Purpose: an independent recomputation of cases D1, D2, R1, C1 and C2 with
plain scikit-learn, pandas and numpy at the versions DiGiTerra pins. Not a
reference application. A DiGiTerra value that differs from the oracle is a
DiGiTerra finding; a value that matches the oracle but differs from R or
AI Studio is a scikit-learn behaviour.
Input: data/<name>_E1_base.csv, _train.csv, _test.csv. Column "order" is
excluded from the predictors. No scaling, no imputation.
Test: asserts the pinned versions, the row counts (120/30 and 353/89), and
that no predictor has a missing value.
Output: oracle_outputs/<case>_oracle.csv, one file per case, values written
with repr precision; oracle_outputs/versions.txt.
Run: python3 scripts/oracle.py   (from the E1 folder)
With E1_ANY_VERSION=1 the version pins are not enforced and the output goes to
oracle_outputs_<machine>_sklearn<version>/ (a platform comparison, not the oracle).
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.neighbors import KNeighborsClassifier

import os
import platform

if os.environ.get("E1_ANY_VERSION"):
    # A second oracle on another platform or library version, written beside the
    # pinned one so the two can be compared (used to attribute the Orange C2 gap).
    OUT = Path(f"oracle_outputs_{platform.machine()}_sklearn{sklearn.__version__}")
else:
    assert sklearn.__version__ == "1.9.0", sklearn.__version__
    assert np.__version__ == "1.26.4", np.__version__
    assert pd.__version__ == "2.3.3", pd.__version__
    OUT = Path("oracle_outputs")

DATA = Path("data")
OUT.mkdir(exist_ok=True)
(OUT / "versions.txt").write_text(
    f"python {sys.version.split()[0]}\nscikit-learn {sklearn.__version__}\n"
    f"numpy {np.__version__}\npandas {pd.__version__}\nmachine {platform.machine()} {platform.platform()}\n")


def load(name):
    tr = pd.read_csv(DATA / f"{name}_E1_train.csv")
    te = pd.read_csv(DATA / f"{name}_E1_test.csv")
    return tr, te


def describe(name, cols):
    """D1 and D2: n, mean, sd (n-1), min, Q1, median, Q3, max; quantiles linear."""
    df = pd.read_csv(DATA / f"{name}_E1_base.csv")
    rows = []
    for c in cols:
        s = df[c]
        rows.append({"column": c, "n": int(s.count()), "mean": s.mean(), "sd": s.std(ddof=1),
                     "min": s.min(), "q1": s.quantile(0.25), "median": s.quantile(0.5),
                     "q3": s.quantile(0.75), "max": s.max()})
    return pd.DataFrame(rows)


# D1, D2
diab_cols = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6", "target"]
iris_cols = ["sepal length", "sepal width", "petal length", "petal width"]
describe("diabetes", diab_cols).to_csv(OUT / "D1_oracle.csv", index=False, float_format="%.17g")
describe("iris", iris_cols).to_csv(OUT / "D2_oracle.csv", index=False, float_format="%.17g")

# R1: OLS on diabetes
tr, te = load("diabetes")
assert (len(tr), len(te)) == (353, 89)
X_cols = diab_cols[:-1]
assert not tr[X_cols].isna().any().any() and not te[X_cols].isna().any().any()
ols = LinearRegression().fit(tr[X_cols], tr["target"])
pred = ols.predict(te[X_cols])
coef = pd.DataFrame({"term": ["intercept"] + X_cols, "value": [ols.intercept_] + list(ols.coef_)})
rmse = float(np.sqrt(mean_squared_error(te["target"], pred)))
metrics = pd.DataFrame({"term": ["test_rmse", "test_mae", "test_r2"],
                        "value": [rmse, mean_absolute_error(te["target"], pred), r2_score(te["target"], pred)]})
pd.concat([coef, metrics]).to_csv(OUT / "R1_oracle_coefficients_metrics.csv", index=False, float_format="%.17g")
pd.DataFrame({"order": te["order"], "target": te["target"], "predicted": pred}).to_csv(
    OUT / "R1_oracle_predictions.csv", index=False, float_format="%.17g")

# C1: k-NN on iris
tr, te = load("iris")
assert (len(tr), len(te)) == (120, 30)
knn = KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="euclidean").fit(tr[iris_cols], tr["species"])
pk = knn.predict(te[iris_cols])
labels = sorted(tr["species"].unique())
pd.DataFrame({"order": te["order"], "species": te["species"], "predicted": pk}).to_csv(
    OUT / "C1_oracle_predictions.csv", index=False)
cm = confusion_matrix(te["species"], pk, labels=labels)
with open(OUT / "C1_oracle_metrics.txt", "w") as f:
    f.write(f"accuracy {accuracy_score(te['species'], pk):.17g}\nlabels {labels}\nconfusion rows=true cols=pred\n")
    f.write("\n".join(" ".join(str(v) for v in r) for r in cm) + "\n")

# C2: logistic regression on iris, scikit-learn defaults (L2, C=1, lbfgs). Probabilities
# are written because the 30-row test set gives accuracy 1.0 for every sane model.
lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=100).fit(tr[iris_cols], tr["species"])
pl = lr.predict(te[iris_cols])
proba = lr.predict_proba(te[iris_cols])
out = pd.DataFrame({"order": te["order"], "species": te["species"], "predicted": pl})
for j, lab in enumerate(lr.classes_):
    out[f"p_{lab}"] = proba[:, j]
out.to_csv(OUT / "C2_oracle_predictions.csv", index=False, float_format="%.17g")
cm = confusion_matrix(te["species"], pl, labels=labels)
with open(OUT / "C2_oracle_metrics.txt", "w") as f:
    f.write(f"accuracy {accuracy_score(te['species'], pl):.17g}\nn_iter {lr.n_iter_.tolist()}\n"
            f"labels {labels}\nconfusion rows=true cols=pred\n")
    f.write("\n".join(" ".join(str(v) for v in r) for r in cm) + "\n")

print("R1", metrics.to_string(index=False))
print("C1 accuracy", accuracy_score(te["species"], pk), "C2 accuracy", accuracy_score(te["species"], pl))
