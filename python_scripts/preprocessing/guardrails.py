"""Data-quality and leakage checks that run before a model is trained.

A high score on a leaky or degenerate dataset is worse than no score, because
it reads as a result. These checks look at the modeling columns the user chose
and report what would mislead: a feature that is the target in disguise,
duplicate rows that land on both sides of the split, constant columns,
identifiers, very high cardinality, heavy missingness, class imbalance, and too
few rows for the requested validation. Findings that make the score
meaningless are blocking; the run stops unless the request sets allowUnsafe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

# Absolute Pearson correlation between a numeric feature and the target at or
# above which the feature is treated as the target itself.
LEAK_BLOCK_R = 0.999
LEAK_WARN_R = 0.95
HIGH_CARDINALITY_ABS = 100
HIGH_CARDINALITY_FRAC = 0.5
MISSING_WARN_FRAC = 0.5
IMBALANCE_WARN_FRAC = 0.10
MIN_ROWS = 30
MIN_TEST_ROWS = 10


@dataclass
class Finding:
    code: str
    severity: str  # "block" or "warn"
    message: str
    columns: list[str] = field(default_factory=list)


@dataclass
class GuardrailReport:
    findings: list[Finding] = field(default_factory=list)
    n_rows: int = 0
    n_features: int = 0

    def add(self, code: str, severity: str, message: str, columns=None) -> None:
        self.findings.append(Finding(code, severity, message, list(columns or [])))

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "block"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warn"]

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "blocking": [asdict(f) for f in self.blocking],
            "warnings": [asdict(f) for f in self.warnings],
        }

    def block_message(self) -> str:
        lines = [f.message for f in self.blocking]
        return (
            "Training stopped because the result would not be meaningful: "
            + " ".join(lines)
            + " Set allowUnsafe to run anyway."
        )


def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def run_guardrails(
    df: pd.DataFrame,
    indicator_names: list[str],
    predictor_names: list[str],
    problem_type: str,
    test_size: float = 0.2,
    cv_folds: int | None = None,
) -> GuardrailReport:
    """Check the modeling columns and return every finding.

    problem_type is "regression", "classification", or "cluster". Clustering has
    no target, so the leakage and class checks are skipped for it.
    """
    report = GuardrailReport(n_rows=int(len(df)), n_features=len(indicator_names))
    features = [c for c in indicator_names if c in df.columns]
    targets = [c for c in predictor_names if c in df.columns] if problem_type != "cluster" else []
    n = len(df)

    # --- rows -----------------------------------------------------------------
    if n < MIN_ROWS:
        report.add(
            "few_rows", "warn", f"The file has {n} rows; estimates from fewer than {MIN_ROWS} rows are unstable."
        )
    test_rows = int(round(n * test_size))
    if test_rows < MIN_TEST_ROWS:
        report.add(
            "small_test_set",
            "warn",
            f"A test size of {test_size:.2f} leaves {test_rows} test rows; a test metric on fewer than "
            f"{MIN_TEST_ROWS} rows says little.",
        )
    dup_cols = features + targets
    if dup_cols:
        n_dup = int(df.duplicated(subset=dup_cols).sum())
        if n_dup:
            report.add(
                "duplicate_rows",
                "warn",
                f"{n_dup:,} rows are exact duplicates across the modeling columns; a duplicate that lands on "
                "both sides of the split inflates the test score.",
            )

    # --- features -------------------------------------------------------------
    constant = [c for c in features if df[c].nunique(dropna=True) <= 1]
    if constant:
        report.add(
            "constant_columns", "warn", f"Constant columns add no information: {', '.join(constant)}.", constant
        )
    id_like = []
    for c in features:
        s = df[c]
        if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == n and n > 20:
            id_like.append(c)
    if id_like:
        report.add(
            "identifier_columns",
            "warn",
            f"Columns with one distinct integer per row look like identifiers, which a tree model can memorize: "
            f"{', '.join(id_like)}.",
            id_like,
        )
    high_card = []
    for c in features:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            k = s.nunique(dropna=True)
            if k > HIGH_CARDINALITY_ABS or (n and k / n > HIGH_CARDINALITY_FRAC):
                high_card.append(c)
    if high_card:
        report.add(
            "high_cardinality",
            "warn",
            f"Text columns with many distinct values expand into one indicator column each when transformed: "
            f"{', '.join(high_card)}.",
            high_card,
        )
    missing = {c: float(df[c].isna().mean()) for c in features + targets if df[c].isna().mean() > MISSING_WARN_FRAC}
    if missing:
        desc = ", ".join(f"{c} ({v:.0%})" for c, v in missing.items())
        report.add(
            "heavy_missingness", "warn", f"Columns with more than half the values missing: {desc}.", list(missing)
        )

    # --- target ---------------------------------------------------------------
    if problem_type == "regression" and targets:
        for t in targets:
            y = _numeric(df[t])
            if y.notna().sum() < 3:
                continue
            for c in features:
                if c == t:
                    report.add("target_as_feature", "block", f"'{c}' is both a feature and the target.", [c])
                    continue
                x = _numeric(df[c])
                mask = x.notna() & y.notna()
                if mask.sum() < 3 or x[mask].std() == 0 or y[mask].std() == 0:
                    continue
                r = float(np.corrcoef(x[mask], y[mask])[0, 1])
                if abs(r) >= LEAK_BLOCK_R:
                    report.add(
                        "target_leakage",
                        "block",
                        f"'{c}' has |r| = {abs(r):.4f} with the target '{t}'; it is the target in another form.",
                        [c],
                    )
                elif abs(r) >= LEAK_WARN_R:
                    report.add(
                        "near_target_leakage",
                        "warn",
                        f"'{c}' has |r| = {abs(r):.3f} with the target '{t}'; check that it is available before the "
                        "target is known.",
                        [c],
                    )
    elif problem_type == "classification" and targets:
        t = targets[0]
        y = df[t].astype(str)
        counts = y.value_counts()
        n_classes = int(len(counts))
        if n_classes >= 2:
            minority = counts.iloc[-1]
            frac = float(minority / n)
            if frac < IMBALANCE_WARN_FRAC:
                report.add(
                    "class_imbalance",
                    "warn",
                    f"The smallest class '{counts.index[-1]}' is {frac:.1%} of rows ({int(minority)} of {n}); accuracy "
                    "alone will look good for a model that never predicts it.",
                    [t],
                )
            if cv_folds and int(minority) < cv_folds:
                report.add(
                    "class_smaller_than_folds",
                    "warn",
                    f"Class '{counts.index[-1]}' has {int(minority)} rows, fewer than the {cv_folds} folds requested; "
                    "stratified folds cannot each contain it.",
                    [t],
                )
            if int(minority) < 2:
                report.add(
                    "singleton_class",
                    "warn",
                    f"Class '{counts.index[-1]}' has one row; it cannot appear in both the training and the test set.",
                    [t],
                )
        for c in features:
            if c == t:
                report.add("target_as_feature", "block", f"'{c}' is both a feature and the target.", [c])
                continue
            s = df[c]
            if s.astype(str).equals(y):
                report.add("target_leakage", "block", f"'{c}' is identical to the target '{t}'.", [c])
                continue
            k = s.nunique(dropna=True)
            if 1 < k <= max(2 * n_classes, 4):
                per_value = df.groupby(s.astype(str), observed=True)[t].nunique()
                if int(per_value.max()) == 1 and n_classes > 1:
                    report.add(
                        "target_leakage",
                        "block",
                        f"Every value of '{c}' maps to exactly one class of '{t}'; it is the target in another form.",
                        [c],
                    )
    return report
