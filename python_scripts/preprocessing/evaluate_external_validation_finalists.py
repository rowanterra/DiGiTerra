"""
External validation for regression model candidates across multiple seeds.

Train on one sheet/target (for example Literature/TREE), evaluate on an external
sheet/target (for example Backvalidation/REEY), and summarize robustness using:
- median external Test_R2
- lower-tail external Test_R2_P2_5

Candidates are configurable via --candidates in this form:
Model:outlier_mode,Model:outlier_mode
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCALING_MODELS = {"Ridge", "KNN", "MLP"}


@dataclass(frozen=True)
class EvalConfig:
    input_path: Path
    train_sheet: str
    test_sheet: str
    train_target: str
    test_target: str
    features: List[str]
    candidates: List[Tuple[str, str]]
    cv_folds: int
    seed_start: int
    seed_count: int
    output_dir: Path
    output_stem: str


class Winsorizer(TransformerMixin, BaseEstimator):
    """Fit quantile bounds on training data and clip features to those bounds."""

    def __init__(self, lower_q: float = 0.01, upper_q: float = 0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.lower_: np.ndarray | None = None
        self.upper_: np.ndarray | None = None

    def fit(self, X, y=None):  # noqa: ANN001
        arr = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(arr, self.lower_q, axis=0)
        self.upper_ = np.nanquantile(arr, self.upper_q, axis=0)
        return self

    def transform(self, X):  # noqa: ANN001
        if self.lower_ is None or self.upper_ is None:
            raise ValueError("Winsorizer must be fit before transform.")
        arr = np.asarray(X, dtype=float)
        return np.clip(arr, self.lower_, self.upper_)


def parse_list_arg(arg_value: str) -> List[str]:
    return [v.strip() for v in arg_value.split(",") if v.strip()]


def parse_candidates_arg(arg_value: str) -> List[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []
    for token in parse_list_arg(arg_value):
        if ":" not in token:
            raise ValueError(
                "Each candidate must be in 'Model:outlier_mode' format. "
                f"Got: '{token}'"
            )
        model_name, outlier_mode = token.split(":", 1)
        candidates.append((model_name.strip(), outlier_mode.strip()))
    if not candidates:
        raise ValueError("No candidates parsed from --candidates.")
    return candidates


def load_dataframe(input_path: Path, sheet_name: str) -> pd.DataFrame:
    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(input_path, sheet_name=sheet_name)
    return pd.read_csv(input_path)


def get_model(model_name: str, seed: int) -> RegressorMixin:
    if model_name == "Ridge":
        return Ridge(alpha=1.0, random_state=seed)
    if model_name == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=500, random_state=seed, n_jobs=-1)
    if model_name == "KNN":
        return KNeighborsRegressor(n_neighbors=7, weights="distance")
    if model_name == "MLP":
        return MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=2000,
            random_state=seed,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def make_pipeline(model_name: str, model: RegressorMixin, outlier_mode: str) -> Pipeline:
    steps: List[Tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if outlier_mode == "winsorize_1_99":
        steps.append(("winsor", Winsorizer(0.01, 0.99)))
    elif outlier_mode != "none":
        raise ValueError(f"Unsupported outlier mode: {outlier_mode}")

    if model_name in SCALING_MODELS:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def aggregate_results(per_seed_results: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["Model", "Outlier_Mode"]
    grouped = per_seed_results.groupby(group_cols, as_index=False)
    summary = grouped.agg(
        Seeds=("Seed", "count"),
        CV_R2_Mean_Median=("CV_R2_Mean", "median"),
        CV_R2_Mean_Mean=("CV_R2_Mean", "mean"),
        CV_R2_Mean_Std=("CV_R2_Mean", "std"),
        Test_R2_Median=("Test_R2", "median"),
        Test_R2_Mean=("Test_R2", "mean"),
        Test_R2_Std=("Test_R2", "std"),
        Test_RMSE_Median=("Test_RMSE", "median"),
        Test_RMSE_Mean=("Test_RMSE", "mean"),
        Test_RMSE_Std=("Test_RMSE", "std"),
        Test_MAE_Median=("Test_MAE", "median"),
        Test_MAE_Mean=("Test_MAE", "mean"),
        Test_MAE_Std=("Test_MAE", "std"),
    )

    q = per_seed_results.groupby(group_cols)["Test_R2"].quantile([0.025, 0.975]).unstack().reset_index()
    q.columns = group_cols + ["Test_R2_P2_5", "Test_R2_P97_5"]
    summary = summary.merge(q, on=group_cols, how="left")
    return summary.sort_values(by=["Test_R2_P2_5", "Test_R2_Median"], ascending=False).reset_index(drop=True)


def evaluate_seed(
    seed: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    candidates: List[Tuple[str, str]],
    cv_folds: int,
) -> List[dict]:
    rows: List[dict] = []
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    for model_name, outlier_mode in candidates:
        model = get_model(model_name, seed=seed)
        pipe = make_pipeline(model_name, model, outlier_mode=outlier_mode)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append(
            {
                "Seed": seed,
                "Model": model_name,
                "Outlier_Mode": outlier_mode,
                "CV_R2_Mean": float(np.mean(cv_scores)),
                "CV_R2_Std": float(np.std(cv_scores)),
                "Test_R2": float(r2_score(y_test, pred)),
                "Test_RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                "Test_MAE": float(mean_absolute_error(y_test, pred)),
                "N_Train": int(len(X_train)),
                "N_Test": int(len(X_test)),
            }
        )
    return rows


def run(cfg: EvalConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = load_dataframe(cfg.input_path, cfg.train_sheet).copy()
    test_df = load_dataframe(cfg.input_path, cfg.test_sheet).copy()

    train_required = cfg.features + [cfg.train_target]
    test_required = cfg.features + [cfg.test_target]
    train_missing = [c for c in train_required if c not in train_df.columns]
    test_missing = [c for c in test_required if c not in test_df.columns]
    if train_missing:
        raise ValueError(f"Missing columns in train sheet '{cfg.train_sheet}': {train_missing}")
    if test_missing:
        raise ValueError(f"Missing columns in test sheet '{cfg.test_sheet}': {test_missing}")

    X_train = train_df[cfg.features].apply(pd.to_numeric, errors="coerce")
    y_train = pd.to_numeric(train_df[cfg.train_target], errors="coerce")
    X_test = test_df[cfg.features].apply(pd.to_numeric, errors="coerce")
    y_test = pd.to_numeric(test_df[cfg.test_target], errors="coerce")

    train_mask = X_train.notna().all(axis=1) & y_train.notna()
    test_mask = X_test.notna().all(axis=1) & y_test.notna()
    X_train = X_train.loc[train_mask].reset_index(drop=True)
    y_train = y_train.loc[train_mask].reset_index(drop=True)
    X_test = X_test.loc[test_mask].reset_index(drop=True)
    y_test = y_test.loc[test_mask].reset_index(drop=True)

    if len(X_train) < cfg.cv_folds:
        raise ValueError(f"Not enough training rows ({len(X_train)}) for cv_folds={cfg.cv_folds}.")
    if len(X_test) < 2:
        raise ValueError(f"Not enough external rows ({len(X_test)}) after filtering.")

    all_rows: List[dict] = []
    for seed in range(cfg.seed_start, cfg.seed_start + cfg.seed_count):
        all_rows.extend(
            evaluate_seed(
                seed=seed,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                candidates=cfg.candidates,
                cv_folds=cfg.cv_folds,
            )
        )

    per_seed = pd.DataFrame(all_rows)
    summary = aggregate_results(per_seed)
    return per_seed, summary


def build_config(args: argparse.Namespace) -> EvalConfig:
    return EvalConfig(
        input_path=Path(args.input).expanduser().resolve(),
        train_sheet=args.train_sheet,
        test_sheet=args.test_sheet,
        train_target=args.train_target,
        test_target=args.test_target,
        features=parse_list_arg(args.features),
        candidates=parse_candidates_arg(args.candidates),
        cv_folds=args.cv_folds,
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        output_dir=Path(args.output_dir).expanduser().resolve(),
        output_stem=args.output_stem,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="External validation robustness for finalist regressors.")
    parser.add_argument("--input", required=True, help="Path to CSV/XLSX input dataset.")
    parser.add_argument("--train-sheet", default="Literature", help="Train sheet name.")
    parser.add_argument("--test-sheet", default="Backvalidation", help="External test sheet name.")
    parser.add_argument("--train-target", default="TREE", help="Train target column.")
    parser.add_argument("--test-target", default="REEY", help="External test target column.")
    parser.add_argument("--features", default="pH,Fe,Mn,Al,SO4", help="Comma-separated features.")
    parser.add_argument(
        "--candidates",
        default="Ridge:winsorize_1_99,ExtraTrees:winsorize_1_99,KNN:none",
        help="Comma-separated Model:outlier_mode entries.",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds on training data.")
    parser.add_argument("--seed-start", type=int, default=0, help="Starting seed (inclusive).")
    parser.add_argument("--seed-count", type=int, default=30, help="How many consecutive seeds to run.")
    parser.add_argument("--output-dir", default="output", help="Output directory for CSVs.")
    parser.add_argument("--output-stem", default="tree_external_backvalidation_finalists_seed0_29", help="CSV stem.")
    args = parser.parse_args()

    cfg = build_config(args)
    per_seed, summary = run(cfg)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = cfg.output_dir / f"{cfg.output_stem}_per_seed.csv"
    summary_path = cfg.output_dir / f"{cfg.output_stem}_summary.csv"
    per_seed.to_csv(per_seed_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Train rows used: {int(per_seed['N_Train'].iloc[0])}")
    print(f"External rows used: {int(per_seed['N_Test'].iloc[0])}")
    print(f"Seeds: {cfg.seed_start}..{cfg.seed_start + cfg.seed_count - 1}")
    print("\nRanking (by Test_R2_P2_5, then Test_R2_Median):")
    print(
        summary[
            [
                "Model",
                "Outlier_Mode",
                "Seeds",
                "Test_R2_Median",
                "Test_R2_P2_5",
                "Test_R2_Mean",
                "Test_R2_Std",
                "Test_RMSE_Median",
                "Test_MAE_Median",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    print(f"\nSaved per-seed results: {per_seed_path}")
    print(f"Saved summary results: {summary_path}")


if __name__ == "__main__":
    main()
