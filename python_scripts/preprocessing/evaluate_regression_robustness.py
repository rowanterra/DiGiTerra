"""
Repeated multi-seed robustness evaluation for regression models.

This utility helps validate that model performance is not an artifact of a
single random seed. For each seed, it:
1) builds a stratified train/test split,
2) runs cross-validation on the train split,
3) evaluates test metrics,
4) repeats for all model and feature-set combinations.

Outputs:
- long-form per-seed results CSV
- aggregated summary CSV (mean/median/std + 95% empirical interval)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


SCALING_MODELS = {
    "LinearRegression",
    "Ridge",
    "ElasticNet",
    "BayesianRidge",
    "SVR_RBF",
    "KNN",
}


@dataclass(frozen=True)
class EvalConfig:
    input_path: Path
    sheet_name: str
    target: str
    features: Sequence[str]
    keep_feature: str
    stratify_col: str
    test_size: float
    cv_folds: int
    seed_start: int
    seed_count: int
    output_dir: Path
    output_stem: str


def parse_list_arg(arg_value: str) -> List[str]:
    return [v.strip() for v in arg_value.split(",") if v.strip()]


def load_dataframe(input_path: Path, sheet_name: str) -> pd.DataFrame:
    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(input_path, sheet_name=sheet_name)
    return pd.read_csv(input_path)


def build_strat_labels(series: pd.Series) -> pd.Series:
    # Use quantile bins first; fallback to equal-width bins when needed.
    for bins in (5, 4, 3, 2):
        try:
            labels = pd.qcut(series, q=bins, labels=False, duplicates="drop")
        except Exception:
            labels = pd.cut(series, bins=bins, labels=False)
        if labels.notna().all():
            counts = labels.value_counts()
            if len(counts) >= 2 and counts.min() >= 2:
                return labels.astype(int)
    raise ValueError("Could not create stable stratification labels from the provided column.")


def make_feature_sets(features: Sequence[str], keep_feature: str) -> Dict[str, List[str]]:
    if keep_feature not in features:
        raise ValueError(f"keep_feature '{keep_feature}' is not in features list: {features}")

    all_set = list(features)
    feature_sets: Dict[str, List[str]] = {"All_features": all_set}
    droppable = [f for f in all_set if f != keep_feature]
    for feat in droppable:
        keep = [f for f in all_set if f != feat]
        feature_sets[f"Drop_{feat}_keep_{keep_feature}"] = keep
    return feature_sets


def get_models(seed: int) -> Dict[str, RegressorMixin]:
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=seed),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=seed, max_iter=10000),
        "BayesianRidge": BayesianRidge(),
        "SVR_RBF": SVR(C=10.0, epsilon=0.1, kernel="rbf"),
        "KNN": KNeighborsRegressor(n_neighbors=7, weights="distance"),
        "GradientBoosting": GradientBoostingRegressor(random_state=seed),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=seed),
        "RandomForest": RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=500, random_state=seed, n_jobs=-1),
    }


def make_pipeline(model_name: str, model: RegressorMixin) -> Pipeline:
    steps: List[Tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if model_name in SCALING_MODELS:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def evaluate_seed(
    seed: int,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    strat_labels: pd.Series,
    feature_sets: Dict[str, List[str]],
    cv_folds: int,
    test_size: float,
) -> List[dict]:
    idx = np.arange(len(X_full))
    idx_train, idx_test = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        stratify=strat_labels,
    )

    y_train = y_full.iloc[idx_train]
    y_test = y_full.iloc[idx_test]

    results: List[dict] = []
    models = get_models(seed)
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    for fs_name, features in feature_sets.items():
        X_train = X_full.iloc[idx_train][features]
        X_test = X_full.iloc[idx_test][features]

        for model_name, model in models.items():
            pipe = make_pipeline(model_name, model)
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)

            results.append(
                {
                    "Seed": seed,
                    "Model": model_name,
                    "Feature_Set": fs_name,
                    "Features": ", ".join(features),
                    "N_Train": int(len(idx_train)),
                    "N_Test": int(len(idx_test)),
                    "CV_R2_Mean": float(np.mean(cv_scores)),
                    "CV_R2_Std": float(np.std(cv_scores)),
                    "Test_R2": float(r2_score(y_test, pred)),
                    "Test_RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                    "Test_MAE": float(mean_absolute_error(y_test, pred)),
                }
            )
    return results


def aggregate_results(per_seed_results: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["Model", "Feature_Set", "Features"]
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

    q = (
        per_seed_results.groupby(group_cols)["Test_R2"]
        .quantile([0.025, 0.975])
        .unstack()
        .reset_index()
    )
    q.columns = group_cols + ["Test_R2_P2_5", "Test_R2_P97_5"]
    summary = summary.merge(q, on=group_cols, how="left")
    return summary.sort_values(by="Test_R2_Median", ascending=False).reset_index(drop=True)


def build_config(args: argparse.Namespace) -> EvalConfig:
    return EvalConfig(
        input_path=Path(args.input).expanduser().resolve(),
        sheet_name=args.sheet,
        target=args.target,
        features=parse_list_arg(args.features),
        keep_feature=args.keep_feature,
        stratify_col=args.stratify_col,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        output_dir=Path(args.output_dir).expanduser().resolve(),
        output_stem=args.output_stem,
    )


def run(cfg: EvalConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = load_dataframe(cfg.input_path, cfg.sheet_name).copy()

    required = list(cfg.features) + [cfg.target, cfg.stratify_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[list(cfg.features)].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[cfg.target], errors="coerce")
    strat_source = pd.to_numeric(df[cfg.stratify_col], errors="coerce")
    mask = X.notna().all(axis=1) & y.notna() & strat_source.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    strat_source = strat_source.loc[mask].reset_index(drop=True)

    if len(X) < 20:
        raise ValueError(f"Not enough rows after filtering: {len(X)}")

    strat_labels = build_strat_labels(strat_source)
    feature_sets = make_feature_sets(cfg.features, cfg.keep_feature)

    all_rows: List[dict] = []
    for seed in range(cfg.seed_start, cfg.seed_start + cfg.seed_count):
        rows = evaluate_seed(
            seed=seed,
            X_full=X,
            y_full=y,
            strat_labels=strat_labels,
            feature_sets=feature_sets,
            cv_folds=cfg.cv_folds,
            test_size=cfg.test_size,
        )
        all_rows.extend(rows)

    per_seed = pd.DataFrame(all_rows)
    summary = aggregate_results(per_seed)
    return per_seed, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Regression robustness check across multiple seeds.")
    parser.add_argument("--input", required=True, help="Path to CSV/XLSX input dataset.")
    parser.add_argument("--sheet", default="Literature", help="Excel sheet name (ignored for CSV).")
    parser.add_argument("--target", required=True, help="Target column name (for example: TREE).")
    parser.add_argument(
        "--features",
        default="pH,Fe,Mn,Al,SO4",
        help="Comma-separated feature columns.",
    )
    parser.add_argument("--keep-feature", default="pH", help="Feature that must always remain in ablations.")
    parser.add_argument("--stratify-col", default="pH", help="Column used to build stratification labels.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--seed-start", type=int, default=0, help="Starting seed (inclusive).")
    parser.add_argument("--seed-count", type=int, default=30, help="How many consecutive seeds to run.")
    parser.add_argument(
        "--output-dir",
        default="docs",
        help="Directory where output CSV files will be written.",
    )
    parser.add_argument(
        "--output-stem",
        default="regression_robustness",
        help="Prefix used for output CSV file names.",
    )
    args = parser.parse_args()
    cfg = build_config(args)

    per_seed, summary = run(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = cfg.output_dir / f"{cfg.output_stem}_per_seed.csv"
    summary_path = cfg.output_dir / f"{cfg.output_stem}_summary.csv"
    per_seed.to_csv(per_seed_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Rows used: {per_seed[['N_Train', 'N_Test']].iloc[0].sum()}")
    print(f"Seeds: {cfg.seed_start}..{cfg.seed_start + cfg.seed_count - 1}")
    print("\nTop 10 by median Test_R2:")
    print(
        summary[
            [
                "Model",
                "Feature_Set",
                "Features",
                "Seeds",
                "Test_R2_Median",
                "Test_R2_Mean",
                "Test_R2_Std",
                "Test_R2_P2_5",
                "Test_R2_P97_5",
            ]
        ]
        .head(10)
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    print(f"\nSaved per-seed results: {per_seed_path}")
    print(f"Saved summary results: {summary_path}")


if __name__ == "__main__":
    main()
