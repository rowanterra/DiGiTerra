# Advanced Modeling Options: Investigation & Reporting

This document summarizes how advanced options (Feature Selection, Outlier Handling, Hyperparameter Search, Cross-Validation) work, how their results are reported, and any inappropriate or unclear behavior.

---

## 1. Pipeline order (all task types)

For both **regression** and **classification**, the order is:

1. **Preprocess** (train/test split, transform, scale)
2. **Outlier handling** (fit on train only; apply action to train; test is never row-removed)
3. **Feature selection** (fit on train; transform train and test)
4. **Hyperparameter search** (CV on the **processed** training data only)
5. **Fit** final model on processed training data
6. **Evaluate** on test set

No test-set information is used for outlier detection, feature selection, or hyperparameter search. No data leakage.

---

## 2. Outlier handling

### What it does

- **Detection** is always fit on **training data only** (IQR, Z-Score, Isolation Forest, Local Outlier Factor).
- **Action "Remove"**: Rows marked as outliers are **removed from the training set only**. The test set is **never** row-removed; all test rows are kept for evaluation.
- **Action "Cap"**: Training (and test) values are clipped to bounds derived from **training** statistics (IQR/Z-Score bounds, or 1st/99th percentiles for Isolation Forest/LOF). Test is capped with the same bounds so deployment is consistent.

### How it’s reported

- **Regression**: `outlier_info` is built in `run_regression_pipeline.py` (method, action, n_outliers, original_samples, remaining_samples, and when action is "remove", removed_row_indices). The frontend shows an Outlier Handling table when Advanced/AutoML and `data.outlier_info` is present. The Excel export includes an Outlier Handling sheet and an Outlier Samples Removed sheet listing removed training row indices.
- **Classification**: Same as regression. The pipeline builds and returns `outlier_info`; the same UI table and Excel sheets (Outlier Handling, Outlier Samples Removed) are written when applicable.

### Is “Remove” clear?

- Rows are removed from the training set only. Metrics and model use outlier-removed train and full test. The UI can add a short note: “Remove: drop outlier rows from **training** data only; test set is unchanged.”

---

## 3. Feature selection

### What it does

- **SelectKBest**: Fit on train with `f_regression` / `f_classif`; same K features applied to train and test.
- **RFE**: Wrapper with RandomForest; fit on train; same selected features applied to train and test.
- **SelectFromModel**: RandomForest-based; fit on train; uses the estimator’s **default threshold** to decide how many features to keep (no explicit K).

### How it’s reported

- **Regression**: `feature_selection_info` (method, k_requested, original_count, selected_count, selected_features) is returned and shown in the “Feature Selection” table in the UI.
- **Classification**: The classification pipeline builds and returns `feature_selection_info`; the “Feature Selection” table appears when feature selection was used.

### Inappropriate / unclear behavior

- **SelectFromModel ignores K**: For “Select From Model”, the code does **not** use the user-entered “Number of Features (K)”. The number of features is determined by the model’s default threshold. So if the user sets K=10, that value is **not** used for SelectFromModel. **Recommendation**: Either (a) document that “K” does not apply to Select From Model, or (b) implement threshold tuning / max_features so K is respected (e.g. `SelectFromModel(..., max_features=k)` if supported, or document as “model-based; K not used”).

---

## 4. Hyperparameter search

### What it does

- **Grid search** or **Randomized search** over a fixed param grid; CV is run on the **processed** training data (after outlier and feature selection). Best params are then used to fit on the full processed training set.
- No test data is used in the search.

### How it’s reported

- Best parameters are reflected in `model_params` (and in the displayed hyperparameter table). There is no separate “hyperparameter search” summary table (e.g. best score per fold); only the chosen hyperparameters and the final model metrics are shown.

---

## 5. Cross-validation

### What it does

- Run only when the user enables it (Advanced/AutoML). Uses the same preprocessing + model; CV is over the **training** data (or full dataset depending on implementation). Results are written to a separate file and the frontend can show a “Cross Validation” table when `cross_validation_summary` is present.

### How it’s reported

- `cross_validation_file` and `cross_validation_summary` in the response; “Cross Validation” option in the additional-tables dropdown when summary data exists.

---

## 6. Regression-specific: NaN fallback

- In `run_regression_pipeline.py`, if NaNs appear in `X_train_s` after transforms, the code attempts to fill numeric columns with column **means** (or 0 if all NaN). This is a **silent fallback** and not a user-chosen imputation strategy. **Recommendation**: Log a warning and/or surface a short message so users know imputation was applied (e.g. “Missing values in transformed features were filled with column means for training”).

---

## 7. Summary table

| Option              | Train only? | Test used? | Reported (regression) | Reported (classification) | Notes |
|---------------------|------------|------------|------------------------|----------------------------|-------|
| Outlier remove      | Yes (rows removed) | No row removal | Yes (UI + Excel with removed indices) | Yes | Training only. |
| Outlier cap         | Bounds from train | Test capped with same bounds | Yes | Yes | Same bounds for test. |
| Feature selection   | Fit on train | Transform only | Yes (`feature_selection_info`) | Yes | SelectFromModel ignores K. |
| Hyperparameter search | CV on train | No | Via `model_params` | Via `model_params` | No separate search summary table. |
| Cross-validation    | Yes        | No         | Yes (file + summary) | Yes (file + summary) | As implemented. |

---

## 8. Outlier and feature selection reporting (regression and classification)

- **Pipelines**: `run_regression_pipeline.py` and `run_classification_pipeline.py` build `outlier_info` (method, action, n_outliers, original_samples, remaining_samples, and when action is "remove", removed_row_indices). Both build `feature_selection_info` when feature selection is used.
- **Excel**: `write_to_excelRegression` and `write_to_excelClassifier` in `python_scripts/helpers.py` accept `outlier_info`. When present, they write an "Outlier Handling" sheet and, when removed_row_indices exists, an "Outlier Samples Removed" sheet.
- **App**: Classification and regression result dicts include `feature_selection_info` and `outlier_info`. The frontend shows the Outlier Handling and Feature Selection tables when these keys are present.
