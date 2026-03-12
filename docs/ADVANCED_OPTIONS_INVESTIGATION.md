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

No test-set information is used for outlier detection, feature selection, or hyperparameter search — **no data leakage**.

---

## 2. Outlier handling

### What it does

- **Detection** is always fit on **training data only** (IQR, Z-Score, Isolation Forest, Local Outlier Factor).
- **Action "Remove"**: Rows marked as outliers are **removed from the training set only**. The test set is **never** row-removed; all test rows are kept for evaluation.
- **Action "Cap"**: Training (and test) values are clipped to bounds derived from **training** statistics (IQR/Z-Score bounds, or 1st/99th percentiles for Isolation Forest/LOF). Test is capped with the same bounds so deployment is consistent.

### How it’s reported

- **Regression**: `outlier_info` is built in `run_regression_pipeline.py` and returned to the app; the response includes `outlier_info` with `method`, `action`, `n_outliers`, `original_samples`, `remaining_samples`. The frontend shows an “Outlier Handling” table (when Advanced/AutoML and `data.outlier_info` is present) with Method, Action, Outliers Detected, Original Samples, Remaining Samples.
- **Classification**: Previously **not** reported: the classification pipeline did not build or return `outlier_info`, so the “Outlier Handling” dropdown/table never appeared for classifiers. This has been fixed so classification now also returns `outlier_info` and the same table is shown when applicable.

### Is “Remove” clear?

- **Yes, it removes rows while running** — but only from the **training** set. Metrics and model are based on (outlier-removed) train and (full) test. The UI could be clarified with a short note, e.g. “Remove: drop outlier rows from **training** data only; test set is unchanged.”

---

## 3. Feature selection

### What it does

- **SelectKBest**: Fit on train with `f_regression` / `f_classif`; same K features applied to train and test.
- **RFE**: Wrapper with RandomForest; fit on train; same selected features applied to train and test.
- **SelectFromModel**: RandomForest-based; fit on train; uses the estimator’s **default threshold** to decide how many features to keep (no explicit K).

### How it’s reported

- **Regression**: `feature_selection_info` (method, k_requested, original_count, selected_count, selected_features) is returned and shown in the “Feature Selection” table in the UI.
- **Classification**: Same as outlier — previously not returned; now the classification pipeline builds and returns `feature_selection_info` so the “Feature Selection” table appears when feature selection was used.

### Inappropriate / unclear behavior

- **SelectFromModel ignores K**: For “Select From Model”, the code does **not** use the user-entered “Number of Features (K)”. The number of features is determined by the model’s default threshold. So if the user sets K=10, that value is **not** used for SelectFromModel. **Recommendation**: Either (a) document that “K” does not apply to Select From Model, or (b) implement threshold tuning / max_features so K is respected (e.g. `SelectFromModel(..., max_features=k)` if supported, or document as “model-based; K not used”).

---

## 4. Hyperparameter search

### What it does

- **Grid search** or **Randomized search** over a fixed param grid; CV is run on the **processed** training data (after outlier and feature selection). Best params are then used to fit on the full processed training set.
- No test data is used in the search — **appropriate**.

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
| Outlier remove      | Yes (rows removed) | No row removal | Yes (`outlier_info`) | Yes (after fix) | Clear once user knows “training only”. |
| Outlier cap         | Bounds from train | Test capped with same bounds | Yes | Yes (after fix) | Appropriate. |
| Feature selection   | Fit on train | Transform only | Yes (`feature_selection_info`) | Yes (after fix) | SelectFromModel ignores K. |
| Hyperparameter search | CV on train | No | Via `model_params` | Via `model_params` | No separate search summary table. |
| Cross-validation    | Yes        | No         | Yes (file + summary) | Yes (file + summary) | As implemented. |

---

## 8. Files touched for classification reporting fix

- `python_scripts/preprocessing/run_classification_pipeline.py`: Build and return `outlier_info` and `feature_selection_info`.
- `python_scripts/helpers.py`: `unpack_classification_result()` extended to return 11 items (including `feature_selection_info`, `outlier_info`).
- `app.py`: All `unpack_classification_result()` call sites unpack the two new values; classification `result` dict includes `'feature_selection_info'` and `'outlier_info'`.

Frontend already supports these keys for the classifier (dropdown + Feature Selection / Outlier Handling tables); no JS changes required once the backend sends the data.
