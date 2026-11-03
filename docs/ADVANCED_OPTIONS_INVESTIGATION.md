# Advanced modeling options

How feature selection, outlier handling, hyperparameter search, and cross-validation run in the regression and classification pipelines, and how each is reported. Clustering is covered at the end.

## Pipeline order

For regression and classification, `run_regression_pipeline.py` and `run_classification_pipeline.py` under `python_scripts/preprocessing/` run in this order:

1. Preprocess (train/test split, transform, scale).
2. Outlier handling, fit on the training rows.
3. Feature selection, fit on the training rows and applied to both sets.
4. Hyperparameter search, with cross-validation on the processed training rows.
5. Final fit on the processed training rows.
6. Evaluation on the test rows.

Outlier detection, feature selection, and hyperparameter search use the training rows only. Mean and median imputation, when selected, is computed on the whole dataset before the split, so the test rows inform the fill values used in training. This is an open item.

## Outlier handling

Detection (IQR, Z-score, Isolation Forest, Local Outlier Factor) is fit on the training rows.

With action Remove, the rows marked as outliers are removed from the training set. The test set keeps every row.

With action Cap, training and test values are clipped to bounds computed from training statistics (IQR or Z-score bounds, or the 1st and 99th percentiles for Isolation Forest and LOF).

Both pipelines return `outlier_info` with `method`, `action`, `n_outliers`, `original_samples`, `remaining_samples`, and, for Remove, `removed_row_indices`. The page shows an Outlier Handling table when `outlier_info` is present. The Excel export (`write_to_excelRegression` and `write_to_excelClassifier` in `python_scripts/helpers.py`) writes an Outlier Handling sheet and, when rows were removed, an Outlier Samples Removed sheet with the training row indices.

## Feature selection

`python_scripts/preprocessing/feature_selection.py` implements three methods. SelectKBest scores with `f_regression` or `f_classif` and keeps K features. RFE wraps a random forest and keeps K features. SelectFromModel wraps a random forest and keeps the features above the estimator's default importance threshold. The K value from the page is not passed to `SelectFromModel`, so the count of kept features is not K for that method.

Both pipelines return `feature_selection_info` with `method`, `k_requested`, `original_count`, `selected_count`, and `selected_features`, and the page shows a Feature Selection table when feature selection ran. The `feature_names` recorded in the model artifact are the columns the model was fit on, so importance charts label the selected features.

## Hyperparameter search

Grid search or randomized search runs over a fixed parameter grid with cross-validation on the processed training rows. The best parameters are used for one final fit. The search's `random_state` and the CV fold shuffle use the run seed. The best parameters appear in `model_params` and in the hyperparameter table. There is no separate search summary.

## Cross-validation

Cross-validation runs when enabled in Advanced or AutoML mode, over the training rows with the same preprocessing and model. RMSE and MAE are reported with a positive sign. The response includes `cross_validation_file` and `cross_validation_summary`, and the page lists Cross Validation in the additional-tables menu when the summary is present.

## NaN fallback in regression

When NaN values remain in `X_train_s` after the transforms, `run_regression_pipeline.py` fills each numeric column with the column mean, or 0 when the whole column is NaN. The fill values are recorded and reused at inference. The fill is not a user-selected option, and the page does not display a message when the fill runs.

## Summary

| Option | Fit on | Applied to test | Reported |
|--------|--------|-----------------|----------|
| Outlier remove | training rows | no rows removed | `outlier_info`, page table, Excel sheets |
| Outlier cap | training bounds | capped with the same bounds | `outlier_info` |
| Feature selection | training rows | transform only | `feature_selection_info` (SelectFromModel ignores K) |
| Hyperparameter search | CV on training rows | no | `model_params` |
| Cross-validation | training rows | no | file and summary |

## Clustering

`app_model_training.py` builds a separate keyword set for clustering that does not include feature selection, outlier handling, hyperparameter search, or cross-validation, so those options have no effect on a clustering run. The progress panel still shows the stages, and the AutoML summary still lists them. This is an open item.
