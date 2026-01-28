# Issues and Bugs Found in DiGiTerra Repository

This document lists issues, bugs, and improvements identified during code review.

**Last Updated:** January 27, 2026

**Handoff:** See **`HANDOFF.md`** for developer notes, security summary, and website-integration guidance.  
**Security Review:** See **`SECURITY_REVIEW.md`** for comprehensive security assessment and recommendations.

---

## Summary

| Severity   | Count | Status    |
|-----------|-------|-----------|
| Critical  | 0     | Resolved  |
| High      | 0     | Resolved  |
| Medium    | 0     | Resolved  |
| Low       | 5     | Code quality |

**Total found:** 15  
**Total fixed:** 10  
**Remaining:** 5 (all low priority, code quality improvements)

**Status:** Production-ready for desktop use. All critical and high-priority issues have been resolved.

---

## Recent Fixes (January 27, 2026)

### Download route path traversal

**Issue:** The `/download/<path:filename>` route used `file_path = USER_VIS_DIR / filename` and `exists()` but did not ensure the resolved path stayed under `USER_VIS_DIR`. Requests such as `/download/../../../etc/passwd` could read files outside the output directory.

**Fix:** Resolve both paths and use `Path.relative_to()` (catch `ValueError`) to enforce that the file is under `USER_VIS_DIR`. Require `file_path.is_file()` so only files are served.

**Location:** `app.py`, `download_visualization` (around line 340).

### Model parameter fixes

**Issue:** Several new models used parameter names that do not match the sklearn API, causing `TypeError: got an unexpected keyword argument`.

**Fixed models:**
1. **RidgeCV:** `store_cv_values` → `store_cv_results`
2. **LassoLarsCV:** Removed deprecated `normalize`
3. **LarsCV:** Removed deprecated `normalize`
4. **LassoLars:** Removed deprecated `normalize`; added `positive`
5. **OrthogonalMatchingPursuit:** Removed deprecated `normalize`
6. **TheilSenRegressor:** Removed invalid `copy_X`
7. **SGDRegressor:** Removed invalid `n_jobs`
8. **RadiusNeighborsRegressor:** Removed invalid `outlier_label`
9. **BaggingRegressor:** Removed deprecated `base_estimator`
10. **BaggingClassifier:** Removed deprecated `base_estimator`
11. **AdaBoostClassifier:** Removed invalid `algorithm`
12. **HDBSCAN:** `core_dist_n_jobs` → `n_jobs`

**Location:** `python_scripts/models/` (regression, classification, cluster `train_*.py`), `templates/index.html`, `static/client_side.js`.

---

## Issues Already Fixed

### 1. File upload path traversal

**Location:** `app.py` lines 342, 1787

**Status:** Fixed. Uses `secure_filename()` from werkzeug.

### 2. Missing file extension validation

**Location:** `app.py` lines 145–158, 338–339

**Status:** Fixed. `allowed_file()` and `ALLOWED_EXTENSIONS = {'csv'}`.

### 3. Hardcoded macOS path

**Location:** `app.py` lines 29–35

**Status:** Fixed. Platform-specific paths (Windows/Linux/macOS).

### 4. Multiple target variables

**Location:** `python_scripts/helpers.py` lines 681–700

**Status:** Fixed. Multiple targets handled correctly.

### 5. Hardcoded threshold in prediction

**Location:** `python_scripts/helpers.py` (previously ~line 478)

**Status:** Fixed. Classification uses proper models; no regression threshold.

### 6. CSV reading error handling

**Location:** `app.py` lines 357–366, 1787–1796

**Status:** Fixed. Try/except with specific error types.

### 7. Type annotation

**Location:** `python_scripts/preprocessing/utilites.py` line 243

**Status:** Fixed. `_fig_save_all_to_pdf` signature corrected.

### 8. Missing input validation

**Location:** `app.py` multiple routes

**Status:** Mostly fixed. Validation on `/preprocess`, `_run_model_training`, `/correlationMatrices`, `/pairplot`, and `stratifyColumn` when relevant.

### 9. Print statements

**Location:** Various

**Status:** Mostly fixed. Removed from `run_clustering_pipeline.py`; utility scripts may still use `print()`.

### 10. File size warnings

**Location:** `app.py` lines 147–150, 350–354

**Status:** Implemented. Advisory size and cell-count warnings only.

---

## Remaining Issues (Low Priority - Code Quality)

These are **not bugs or security issues**, but rather code quality improvements that can be addressed over time.

### 11. Global memory storage

**Location:** `app.py` line 147

**Status:** Not an issue for desktop app - This is an intentional design choice for single-user desktop deployment. The global `memStorage` dictionary stores models, data, and scalers in memory, which is appropriate for desktop use.

**Note:** If deploying as a multi-user web application, this should be replaced with session-based storage or a database. See `SECURITY_REVIEW.md` for details.

### 12. Inconsistent error handling

**Status:** Mostly standardized. Some routes could use more consistent error response formats for better API consistency.

**Priority:** Low - does not affect functionality.

### 13. Missing docstrings

**Status:** Code quality improvement. Add docstrings (Google or NumPy style) for public API functions to improve maintainability.

**Priority:** Low - does not affect functionality.

### 14. Magic numbers

**Status:** Mostly addressed. Many magic numbers have been extracted to constants (see `app.py` lines 152-186). Extract remaining values to constants as you touch code.

**Priority:** Low - does not affect functionality.

### 15. Inconsistent string formatting

**Status:** Style preference only. Code uses a mix of f-strings, `.format()`, and concatenation. Prefer f-strings when modifying code for consistency.

**Priority:** Very low - style only.

---

## Design Choices (Not Issues)

The following are **intentional design decisions**, not bugs or issues:

1. **Global `memStorage`**: Appropriate for single-user desktop app
2. **No authentication**: Desktop apps run locally; OS-level security applies
3. **Flask development server**: Acceptable for desktop; use production WSGI for web
4. **File size warnings (non-restrictive)**: Users may legitimately need large files
5. **No dependency version pinning**: Flexibility vs. reproducibility trade-off

See `SECURITY_REVIEW.md` for detailed security considerations and recommendations for web deployment.

---

## Recommended Priority for Remaining Work

1. Add docstrings (public API first) - improves maintainability
2. Extract remaining magic numbers to constants - improves readability
3. Standardize string formatting (f-strings) when editing - style consistency
4. Align error response formats across routes - API consistency
5. Consider session storage only if deploying as multi-user web - see `SECURITY_REVIEW.md`

---

## Notes

- Critical security and functionality issues are resolved.
- Application is production-ready for desktop use.
- Remaining items are code quality improvements (low priority).
- See `SECURITY_REVIEW.md` for comprehensive security assessment.
- See `DEBUG_REPORT.md` for verification results and current status.
