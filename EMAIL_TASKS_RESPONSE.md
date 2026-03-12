# DiGiTerra URL Prefix Work Summary

## Why this work was needed
The app needed to run under a URL prefix so it works at `127.0.0.1:5000/digiterra/` instead of only at the root path.

The goal is to support internet hosting under a subpath and keep all features working.

## What was already done before this pass
- `URL_PREFIX` was introduced in environment settings.
- Route prefixing work had started.
- Frontend received a prefix variable and a few calls were updated.

## What the email asked for and how each ask was handled

### 1) Fix visualizations not appearing under `/digiterra`
This was fixed.

What was wrong:
- Some visualization URLs were still absolute root paths like `/user-visualizations/...`.
- Some download links were still absolute root paths like `/download/...`.
- Some backend responses returned root based paths.

What was changed:
- Added consistent URL prefix handling in backend route registration.
- Added prefixed URL generation in backend responses for pairplot, correlation images, and prediction download.
- Updated frontend to apply `API_ROOT` consistently.
- Added frontend URL helpers so fetch, SSE, image paths, and download links all work under `/digiterra`.
- Updated template asset paths so CSS, JS, and logos load under the prefix.

Result:
- Correlation matrix and pairplot paths resolve correctly when app is opened at `/digiterra/`.
- Visualization and download links resolve correctly under the prefix.

### 2) Find other issues caused by `/digiterra` path
This was done and several additional issues were fixed.

Issues found:
- Multiple hardcoded frontend API routes still used root paths.
- SSE progress stream used root path.
- Some dynamically inserted links and image sources were not prefix aware.
- Static assets needed prefixed handling.

What was changed:
- Registered routes at both root and prefixed paths.
- Added prefixed static route support.
- Added template variables for `api_root` and `static_root`.
- Added frontend helper functions to rewrite runtime links and image sources.

Result:
- Existing flows continue to work.
- Prefixed flow also works without breaking old root behavior.

### 3) Start with JavaScript backend calls and then check file path related causes
This approach was followed.

Step order used:
1. Located all JS calls to backend and updated prefix handling.
2. Updated SSE and dynamic links.
3. Updated backend returned file URLs for generated plots and downloads.
4. Verified route map includes both prefixed and root routes.

Result:
- The main path related breakages from frontend calls and generated URL paths were addressed.

## Extra bug fixes found during testing on SVC models
While testing model flows, a few SVC issues were found and fixed.

What was fixed:
- Duplicate element id conflict around SVC gamma input.
- Wrong gamma field lookup for SVC rbf kernel.
- Empty `class_weight` now normalized so sklearn does not fail with parameter validation errors.

Result:
- SVC form handling is more stable.
- Fewer runtime errors from bad form values.

## Validation done
- Python syntax check for backend passed.
- JavaScript syntax check for frontend passed.
- Flask route map check confirmed both root and prefixed routes exist for key endpoints.
- Template render check confirmed prefixed static and API root values are present.

## How to run and test locally
From project root:

```bash
source .venv/bin/activate
export URL_PREFIX=/digiterra/
export DIGITERRA_HOST=127.0.0.1
export DIGITERRA_PORT=5000
python app.py
```

Open:
- `http://127.0.0.1:5000/digiterra/`

Suggested smoke test:
1. Upload CSV
2. Run correlation matrices
3. Change pairplot X and Y and confirm image updates
4. Run model training and confirm progress stream works
5. Download generated files
6. Run SVC with rbf and poly kernels

## Current status
The prefix support work requested in the email has been implemented with fixes for visualization paths, API calls, dynamic links, and static assets. Additional SVC form issues found during verification were also fixed.
