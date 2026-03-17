# DiGiTerra: Handoff Notes for New Developers


## Quick Start (Get It Running in ~2 Minutes)

**For Development/Testing:**
```bash
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser. That’s the web UI. For the desktop app (pywebview wrapper), run `python desktop_app.py` instead.

**For Production Deployment:**
The Docker container uses **gunicorn** (production WSGI server) instead of Flask's development server. For local production testing:
```bash
pip install -r requirements.txt
gunicorn --bind 0.0.0.0:5000 --workers 1 --timeout 120 app:app
```

Use `--workers 1` (not 4+) because the app uses a global `memStorage` dictionary. Multiple workers have separate memory, causing "No data uploaded" errors. See "Production Deployment Notes" below for details.

`python app.py` uses Flask's built-in development server (not suitable for production). Use gunicorn or another production WSGI server for production deployments.

**Running under a URL prefix (e.g. for GKE / `.../researcher-apps/digiterra/`):**  
To serve the app at `http://127.0.0.1:5000/digiterra/` instead of the root, set `URL_PREFIX` before starting:
```bash
export URL_PREFIX=digiterra
python app.py
```
Then open **http://127.0.0.1:5000/digiterra/** . The Dockerfile sets `URL_PREFIX=digiterra` so container runs use the prefix by default.

---

## Repo Layout: Where Things Live

| Path | What it is |
|------|------------|
| **`app.py`** | Web entry point (one-liner). Imports and exposes the Flask app from `core/flask_app.py`. Run: `python app.py` or `gunicorn app:app`. |
| **`desktop_app.py`** | Desktop entry point (one-liner). Calls `core/desktop.py` to start Flask in a thread and open the pywebview window. Run: `python desktop_app.py`. |
| **`core/`** | All app logic that root entry points use: `flask_app.py` (Flask app, create_app, run_app), `desktop.py` (desktop launcher + DesktopApi), plus `constants.py`, `helpers.py`, `state.py` for routes. |
| **`scripts/`** | Standalone dev/maintainer scripts (e.g. `check_requirements.py`). Not imported by the app. App and pipeline code live in `python_scripts/` and `routes/`. |
| **`templates/index.html`** | Single-page UI. All tabs (Upload, Data Exploration, Model Preprocessing, Modeling, Inference) are in this file. |
| **`static/client_side.js`** | Front-end logic: uploads, API calls, progress polling, result display, downloads. |
| **`static/style.css`** | Styles. |
| **`python_scripts/`** | Core ML and preprocessing. `helpers.py`: shared helpers (`preprocess_data`, `prediction`). `config.py`: `VIS_DIR` (output folder). `models/`: regression, classification, and clustering trainers. `preprocessing/` and `plotting/`: pipelines and plots. **`app_model_training.py`**: model training orchestration. **`app_exploration.py`**: data exploration (correlation matrices, pairplot, auto-detect). **`app_prediction.py`**: inference/prediction. All three are invoked by `app.py`. |
| **`deploy/`** | Docker and Kubernetes (Helm) deployment. See `deploy/README.md` for Docker build/run and Helm install. |
| **`examples/`** | Example datasets: 3 classification, 3 regression, 3 clustering. See `examples/README.md`. Not bundled in the app. |
| **`docs/`** | Extra guides: build instructions, feature plans, git sync, etc. |

---

## Splitting app.py

Splitting the main application into components is recommended. It improves navigation and review. The codebase already uses one form of this split; you can extend it in two ways.

**Current structure (refactor completed)**  
Heavy logic has been moved into three modules. **`app_model_training.py`**: `run_model_training(session_id, data, storage_session_id, get_storage)`; all trainer imports and training pipeline. **`app_exploration.py`**: `handle_auto_detect_transformers`, `handle_auto_detect_nan_zeros`, `handle_corr`, `handle_pairplot` for data exploration routes. **`app_prediction.py`**: `run_predict(store, request, ...)` for the inference route. `app.py` keeps thin route handlers that call these and pass in dependencies (e.g. `_get_session_storage`, `_with_prefix`) to avoid circular imports. `app.py` is about 730 lines; the three modules are about 1,600, 290, and 180 lines.

**Ways to split further**

1. **Extract more logic into modules**  
   Move other heavy logic from `app.py` into modules under `python_scripts/` (or a `routes/` package). Keep `app.py` as the place that defines routes and calls into those modules. Pass dependencies (e.g. a `get_storage` callback) from `app.py` into the module so the module does not import `app`, avoiding circular imports.

2. **Flask Blueprints**  
   Group related routes into Blueprints (e.g. `routes/upload.py`, `routes/exploration.py`, `routes/modeling.py`) and register them in `routes/__init__.py` via `register_blueprints(app)`. When `URL_PREFIX` is set, blueprints are registered both at root and under the prefix so both `/` and e.g. `/digiterra/` work (health checks and mounts keep working).

You can use both: route registration in `app.py` or in Blueprint modules, and non-route logic in `python_scripts/` or `routes/` helpers.

**Further reduction (optional)**  
`app.py` is about 730 lines after moving training, exploration, and prediction into modules. The largest remaining block is **Section 3** (`upload_file`, roughly 135 lines). Extracting it would bring `app.py` to about 600 lines. Leaving it as is is fine; the file is manageable with section comments.

---

## Important Behaviors to Know

- **Single-user, in-memory state**  
  `app.py` uses a global `memStorage` dict for models, data, scalers, and feature order. Fine for desktop or a single-user lab tool. 

- **CSV only**  
  Upload and prediction endpoints accept only `.csv`. Validation is in `allowed_file()` and we use `secure_filename()` for safe filenames.
  
- **Outputs**  
  Plots, PDFs, and Excel files go to the user visualizations directory (see “Paths” below). The UI fetches them via `/user-visualizations/<filename>` (for display) and `/download/<path:filename>` (for download). Path traversal is blocked on the download route.

---

## Paths and Environment

- **User visualizations directory** (plots, predictions, etc.): Set via `config.VIS_DIR`; at runtime the app uses `_ensure_user_vis_dir()` in `app.py`. Default paths:
  - macOS: `~/Library/Application Support/DiGiTerra/user_visualizations/`
  - Windows: `%APPDATA%\DiGiTerra\user_visualizations/`
  - Linux: `~/.local/share/DiGiTerra/user_visualizations/`

  Override with `DIGITERRA_OUTPUT_DIR`.

- **Uploads** go to `APP_SUPPORT_DIR / "uploads"` (same base as above, but `uploads` subdir).

- **Base directory** for templates/static can be overridden with `DIGITERRA_BASE_DIR` (useful for PyInstaller builds).

---

## Inference Page and Model Visuals

- The inference results view is titled **"Inference Results"**. Copy and UI labels use "inference" (e.g. "Inference summary", "Run inference on another dataset", "Your inference results for '...' are ready to download").
- Left panel: **Inference summary** table (descriptive stats for predicted values), then **inference visuals**: the same training plot as on the Modeling page (Predicted vs Actual + Residuals in one composite image) with inference points overlaid on the diagonal. The overlay is drawn only over the **left half** of that image (the scatter panel); the right half is the Residuals panel, so the overlay does not cover it. Overlay points are small gray circles (r 0.38, opacity 0.5) so they do not obscure train/test points. Next to it is the **Inference distribution** histogram (x-axis: Predicted units, y-axis: Count). The left column has a max-height and scrolls if needed so the distribution is not cut off; the training-plot block and distribution block use flex so both get space.
- Right panel: training target summary table and the **model used** graphic (same training plot, no overlay). That graphic is generated during training by `visualize_predictions()` and uses the shared plot style. Style is applied via `apply_plot_style()` in `python_scripts/plotting/plot_style.py`; it is called at the start of `visualize_predictions()` and `plot_regression_bundle()` in `utilites.py`. If a saved image still looks like the old style, re-run model training to regenerate it.
- CSS: `.inference-summary-section` uses `max-height: min(85vh, 900px)` and `overflow-y: auto`. `.inference-training-plot-img` is capped at 300px height. `.inference-overlay-svg` is `width: 50%` so it sits over the scatter panel only.

---

## Regression Plots: Redundancy and Options

Regression currently produces several plots:

- **Primary (always):** `target_plot_1` / `target_plot_1_advanced` from `visualize_predictions()`: Predicted vs Actual (train/test) + Test Residuals (+ optional metrics table). This is the main composite shown on the Modeling and Inference pages.
- **From `export_plots` to `plot_regression_bundle()`:** Predicted vs Actual (summary), Residuals histogram, Residuals vs Fitted, Actual vs Predicted density (2D), Feature (or Permutation) importance, and optionally PDPs.

**Redundancy:** The summary "Predicted vs Actual" and "Residuals" in the bundle largely duplicate the composite. The bundle adds Residuals vs Fitted, 2D density, and importance/PDP.

**Ways to reduce and add options:**

1. **Rely on the composite only for the main view** and treat the bundle as "extra exports": generate only importance (+ optionally PDP) by default, and add a UI or config flag to include "full bundle" (all current plots) for power users.
2. **Make the bundle configurable** in the backend: e.g. a list like `regression_plots: ['pred_vs_actual', 'residuals_hist', 'residuals_vs_fitted', 'density', 'importance', 'pdp']` so the pipeline only generates selected items.
3. **Combine into fewer assets:** e.g. one "diagnostics" figure with 2×2 panels (Pred vs Actual, Residuals hist, Res vs Fitted, Density) so there are fewer files and less redundancy with the composite.

Implementing (2) or (3) would require changes in `utilites.plot_regression_bundle()` and possibly the regression pipeline or app config.

---

## Security & Hardening: What’s Done and What to Watch

**Already in place:**

- File type check: only `.csv` allowed; `allowed_file()` + `secure_filename()` on upload and prediction.
- Path traversal fix on **`/download/<path:filename>`**: we resolve paths and ensure the requested file stays under the user visualizations directory (see `download_visualization` in `app.py`).
- CSV read errors are caught and returned as clear HTTP responses instead of 500s.
- Input validation on important routes (preprocess, model training, correlation matrices, pairplot, etc.).


---


## Existing Docs to Use

- **`README.md`**: User-facing run instructions, desktop build, and config (ports, debug, log paths).
- **`docs/documentation.md`**: High-level workflow and concepts.
- **`docs/BUILD_INSTRUCTIONS.md`**: Cross-platform desktop build (PyInstaller).
- **`docs/FEATURE_ADDITION_PLAN.md`**: Ideas and suggested spots for new features.
- **`ISSUES_FOUND.md`**: Historical bug list and fixes; **`DEBUG_REPORT.md`**: Current status and checks.

---

## Handy Commands

```bash
# Dependency check (catches missing imports)
python scripts/check_requirements.py

# Development server (Flask built-in - NOT for production)
python app.py

# Production server (gunicorn - for production)
# NOTE: Using 1 worker due to memStorage limitation (see Production Deployment Notes)
gunicorn --bind 0.0.0.0:5000 --workers 1 --timeout 120 app:app

# Docker build & run (uses gunicorn automatically)
docker build -f deploy/docker/Dockerfile -t digiterra:local .
docker run --rm -p 5000:5000 digiterra:local

# Kubernetes (Helm)
helm install digiterra deploy/helm/digiterra --set image.repository=<your-registry>/digiterra --set image.tag=latest
# With persistence: --set persistence.enabled=true --set persistence.size=5Gi
# See deploy/README.md for options.
```

---

## Production Deployment Notes

- **Development vs Production:** `python app.py` runs Flask's development server (single-threaded, not suitable for production). For production, use **gunicorn** (included in requirements.txt) or another production WSGI server.
- **CRITICAL - Worker Count:** **Always use `--workers 1`** when running gunicorn (both in Docker and when running directly). The app uses a global `memStorage` dictionary for in-memory state. With multiple workers, each process has separate memory, so data uploaded/preprocessed in one worker isn't visible to other workers, causing "No data uploaded" errors when training models.
- **Docker:** The Dockerfile uses gunicorn with **1 worker** (not the typical 4+ workers) for the reason above.
- **Multi-Worker Limitation:** The current architecture (`memStorage` global dict) is designed for single-process use. To support multiple workers/users properly, refactor to:
  - **Session-based storage:** Use Flask sessions with a session store (Redis, database, or encrypted cookies)
  - **Shared storage:** Use Redis or a database to store state that all workers can access
  - **File-based storage:** Persist state to disk (slower but works)
- **Desktop App:** `desktop_app.py` still uses Flask's development server internally (fine for single-user desktop app).

## Summary

- **`app.py`** = HTTP API and route definitions; **`python_scripts/app_model_training.py`** = model training orchestration (invoked by app.py); **`desktop_app.py`** = desktop wrapper.  
- **`memStorage`** = in-memory, single-user, single-process; **MUST be refactored for multi-worker/multi-user web deployments**. Current gunicorn config uses 1 worker to work around this limitation.  
- **Security:** upload validation and download path traversal are addressed; add CSRF, reverse proxy, HTTPS, and dependency pinning for production web use.  
- **Paths:** User visualizations directory (config.VIS_DIR), `DIGITERRA_OUTPUT_DIR`, and `DIGITERRA_BASE_DIR` control where things live.
- **Production:** Use gunicorn for production web deployments. Docker container is configured with gunicorn (1 worker due to memStorage limitation).

