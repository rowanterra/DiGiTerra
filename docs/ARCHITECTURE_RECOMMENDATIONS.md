# DiGiTerra: Architecture Recommendations (More Straightforward)

This document summarizes the current architecture and recommends changes to make the project easier to navigate, extend, and maintain. Focus is on **simpler control flow**, **less duplication**, and **clearer boundaries**.

---

## Current Layout (Quick Reference)

| Component | Location | Size | Role |
|-----------|----------|------|------|
| **Routes & app** | `app.py` | ~740 lines | All HTTP routes, session, upload/preprocess/process/predict; calls into python_scripts |
| **Desktop launcher** | `desktop_app.py` | ~230 lines | Starts Flask in thread, pywebview, save-file API |
| **Training orchestration** | `python_scripts/app_model_training.py` | ~1,660 lines | Single entry `run_model_training`; **69 `elif modelName == ...` branches** that call individual `train_*` functions |
| **Pipelines** | `python_scripts/preprocessing/run_*_pipeline.py` | Various | Actual fit/predict logic: `run_regression`, `run_classification`, `run_clustering` |
| **Model wrappers** | `python_scripts/models/{regression,classify,cluster}_models/train_*.py` | ~30–80 lines each | Thin wrappers: build sklearn model, call corresponding pipeline |
| **Frontend logic** | `static/client_side.js` | **~8,316 lines** | Upload, exploration, preprocess, modeling, inference, all UI state and result rendering in one file |
| **UI markup** | `templates/index.html` | **~7,474 lines** | Single-page app: all tabs and sections in one HTML file |
| **Helpers** | `python_scripts/helpers.py` | ~772 lines | `preprocess_data`, `prediction`, Excel writers, etc. |

---

## Problems That Make It Less Straightforward

1. **Model dispatch is one giant if/elif**  
   Adding a model requires editing the 1,600+ line orchestrator in two places (branch + hyperparameter coercion). Easy to miss one or introduce inconsistency.

2. **One huge JS file**  
   All behavior lives in `client_side.js`. Hard to find “where does inference run?” or “where is the cluster result table built?” without searching. No clear modules or boundaries.

3. **One huge HTML file**  
   All tabs and panels in a single 7k-line template. Editing one flow (e.g. Modeling) means scrolling through unrelated markup.

4. **All routes in one file**  
   `app.py` does a lot. HANDOFF.md already suggests Blueprints; splitting by domain (upload, exploration, modeling, inference, assets) would make routes easier to find and test.

5. **Duplicate path/platform logic**  
   `app.py` and `desktop_app.py` both define or use `APP_SUPPORT_DIR`, `LOG_DIR`, etc. Config lives in more than one place.

6. **Visualization and script code is scattered**  
   Plot generation lives in `python_scripts/plotting/` but bundle logic (e.g. `plot_clustering_bundle`, `plot_regression_bundle`) and `visualize_predictions.py` live under `preprocessing/`. Standalone or utility scripts don’t have a single obvious place.

---

## Folder structure: visualizations, scripts, model types

Use **dedicated folders** so “where does X live?” has a clear answer.

### Visualizations folder

**Goal:** One place for all code that produces figures (plots, charts, exports).

- **Keep and treat as canonical:** `python_scripts/plotting/` (or rename to `python_scripts/visualizations/` if you prefer that name).
- **Put inside it:**
  - All plot modules (e.g. `plot_roc_curve.py`, `plot_confusion_matrix.py`, `plot_style.py`, `plot_shap_summary_graphic.py`, etc.).
  - Any “bundle” logic that generates multiple figures for a problem type. Today `plot_clustering_bundle`, `plot_regression_bundle`, and `plot_classification_bundle` live in `preprocessing/utilites.py`; consider moving them into the visualizations folder (e.g. `plotting/bundles.py` or `visualizations/regression_bundle.py`, `visualizations/classification_bundle.py`, `visualizations/clustering_bundle.py`) so everything that draws is in one place.
  - `visualize_predictions.py` (regression training plot) can move here as well (e.g. `plotting/regression_training.py` or under `visualizations/`).
- **Result:** “Where is the cluster silhouette plot defined?” → look in `python_scripts/plotting/` (or `visualizations/`). No hunting through preprocessing.

### Scripts folder

**Goal:** A single place for **runnable utility scripts** (one-off jobs, dev helpers, data prep, build-related scripts) so they’re not mixed with importable app code.

- **Add a top-level folder:** `scripts/` (or `python_scripts/scripts/` if you want them under python_scripts).
- **Use it for:** Standalone Python (or shell) scripts that are run directly (e.g. `python scripts/export_example_data.py`), not imported by the app. Keep `app.py`, `desktop_app.py`, and `build/*.sh` where they are; move or add one-off utilities into `scripts/`.
- **Result:** App code stays in `app.py`, `python_scripts/`, etc.; “run a quick script” means `scripts/` and nothing else.

### Model types folder (already in place — keep it explicit)

**Goal:** One folder per **model type** (problem type), so adding a new model or a new problem type has a clear location.

- **Current layout (good):**
  - `python_scripts/models/regression_models/` – all regression `train_*.py` modules.
  - `python_scripts/models/classify_models/` – all classification `train_*.py` modules.
  - `python_scripts/models/cluster_models/` – all clustering `train_*.py` modules.
- **Recommendation:** Keep this structure and document it. When adding a **new model**, add one file to the correct type folder (e.g. `train_foo.py` in `regression_models/`). If you ever add a new **problem type** (e.g. ranking), add a new folder: `python_scripts/models/ranking_models/` and register it in the app/orchestrator.
- **Result:** “Where do regression models live?” → `models/regression_models/`. “Where do I add a new cluster model?” → `models/cluster_models/train_newname.py`.

### Target folder layout (summary)

```
DiGiTerra/
├── app.py
├── desktop_app.py
├── scripts/                          # Standalone runnable scripts (utility, dev, one-off)
├── routes/                           # Optional: Blueprints (upload, exploration, modeling, inference, assets)
├── templates/
├── static/
│   └── (client_*.js modules when split)
└── python_scripts/
    ├── config.py
    ├── helpers.py
    ├── app_model_training.py
    ├── app_exploration.py
    ├── app_prediction.py
    ├── models/                        # Model types: one folder per problem type
    │   ├── regression_models/
    │   ├── classify_models/
    │   └── cluster_models/
    ├── plotting/                     # Or visualizations/ — all plot/bundle code
    │   ├── plot_style.py
    │   ├── plot_*.py
    │   └── (bundles: regression, classification, clustering)
    └── preprocessing/                # Pipelines, data prep, no plot generation
        ├── run_regression_pipeline.py
        ├── run_classification_pipeline.py
        ├── run_clustering_pipeline.py
        └── ...
```

---

## Recommendations (Prioritized)

### 1. **Model registry instead of 69 if/elif branches** (High impact)

**Idea:** Replace the long `if modelName == ... elif modelName == ...` chain with a **registry**: a single dict mapping `model_name → (train_function, problem_type, optional_default_hyperparams)`.

- **Dispatch:** `train_fn, problem_type, _ = MODEL_REGISTRY[modelName]` then call `train_fn(...)`.
- **Adding a model:** Add one entry to the registry and one `train_*.py` file. No touch to the central dispatch loop.
- **Hyperparameters:** Either keep a small “parser” per problem type (regression / classification / clustering) that turns request payload into kwargs, or store per-model default/parsing in the registry.

**Result:** `app_model_training.py` shrinks dramatically; one clear place to register models; fewer merge conflicts and mistakes when adding models.

---

### 2. **Split the frontend into multiple JS modules** (High impact)

**Idea:** Break `client_side.js` into several files by **feature** (not by “all in one”). Load them in order via `<script>` tags, or introduce a minimal bundler later.

Suggested split:

- **`client_config.js`** – `API_ROOT`, `withApiRoot`, fetch/EventSource wrapping, URL rewriting.
- **`client_ui.js`** – Tabs, show/hide, focus, accessibility, header offset, shared DOM helpers.
- **`client_upload.js`** – Upload form, file handling, column selection.
- **`client_exploration.js`** – Correlation matrices, pairplot, auto-detect.
- **`client_preprocess.js`** – Preprocess form, train/test size, indicators/predictors, “Process” flow.
- **`client_modeling.js`** – Model choice, simple/advanced/automl, “Run” training, progress, result display (regression / classification / cluster).
- **`client_inference.js`** – Prediction upload, run inference, inference results UI.

Each file can attach to a single namespace (e.g. `window.DiGiTerra`) to avoid globals. No framework required; same HTML, just more `<script src="...">` (or one bundle).

**Result:** “Where is cluster result HTML built?” → open `client_modeling.js`. Easier to work on one flow without scrolling 8k lines; easier to test or stub one area.

---

### 3. **Use Flask Blueprints for routes** (Medium impact)

**Idea:** Group routes as in HANDOFF.md:

- `routes/upload.py` – upload, file validation.
- `routes/exploration.py` – correlation, pairplot, auto-detect.
- `routes/modeling.py` – preprocess, process (run training), progress stream.
- `routes/inference.py` – predict.
- `routes/assets.py` – static, user-visualizations, download.

Register in `app.py` with the same `URL_PREFIX` so existing URLs still work. Keep `app.py` to: create app, apply prefix/session middleware, register blueprints, optionally call `create_app()`.

**Result:** Smaller `app.py`; each file is “everything for upload” or “everything for modeling.” Clearer for new contributors and for debugging.

---

### 4. **Split the single HTML template into partials** (Medium impact)

**Idea:** Replace one 7k-line `index.html` with a small main template that **includes** Jinja2 partials, e.g.:

- `_header.html` – logo, nav tabs, documentation button.
- `_upload_tab.html` – upload tab content.
- `_exploration_tab.html` – data exploration tab.
- `_model_preprocessing_tab.html` – model preprocessing tab.
- `_modeling_tab.html` – modeling (simple/advanced/automl) and result placeholders.
- `_inference_tab.html` – inference upload and results.
- `_documentation_tab.html` – documentation content.

Main `index.html`: layout shell + `{% include ... %}` for each. No change to URLs or behavior.

**Result:** Edit “modeling” markup in one small file; less scrolling; clearer ownership of sections.

---

### 5. **Single pipeline entry per problem type + model factory** (Medium impact, bigger refactor)

**Idea:** Today: `app_model_training` → 69 branches → 69 different `train_*` functions → each calls `run_regression` / `run_classification` / `run_clustering`. The real logic lives in the pipelines; the `train_*` files are thin (build model, call pipeline).

**Straightforward alternative:**  
`app_model_training` has **three** high-level branches: regression, classification, clustering. Each branch:

1. Calls a **model factory** (e.g. `get_model(model_name, hyperparameters, problem_type)`) that returns the sklearn model (and maybe minimal metadata).
2. Calls **one** pipeline: `run_regression(...)`, `run_classification(...)`, or `run_clustering(...)` with that model.

All model-specific hyperparameter parsing (and defaults) move into the factory or into small per-model configs the factory reads. The 69 `train_*` modules can become **config + one-liner** (e.g. “build this model with these kwargs”) or be inlined into the factory.

**Result:** One place that knows “how to build a model by name”; one place that knows “how to run regression/classification/clustering.” No 69-branch if/elif in the orchestrator. Pipelines stay as they are; only the way the model is created and passed in changes.

---

### 6. **Centralize paths and config** (Lower effort)

**Idea:** Put all path and env-derived config in one module (e.g. `python_scripts/config.py` or a dedicated `paths.py`):

- `BASE_DIR`, `APP_SUPPORT_DIR`, `UPLOAD_DIR`, `VIS_DIR`, `LOG_DIR`, `URL_PREFIX`.

Have `app.py` and `desktop_app.py` **import** these (and set `VIS_DIR` once at startup) instead of redefining platform logic in both. Document overrides (e.g. `DIGITERRA_OUTPUT_DIR`, `URL_PREFIX`) in one place.

**Result:** Single source of truth for “where things live”; less duplication and fewer surprises when adding a new path.

---

### 7. **Extract upload handler** (Low effort)

**Idea:** As in HANDOFF, move the large `upload_file` implementation from `app.py` into a helper (e.g. `python_scripts/app_upload.py` or under a `routes/` package). The route in `app.py` (or in a Blueprint) becomes a thin wrapper that calls `handle_upload(request, get_storage, ...)` and returns the response.

**Result:** Shorter `app.py`; upload logic in one testable module.

---

### 8. **Fix typo: `utilites.py` → `utilities.py`** (Low effort)

**Idea:** Rename `python_scripts/preprocessing/utilites.py` to `utilities.py` and update all imports. Purely for clarity and consistency.

---

## Suggested Order of Work

1. **Quick wins:** Centralize config/paths (#6), extract upload handler (#7), rename utilites (#8).
2. **Folders:** Add `scripts/` for standalone runnable scripts; move plot bundles and `visualize_predictions` into `python_scripts/plotting/` (or `visualizations/`) so all visualization code lives in one folder; keep model-type folders explicit (`regression_models/`, `classify_models/`, `cluster_models/`).
3. **High impact:** Model registry (#1), then split JS (#2). These reduce the two biggest “one giant block” pain points.
4. **Structure:** Blueprints (#3), HTML partials (#4). Better navigation without changing behavior.
5. **Larger refactor (optional):** Single pipeline entry + model factory (#5) for a minimal “three-branch” orchestrator and one place to add models.

---

## What to Leave As-Is (For Now)

- **Pipelines** (`run_regression`, `run_classification`, `run_clustering`) – Already clear boundaries; keep them.
- **Individual `train_*.py` files** – They’re thin and consistent. A registry or factory can wrap them; no need to delete them until you adopt #5.
- **Single-page app behavior** – No need to switch to a heavy frontend framework; modular JS and HTML partials already make the SPA more straightforward.
- **In-memory session state** – Fine for single-user/desktop; HANDOFF already notes moving to session/DB only when going multi-user.

These recommendations aim to make the codebase more straightforward to work in without a full rewrite: fewer giant files, one place to add models, and clearer boundaries between upload, exploration, modeling, and inference.
