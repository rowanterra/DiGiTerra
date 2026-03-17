# Contributing to DiGiTerra

## Branching and pull requests

Work on a feature branch (e.g. `git checkout -b feature/your-feature`). When ready, push the branch and open a **pull request** against `main` (or `master`). CI will run tests; fix any failures before merging.

## Running tests

From the project root:

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest tests/ -v --tb=short
```

Set `MPLBACKEND=Agg` if you hit display/backend issues. Integration tests under `tests/integration/` use the Flask test client and may require `examples/iris.csv` (skipped if missing). See `docs/DEPENDENCIES.md` for dependency constraints and verification.

## Where things live

- **App entry:** `app.py` (creates Flask app, session cookie, registers blueprints).
- **Routes:** `routes/` package: `main.py` (index, progress SSE), `upload.py`, `exploration.py`, `preprocess.py`, `modeling.py`, `prediction.py`, `assets.py`. Shared state in `app_state.py`, helpers in `app_helpers.py`, constants in `app_constants.py`.
- **Frontend:** `static/js/core.js` (API prefix, DOM refs, utilities, tab/nav), `static/js/app.js` (upload, exploration, preprocess, modeling, inference). Legacy single file: `static/client_side.js`. UI structure: `templates/index.html`.
- **Config and paths:** `python_scripts/config.py` (BASE_DIR, APP_SUPPORT_DIR, UPLOAD_DIR, LOG_DIR, VIS_DIR, URL_PREFIX).
- **Models and training:** `python_scripts/app_model_training.py` (orchestration), **`python_scripts/model_registry.py`** (single place for model lookup and constructor kwargs; see “Adding a new model” below).
- **Plotting:** `python_scripts/plotting/` (bundles, visualize_predictions).
- **Scripts:** `scripts/` for standalone runnable scripts; see `scripts/README.md`.

## Adding a new model

1. **Register the model** in `python_scripts/model_registry.py`:
   - Add an entry to `MODEL_REGISTRY`: key (e.g. `"Ridge"`), value `(problem_type, model_class)`.
   - Implement or extend `get_model_kwargs(model_key, ...)` to return the constructor kwargs for that key (including `random_state` where applicable).
2. **Use the registry in training:** `app_model_training.py` already dispatches via `MODEL_REGISTRY` and `get_model_kwargs`; no new branches needed for standard regression/classification/clustering models.
3. **Optional:** Add a small unit test in `tests/unit/test_model_registry.py` for the new key and `get_model_kwargs` return value.

## Code style and CI

- The repo uses pytest for tests; see `pyproject.toml` for pytest options.
- CI runs on push/PR to `main` or `master` (`.github/workflows/tests.yml`): installs deps and runs `pytest tests/`.

- **API:** JSON success/error shape is described in `docs/API_RESPONSES.md`.

For more on architecture and alignment with standard practice, see `docs/ARCHITECTURE_RECOMMENDATIONS.md` and `docs/SOFTWARE_ALIGNMENT.md`.
