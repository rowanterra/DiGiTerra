# Contributing to DiGiTerra

## Branches and pull requests

1. Create a feature branch from `main`.
2. Push the branch and open a pull request against `main` (or `master`).
3. Fix any CI failure before merging.

## Tests

From the project root:

```bash
pip install -r requirements.txt -r requirements-dev.txt
MPLBACKEND=Agg pytest tests/ -q
```

`make test` runs the same command. Integration tests under `tests/integration/` use the Flask test client and are skipped when `examples/iris.csv` is missing.

`tests/conftest.py` sets `DIGITERRA_APP_SUPPORT_DIR` and `DIGITERRA_OUTPUT_DIR` to a directory under the system temp directory before the app is imported, so uploads and generated files stay out of the real app-support directory. The log directory is not redirected. A test run writes to `LOG_DIR` from `python_scripts/config.py` (on Linux `~/.cache/DiGiTerra/logs`, on macOS `~/Library/Logs/DiGiTerra`, on Windows `%APPDATA%\DiGiTerra\Logs`).

## Repository layout

| Path | Contents |
|------|----------|
| `app.py` | Web entry point. Imports the Flask app from `core/flask_app.py`. |
| `desktop_app.py` | Desktop entry point. Calls `core/desktop.py`. |
| `core/` | `flask_app.py` (app object, session cookie, upload size limit), `desktop.py` (pywebview window and save dialog), `errors.py` (JSON error handlers), `state.py` (bounded session store), `constants.py`, `helpers.py`. |
| `routes/` | Blueprints: `main.py` (index, progress SSE), `upload.py`, `exploration.py`, `preprocess.py`, `modeling.py`, `prediction.py`, `assets.py`. |
| `python_scripts/` | `config.py` (paths, URL prefix, session output scope), `model_registry.py`, `app_model_training.py`, `app_exploration.py`, `app_prediction.py`, `helpers.py`, `models/`, `preprocessing/`, `plotting/`. |
| `templates/index.html` | The single page. |
| `static/js/` | `init.js`, `core.js`, `upload.js`, `preprocess.js`, `modeling.js`, `inference.js`, `app.js`, and the built `app.bundle.js`. |
| `scripts/` | Maintainer scripts that the app does not import. |
| `tests/` | `unit/` and `integration/`. |
| `deploy/` | Dockerfile and Helm chart. |
| `docs/` | Guides. |

`core.js` defines the API root, fetch wrapper, `showTab`, and shared helpers such as `formatDateTimeForFilename` and `downloadFile`. `upload.js` handles upload and exploration, `preprocess.js` the preprocess form, `modeling.js` training and results, `inference.js` prediction and the AutoML and advanced forms, and `app.js` the welcome screen, navigation, and popups.

## Frontend bundle

`npm run build` writes `static/js/app.bundle.js` from the split files. `npm run lint` builds the bundle, lints the split files, and lints the bundle with `no-undef` on. `npm run check-bundle` fails when the committed bundle differs from a fresh build, so after editing a file under `static/js/`, run `npm run build` and commit the bundle with the change.

## Adding a model

1. Add an entry to `MODEL_REGISTRY` in `python_scripts/model_registry.py`. The key is the model name the page sends (for example `"Ridge"`), and the value is the tuple `(train_function, problem_type)`, where `problem_type` is `"regression"`, `"classification"`, or `"clustering"`.
2. Extend `get_model_kwargs(model_key, ...)` to return the constructor kwargs for the key, including `random_state` where the estimator accepts one.
3. Add a unit test in `tests/unit/test_model_registry.py` for the key and the kwargs.

`app_model_training.py` dispatches through the registry, so no new branch is needed there.

## Style and CI

`ruff check .` uses the configuration in `pyproject.toml` (line length 120, rule sets E, F, I, B, and UP). The baseline is not yet clean, so CI reports ruff findings without failing.

CI (`.github/workflows/tests.yml`) runs on push and pull request to `main` or `master`, on Python 3.11 and 3.12. Each job installs the requirements, runs `scripts/check_requirements.py`, `npm run lint`, `npm run check-bundle`, `ruff check .`, and `pytest tests/`.

The JSON response shape is described in `docs/API_RESPONSES.md`.
