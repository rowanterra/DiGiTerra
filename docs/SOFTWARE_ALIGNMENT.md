# DiGiTerra: Aligning With Standard Software Practice

Practical next steps to make the project more in line with typical software engineering: testability, repeatability, maintainability, and production readiness.

---

## 1. Testing

**Current:** One manual flow test (`tests/integration/test_flow.py`: upload, correlation, preprocess). No pytest, no unit tests, no coverage.

**Align:**

- **Add pytest** to `requirements.txt` (or `requirements-dev.txt`) and add a `tests/` directory.
- **Unit tests** for pure logic that doesn’t need Flask:
  - `python_scripts/preprocessing/utilities.py`: e.g. `sort_class_labels_numeric_bins`, `choose_columns_from_df`, scaling/preprocessor helpers.
  - `python_scripts/model_registry.py`: `get_model_kwargs` for a few models (e.g. Ridge, kmeans) and that `MODEL_REGISTRY` has expected keys.
  - `python_scripts/config.py`: path resolution (e.g. `VIS_DIR`, `APP_SUPPORT_DIR`) under a temporary or env override.
- **Integration tests** (Flask test client):
  - Expand `tests/integration/test_flow.py` or add more under `tests/`: upload → preprocess → run one regression and one classification model (small data), assert response shape and no 500.
- **Optional:** `pytest.ini` or `pyproject.toml` with `[tool.pytest.ini_options]` and a `tests/` layout (e.g. `tests/unit/`, `tests/integration/`).

**Result:** Changes can be validated automatically; refactors are safer.

---

## 2. Dependencies

**Current:** `requirements.txt` with no version pins (“No version pins” in docs/HANDOFF.md).

**Align:**

- **Pin versions** (at least major.minor) for repeatable installs and fewer “works on my machine” issues, e.g.:
  - `flask>=3.0,<4`
  - `scikit-learn>=1.3,<2`
  - `pandas>=2.0`
  - etc.
- **Optional:** Split into `requirements.txt` (runtime, pinned) and `requirements-dev.txt` (pytest, lint, type-check).
- **Optional:** Add `pyproject.toml` with `[project]` and optional `[tool.setuptools]` or use it only for tool config (pytest, mypy, ruff).

**Result:** Builds and CI are reproducible; upgrades are explicit.

---

## 3. CI (Continuous Integration)

**Current:** No `.github/workflows/`, Jenkins, or other CI.

**Align:**

- **GitHub Actions** (if the repo is on GitHub):
  - On push/PR: install deps, run pytest, optionally a linter (e.g. ruff or flake8).
  - Optional: run `pytest tests/integration/` (or the expanded integration test) with a small example CSV.
- **Minimal workflow:** one job that `pip install -r requirements.txt` and `pytest tests/` (or `pytest tests/`).

**Result:** Every push/PR is checked automatically; regressions are caught early.

---

## 4. Structure (already partially done)

**Done:** Config centralized, model registry, plotting/scripts/model-type folders, utilities rename.

**Still useful:**

- **Flask Blueprints:** Split routes in `app.py` into e.g. `routes/upload.py`, `routes/modeling.py`, `routes/inference.py`, `routes/assets.py` and register with `url_prefix`. Keeps `app.py` small and groups endpoints by feature.
- **Split frontend:** Break `client_side.js` into modules (e.g. config, upload, preprocess, modeling, inference) and load them in order or bundle; same for `templates/index.html` (Jinja2 includes). See `docs/ARCHITECTURE_RECOMMENDATIONS.md`.

**Result:** Easier navigation and ownership; fewer merge conflicts.

---

## 5. API and error handling

**Current:** Routes return JSON with `error` keys and various status codes; validation is ad hoc.

**Align:**

- **Stable JSON contract:** Document or enforce a small response schema for main endpoints (e.g. success: `{ "ok": true, "data": ... }`, error: `{ "ok": false, "error": "..." }`) and use consistent status codes (400 validation, 404 not found, 500 server error).
- **Centralized validation:** For `/process`, `/preprocess`, `/predict`, consider a small layer (e.g. a function per route or shared validator) that returns 400 + message when required fields are missing or invalid, so route handlers stay thin.
- **Structured errors in logs:** Log exceptions with a consistent format (e.g. route name, traceback, request id) so production debugging is easier.

**Result:** Frontend and API consumers get predictable responses; support and debugging are simpler.

---

## 6. Logging and observability

**Current:** Some `logger.info` / `logger.debug`; no standard format or levels.

**Align:**

- **Single logging config** at app startup (e.g. in `app.py` or `config.py`): level (INFO in prod, DEBUG in dev), format (e.g. `%(asctime)s [%(levelname)s] %(name)s: %(message)s`), and optional file handler for a log file under `LOG_DIR`.
- **Use levels consistently:** DEBUG for noisy detail, INFO for key actions (e.g. “model X training started”), WARNING for recoverable issues, ERROR for failures.
- **Optional:** Request id (e.g. from middleware) in log lines so one request can be traced across logs.

**Result:** Production issues are easier to diagnose; log volume is controllable.

---

## 7. Type hints and static checks

**Current:** Some type hints; no mypy or strict typing.

**Align:**

- **Add type hints** to new or touched code (especially `model_registry.py`, config, and route handlers). Prefer at least function signatures and return types.
- **Optional:** Run `mypy` on `python_scripts/` and `app.py` (start with a loose config; tighten over time). Add mypy to `requirements-dev.txt` and CI.

**Result:** Fewer type-related bugs; better IDE support and refactoring safety.

---

## 8. Production and security (already partly documented)

**From docs/HANDOFF.md / existing practice:**

- Use **gunicorn** with **1 worker** (in-memory state).
- **CSRF:** Enable CSRF protection for forms if the app uses session-based auth or state-changing GETs.
- **HTTPS / reverse proxy:** Put the app behind HTTPS and a reverse proxy (e.g. nginx) in production.
- **Secrets:** No secrets in repo; use env vars or a secret manager for any API keys or credentials.

**Optional:** Add a short “Production checklist” (env vars, worker count, HTTPS, logging path) in `README.md` or `deploy/README.md`.

---

## 9. Documentation for contributors

**Current:** docs/HANDOFF.md, docs/ARCHITECTURE_RECOMMENDATIONS.md, README, and various docs.

**Align:**

- **README.md:** Keep a clear “Quick start” (install, run web, run desktop), link to docs/HANDOFF.md and docs/ARCHITECTURE_RECOMMENDATIONS.md for contributors.
- **CONTRIBUTING.md (optional):** How to run tests, how to add a model (registry + kwargs), where routes and frontend live. Keeps onboarding consistent.

**Result:** New contributors know how to run, test, and extend the app.

---

## Suggested order

1. **Quick wins:** Pin dependency versions, add a minimal pytest suite (a few unit tests + keep/run `tests/integration/test_flow.py`), add one GitHub Actions workflow that runs tests.
2. **Structure:** Blueprints and/or split JS/HTML as in ARCHITECTURE_RECOMMENDATIONS.
3. **Robustness:** Centralize validation and error response shape; improve logging config.
4. **Polish:** Type hints + mypy in CI; CONTRIBUTING.md and a short production checklist.

This keeps the codebase in line with common software practice without a full rewrite.
