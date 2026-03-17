# Repo polish checklist

Checklist for tightening repo hygiene and presentation. Done items are listed first.

---

## Done in this pass

- **Git hygiene**  
  `node_modules/` is in `.gitignore`. Commit `package-lock.json` for reproducible frontend installs:  
  `git add package-lock.json && git commit -m "chore: track package-lock.json for reproducible npm installs"`

- **Python version alignment**  
  `pyproject.toml` now requires `>=3.11` to match `docs/DEPENDENCIES.md`.

- **Hermetic integration tests**  
  `tests/conftest.py` sets `DIGITERRA_APP_SUPPORT_DIR` and `DIGITERRA_OUTPUT_DIR` to a temp dir before any app/config import, so upload tests no longer touch `~/Library/Application Support/DiGiTerra` (or equivalent).

- **Lint scope and cap**  
  `package.json` lint script now runs ESLint on the files the app actually loads (`static/js/app.js`, `static/js/core.js`) plus legacy `static/client_side.js`, with `--max-warnings 35` so the current 35 warnings don’t block CI but new ones do.

---

## Recommended next steps

### 1. Commit lockfile and clean status

- Commit `package-lock.json` (see above).  
- Run `git status` and ensure no unintended untracked files; add any other intentional ignores to `.gitignore` if needed.

### 2. Tighten lint over time

- **Done.** `--max-warnings 0`. All 86+ warnings resolved: unused vars prefixed with `_` or annotated (HTML-called functions), `.eslintrc.cjs` and `.eslintrc.json` use `varsIgnorePattern: "^_"`. MLP `hidden_layer_size1` typo fixed in `app.js` and `client_side.js`. CI runs `npm run lint` with zero warnings.

### 3. Frontend migration and consistency

- **Documented.** CONTRIBUTING “Where things live” states canonical sources (`static/js/app.js`, `static/js/core.js`), legacy (`static/client_side.js`), and that new changes go in canonical files; lint covers all three.  
- You can complete the migration (remove or replace `client_side.js` and update references) when ready.

### 4. Shrink large UI assets (medium effort)

- **Phase 1 done.** `static/js/app.js` split by tab/feature: `upload.js` (upload + correlation), `preprocess.js`, `modeling.js`, `inference.js`; thin `app.js` (welcome, nav, popups). Load order in `templates/index.html`: core.js, upload, preprocess, modeling, inference, app. Shared helpers (`formatDateTimeForFilename`, `downloadFile`, `showCrossValidationUnavailable`, `downloadAdditionalInfoTable`) moved to `core.js`. No bundler.
- **Phase 2 done.** Inline scripts moved out of `templates/index.html`: light-mode init and `goToModelPreprocessing` fallback live in `static/js/init.js`; config (API_ROOT, STATIC_ROOT) remains inline (Jinja). Init loaded in `<head>`.
- **Bundle.** Run `npm run build` to produce `static/js/app.bundle.js`. Set env `DIGITERRA_USE_JS_BUNDLE=1` in production to load the single bundle instead of the seven scripts. See `scripts/build-js-bundle.cjs` and `docs/BUILD_INSTRUCTIONS.md`.

### 5. CI and docs

- CI runs `pytest tests/` (unit + integration) and `npm run lint`; conftest enforces a temp dir for tests.  
- CONTRIBUTING.md states that tests are hermetic and do not write to the real app support dir.

---

## Summary

| Area              | Before                         | After (this pass)                    |
|-------------------|--------------------------------|-------------------------------------|
| Git hygiene       | Untracked node_modules/lock    | node_modules ignored; lockfile ready to commit |
| Python version    | 3.10 vs 3.11+ mismatch         | Aligned to 3.11+                     |
| Integration tests | Wrote to app-support dir       | Hermetic (temp dir)                  |
| Lint              | Legacy file only, 999 warnings | 0 warnings (hard gate)              |

Working through the recommended next steps (lockfile commit, lint to 0, frontend consistency, splitting large files) improves presentation and engineering hygiene.
