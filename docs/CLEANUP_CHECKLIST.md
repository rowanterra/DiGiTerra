# Repo polish checklist


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

- Current: `--max-warnings 88` (baseline after adding app-loaded JS to lint). ESLint config: `.eslintrc.json`; CI runs `npm run lint`.  
- Goal: fix warnings and set `--max-warnings 0` so lint is a hard gate.

### 3. Frontend migration and consistency

- **Canonical sources:** The app loads `templates/index.html` → `static/js/app.js`. `static/client_side.js` is documented as legacy; canonical sources are `static/js/core.js` and `static/js/app.js`.  
- Either complete the migration (remove or replace `client_side.js` and update all references) or clearly document “legacy vs canonical” and keep lint covering both until migration is done.  
- Ensure `package.json` scripts (lint, and any future build/test) match the real entrypoints.

### 4. Shrink large UI assets (medium effort)

- `static/client_side.js`, `static/js/app.js`, and `templates/index.html` are very large, which hurts maintainability.  
- Consider splitting by feature or route (e.g. upload, exploration, preprocessing, modeling, inference), introducing a small build step if needed, and/or moving inline scripts out of `index.html` into modules.  
- Improves “healthy repo” signal and makes the UI layer easier to work on.

### 5. CI and docs

-  CI runs `pytest tests/` (unit + integration) and `npm run lint`; conftest enforces a temp dir for tests.  
-  CONTRIBUTING.md states that tests are hermetic and do not write to the real app support dir.

---

## Summary

| Area              | Before                         | After (this pass)                    |
|-------------------|--------------------------------|-------------------------------------|
| Git hygiene       | Untracked node_modules/lock    | node_modules ignored; lockfile ready to commit |
| Python version    | 3.10 vs 3.11+ mismatch         | Aligned to 3.11+                     |
| Integration tests | Wrote to app-support dir       | Hermetic (temp dir)                  |
| Lint              | Legacy file only, 999 warnings | App JS + legacy, 35 max warnings     |

Tackling “Recommended next steps” (lockfile commit, lint → 0, frontend consistency, splitting large files) will move the repo toward a 9/10 on presentation and engineering hygiene.
