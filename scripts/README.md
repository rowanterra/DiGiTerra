# Scripts

Maintainer scripts that the application does not import.

- `check_requirements.py` imports each package listed in `requirements.txt` and reports the ones that fail to import. Run with `python scripts/check_requirements.py` from the project root.
- `build-js-bundle.cjs` concatenates the files under `static/js/` into `static/js/app.bundle.js`. Run with `npm run build`.

Application and pipeline code is under `core/`, `routes/`, and `python_scripts/`.
