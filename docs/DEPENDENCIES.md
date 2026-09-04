# Dependencies and installs

## Python

Python 3.11 or 3.12. CI runs the tests on both.

## Install

From the project root:

```bash
pip install -r requirements.txt
```

For development and tests:

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

## Pins

`requirements.txt` pins every package with `==` to the versions recorded by `pip freeze` on 2026-09-03 under Python 3.11: Flask 3.1.3, gunicorn 23.0.0, matplotlib 3.11.1, NumPy 1.26.4, openpyxl 3.1.5, pandas 2.3.3, scikit-learn 1.9.0, SciPy 1.17.1, scikit-optimize 0.10.2, seaborn 0.13.2, SHAP 0.49.1, Werkzeug 3.1.8, XlsxWriter 3.2.9, pywebview 5.4, and pyarrow 25.0.1 (added 2026-09-04 for Parquet). `requirements-dev.txt` adds pytest 8.4.2 and ruff 0.16.6.

NumPy is pinned to 1.26.4. The test suite has been run against the pinned set only, so NumPy 2 is untested with this code. An upgrade of any pin is a change to test with `pytest tests/` and one training run per problem type before committing.

## Install check

```bash
python scripts/check_requirements.py
```

The script imports each package in `requirements.txt` and lists the ones that fail. The script does not scan the source for imports.

## Builds

The desktop build (PyInstaller) uses the environment where `pip install -r requirements.txt` ran. `docs/BUILD_INSTRUCTIONS.md` has the platform steps.
