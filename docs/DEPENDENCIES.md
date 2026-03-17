# Dependencies and installs

## Python

Use Python 3.11 or higher. The project uses type hints and dependencies that assume 3.11+.

## Install

From the project root:

```bash
pip install -r requirements.txt
```

For development and tests:

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

## Key constraints

- **NumPy**: `numpy>=1.24,<3`. NumPy 2 is supported.
- **SHAP**: `shap>=0.46,<0.51`. SHAP 0.46+ supports NumPy 2; earlier versions do not.
- **scikit-learn**: `>=1.3,<2`. Do not upgrade to 2.x without testing; API changes are possible.
- **pandas**: `>=2.0,<3`.

Other packages in `requirements.txt` are pinned to major (or minor) ranges for reproducible installs and to avoid known incompatibilities.

## Verify install

After installing, you can check that all imports used by the app are satisfied:

```bash
python scripts/check_requirements.py
```

This script reports any missing or problematic packages. Fix any reported issues before running the app or tests.

## Builds

When building the desktop app (PyInstaller), use the same environment where `pip install -r requirements.txt` was run. See `docs/BUILD_INSTRUCTIONS.md` for platform-specific steps.
