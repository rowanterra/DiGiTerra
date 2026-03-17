#!/usr/bin/env python3
"""
Verify that runtime dependencies from requirements.txt are installable and
importable. Run from project root: python scripts/check_requirements.py
"""
import re
import subprocess
import sys
from pathlib import Path

# Map requirement names (may have extras []) to import names
REQUIREMENT_TO_IMPORT = {
    "flask": "flask",
    "gunicorn": "gunicorn",
    "requests": "requests",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "openpyxl": "openpyxl",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "scikit-optimize": "skopt",
    "seaborn": "seaborn",
    "shap": "shap",
    "werkzeug": "werkzeug",
    "xlsxwriter": "xlsxwriter",
    "pywebview": "webview",
}


def main():
    root = Path(__file__).resolve().parent.parent
    req_file = root / "requirements.txt"
    if not req_file.exists():
        print("requirements.txt not found (run from project root)", file=sys.stderr)
        sys.exit(1)

    # Parse top-level package names from requirements.txt (skip comments, -r, -e)
    names = set()
    for line in req_file.read_text().splitlines():
        line = line.strip().split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[=<>\[\]]", line)[0].strip().lower()
        if name:
            names.add(name)

    failed = []
    for name in sorted(names):
        imp = REQUIREMENT_TO_IMPORT.get(name, name.replace("-", "_"))
        try:
            __import__(imp)
        except ImportError as e:
            failed.append((name, str(e)))

    if failed:
        print("Missing or broken packages:", file=sys.stderr)
        for name, err in failed:
            print(f"  {name}: {err}", file=sys.stderr)
        sys.exit(1)

    # Optional: run pip check for dependency conflicts (advisory only)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        cwd=str(root),
    )
    if result.returncode != 0 and (result.stdout or result.stderr):
        print("Note: pip check reported issues (may be transitive deps):", file=sys.stderr)
        print((result.stdout or result.stderr or "").strip(), file=sys.stderr)

    print("OK: requirements.txt packages are importable.")


if __name__ == "__main__":
    main()
