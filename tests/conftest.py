"""Pytest fixtures and config. Ensures project root is on path when running tests."""
import os
import sys
import tempfile
from pathlib import Path

# Use a temp app-support dir for tests so uploads and logs don't touch real app data.
# Must be set before any import of app or python_scripts.config.
_test_support = Path(tempfile.gettempdir()) / "digiterra_test_support"
_test_support.mkdir(parents=True, exist_ok=True)
os.environ["DIGITERRA_APP_SUPPORT_DIR"] = str(_test_support)
os.environ["DIGITERRA_OUTPUT_DIR"] = str(_test_support / "user_visualizations")

# Non-interactive backend for matplotlib (avoids display/GTK errors in CI)
os.environ.setdefault("MPLBACKEND", "Agg")

# Run from project root so python_scripts and app are importable
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
os.chdir(_root)
