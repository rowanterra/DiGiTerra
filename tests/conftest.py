"""Pytest fixtures and config. Ensures project root is on path when running tests."""
import os
import sys

# Non-interactive backend for matplotlib (avoids display/GTK errors in CI)
os.environ.setdefault("MPLBACKEND", "Agg")

# Run from project root so python_scripts and app are importable
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
os.chdir(_root)
