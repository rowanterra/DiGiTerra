"""The documentation must name every environment variable the code reads.

A limit that exists only in code is a limit an operator cannot find. This walks
the source for DIGITERRA_* names and fails when one is missing from both
docs/documentation.md and docs/SECURITY_REVIEW.md, which is how the published
security posture drifted from the code once already.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = ("core", "routes", "python_scripts", "scripts")
DOCS = ("docs/documentation.md", "docs/SECURITY_REVIEW.md", "README.md", "CHANGELOG.md")
# Set by the test and deployment tooling rather than read as a limit.
EXEMPT = {"DIGITERRA_APP_SUPPORT_DIR", "DIGITERRA_OUTPUT_DIR", "DIGITERRA_BASE_DIR", "DIGITERRA_PORT"}


def _names_in_code() -> set[str]:
    names = set()
    for d in SOURCE_DIRS:
        for path in (ROOT / d).rglob("*.py"):
            names |= set(re.findall(r"DIGITERRA_[A-Z0-9_]+", path.read_text(encoding="utf-8")))
    return names - EXEMPT


def _names_in_docs() -> set[str]:
    text = "\n".join((ROOT / d).read_text(encoding="utf-8") for d in DOCS)
    return set(re.findall(r"DIGITERRA_[A-Z0-9_]+", text))


def test_every_environment_variable_is_documented():
    undocumented = sorted(_names_in_code() - _names_in_docs())
    assert not undocumented, f"Add these to docs/documentation.md: {', '.join(undocumented)}"


def test_the_security_review_matches_the_desktop_limits():
    """The published posture and desktop_app.py stated different limits once."""
    desktop = (ROOT / "desktop_app.py").read_text(encoding="utf-8")
    review = (ROOT / "docs" / "SECURITY_REVIEW.md").read_text(encoding="utf-8")
    for var, value in re.findall(r'setdefault\("(DIGITERRA_MAX_[A-Z_]+)",\s*"(\d+)"\)', desktop):
        if var.endswith("_TOTAL_FRAME_MB"):
            continue  # documented in words as "no total across sessions"
        assert value in review, f"{var}={value} in desktop_app.py is not stated in SECURITY_REVIEW.md"
    assert "no longer disables" in review
