# Notebook

Newest entry last. One entry for each task on each day (CMP-004, Table 2).
Earlier history: `CHANGELOG.md`, `docs/COMMIT_REVIEW_2026-09-04.md`, `docs/differential_test_E1/NOTEBOOK.txt`, and the dated records in `docs/`.

## 2026-09-29  Layout brought to CMP-001

- Objective: the same top-level shape as Terrahedral and PHREEQ-N-Web.
- Inputs: commit 8e9f171 (branch audit-fixes).
- Command: `git mv AUDIT_RESUME.md docs/reviews/`; README layout table; `scripts/check_language.py` PROSE path updated.
- Result: no code path changed. The dated records in `docs/` stay because `README.md`, the tests, `python_scripts/reporting/bundle.py` and `pyproject.toml` name them by path. Uncommitted edits to `README.md` and `AUDIT_RESUME.md` were set aside and put back on the working tree after the commit.
- Open: `requirements-lock.txt` is not written. `requirements.txt` is the pinned `pip freeze` of 2026-09-04 under Python 3.11 (docs/DEPENDENCIES.md); the `.venv/` in the folder is Python 3.13 from the 2026-09-03 audit, so a lock from it would not match the supported versions. Write the lock from the 3.11 environment (CMP-003 step 11.5).
