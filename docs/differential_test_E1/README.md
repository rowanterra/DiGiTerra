# Differential test E1, 2026-09-24

A black-box comparison of DiGiTerra against Orange 3.39.0, RapidMiner AI Studio 2026.0.1 and R 4.5.1 on fixed train and test halves of two example files, with a plain scikit-learn 1.9.0 script as the arbiter. `PLAN.txt` states the design, `PROTOCOL.txt` the click-level steps, `NOTEBOOK.txt` the dated record of every run and correction, and `RESULT_2026-09-24_partial.txt` the findings. `python3 scripts/compare.py` rebuilds `RESULTS.csv` and `RESULT_<date>.txt` from the saved outputs.

`digiterra_outputs/` holds the run records after the fixes in commit fb5453d; `digiterra_outputs_pre_fix_0942e0c/` holds the records that found the three defects. Fitted models, model-card HTML and figure PDFs are left out of this copy; the manifests, results, predictions and metrics are complete. The Excel cells of the plan were not run.
