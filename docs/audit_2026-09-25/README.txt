Audit documents of 2026-09-25: LaTeX sources, figures and the scripts that made them.

joss_supplement.tex   supplementary material for the JOSS submission
netl_record.tex       internal record for the EDX deployment (adds deployment context and operating notes)
web_page.tex          the same sections for the repository web page (pandoc to HTML)
common/               shared preamble and sections
figures/              fig_*.pdf from make_figures.py; screens/ are first-hand screenshots of 2026-09-25
make_figures.py       python3 make_figures.py <repo_root>  (reads RESULTS.csv, the sweep and round-trip JSON)
shoot_local.py        Playwright screenshots of a locally running DiGiTerra and H2O Flow

Compile: xelatex joss_supplement.tex (twice). Latin Modern through fontspec.
The compiled PDF, Word and HTML files and the citation audit workbook are in
DiGiTerra/audit_docs_2026-09-25/ beside the repository, not in git.
