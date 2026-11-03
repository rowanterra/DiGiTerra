DiGiTerra 0.1.0 guide of 2026-09-25: sources for docs/digiterra_guide_2026-09-25.html.

build_guide.py        writes ../digiterra_guide_2026-09-25.html (run from the repository root)
drive_guided.py       Playwright run of the guided page on examples/penguins.csv (random forest, predictions on penguins_new.csv)
drive_guided_knn.py   the same in Full mode with KNeighbors and n_neighbors 3
make_cite_workbook.py citation workbook for the guide, sentences read from the built HTML
build_webpage.py      restyles audit_docs_2026-09-25/DiGiTerra_Web_Page_2026-09-25.html to the guide's tokens (writes the ...b.html copy)
cayman.css            stylesheet copied from Terrahedral/docs/terrahedral_guide_2026-09-24.html
shots/                screenshots of 2026-09-25 at commit d705ff9, Linux x86_64, Python 3.11.15, scikit-learn 1.9.0
fig/                  docs/audit_2026-09-25/figures/fig_e1_agreement.pdf and fig_sweep_roundtrip.pdf at 150 dpi

The guide is committed at docs/digiterra_guide_2026-09-25.html. The restyled web page
(DiGiTerra_Web_Page_2026-09-25b.html) and the guide's citation workbook are outside git,
beside the repository in DiGiTerra/audit_docs_2026-09-25/.
