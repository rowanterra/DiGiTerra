"""Write the citation workbook for digiterra_guide_2026-09-25.html.

The document sentence column is read from the built HTML, not typed, so it matches the page.
Run: python3 make_cite_workbook.py GUIDE.html OUT.xlsx
"""
import html, re, sys
import openpyxl
from openpyxl.styles import Alignment, Font

guide, out = sys.argv[1], sys.argv[2]
s = open(guide).read()
s = re.sub(r"<style>.*?</style>|<script>.*?</script>", "", s, flags=re.S)
s = re.sub(r'src="data:[^"]+"', "", s)
s = re.sub(r"</(p|h\d|li|td|th|figcaption|summary|pre|div)>", "\n", s)
text = html.unescape(re.sub(r"<[^>]+>", "", s))
sentences = []
for line in text.split("\n"):
    line = re.sub(r"\s+", " ", line).strip()
    if line:
        sentences += re.split(r"(?<=[.])\s+(?=[A-Za-z])", line)

REFS = {
 "Pedregosa": ("Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
   "Peer-reviewed journal article", "Yes", "Scikit-learn is a Python module integrating a wide range of state-of-the-art machine learning algorithms for medium-scale supervised and unsupervised problems.", "p. 2825, Abstract (Pedregosa_etal_2011_JMachLearnRes.pdf, read for the 2026-09-09 audit)", "Verified", ""),
 "Gorman": ("Gorman, K. B., Williams, T. D., & Fraser, W. R. (2014). Ecological sexual dimorphism and environmental variability within a community of Antarctic penguins (genus Pygoscelis). PLoS ONE, 9(3), e90081. https://doi.org/10.1371/journal.pone.0090081",
   "Peer-reviewed journal article", "Yes", "We examined ecological sexual dimorphism among sympatric nesting Pygoscelis penguins asking whether environmental variability is associated with differences in male and female pre-breeding foraging niche.", "p. 1 (e90081), Abstract (Gorman_etal_2014_PLoSONE.pdf)", "Verified", "Source of examples/penguins.csv (344 rows); counts in the sentence are from examples/README.md and the upload status line."),
 "skops developers": ("skops developers. (n.d.). skops (Version 0.14.0) [Computer software]. https://github.com/skops-dev/skops",
   "Open-source software repository", "Yes", "n/a (software). skops 0.14.0 released on PyPI 2026-04-20.", "pypi.org/project/skops/0.14.0/, read 2026-09-09; version also in manifest.artifact.serializer of the 2026-09-25 penguin run", "Verified", ""),
 "Liu et al., 2008": ("Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest. In Proceedings of the 2008 Eighth IEEE International Conference on Data Mining (pp. 413–422). IEEE. https://doi.org/10.1109/ICDM.2008.17",
   "Peer-reviewed conference paper", "Yes (author copy)", "This paper proposes a fundamentally different model-based method that explicitly isolates anomalies instead of profiles normal points.", "Abstract, author copy at https://www.lamda.nju.edu.cn/publication/icdm08b.pdf, read 2026-09-25", "Verified (quote); pages from the IEEE/ACM listing", "The author copy carries no page numbers; pp. 413–422 and the DOI are from the IEEE Xplore and ACM DL listings returned by search on 2026-09-25. The detector in the manifest is IsolationForest (applicability.detector)."),
 "Lei et al., 2018": ("Lei, J., G’Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-free predictive inference for regression. Journal of the American Statistical Association, 113(523), 1094–1111. https://doi.org/10.1080/01621459.2017.1307116",
   "Peer-reviewed journal article", "Yes (arXiv preprint 1604.04173)", "guaranteeing finite-sample marginal coverage even when these assumptions do not hold.", "Abstract, arXiv:1604.04173v2, read 2026-09-25", "Verified (quote from preprint)", "Quote read from the arXiv version, not the journal version. Volume and issue from the Taylor & Francis listing; pages 1094–1111 from a citation index. Confirm pages on the publisher page before print."),
 "World Wide Web Consortium": ("World Wide Web Consortium. (2024). Web Content Accessibility Guidelines (WCAG) 2.2 (W3C Recommendation, 12 December 2024). https://www.w3.org/TR/WCAG22/",
   "Standard (W3C Recommendation)", "Yes", "Web Content Accessibility Guidelines (WCAG) 2.2 covers a wide range of recommendations for making web content more accessible.", "Abstract, https://www.w3.org/TR/WCAG22/, read 2026-09-25", "Verified", "The page states the target as built to WCAG 2.2 AA; conformance is not claimed (docs/ACCESSIBILITY.md)."),
 "Efron et al., 2004": ("Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). Least angle regression. The Annals of Statistics, 32(2), 407–499. https://doi.org/10.1214/009053604000000067",
   "Peer-reviewed journal article", "No", "", "DOI resolves to Project Euclid, Annals of Statistics vol. 32 issue 2, 'Least angle regression' (checked 2026-09-25)", "Metadata verified; no quote", "Attribution of the diabetes dataset distributed with scikit-learn; the README carries the same reference."),
 "Fisher, 1936": ("Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. Annals of Eugenics, 7(2), 179–188. https://doi.org/10.1111/j.1469-1809.1936.tb02137.x",
   "Peer-reviewed journal article", "Yes (publisher PDF)", "Table I shows measurements of the flowers of fifty plants each of the two species Iris setosa and I. versicolor", "p. 179, Arithmetical procedure (Fisher_1936_AnnEugen.pdf)", "Verified (near)", "Quote check 'near' in the 2026-09-09 audit: the scan renders 'versicolor' imperfectly."),
 "Pallets": ("Pallets. (2026). Flask (Version 3.1.3) [Computer software]. https://flask.palletsprojects.com",
   "Software", "Yes", "n/a (software). Flask 3.1.3 released on PyPI 2026-02-19.", "pypi.org release metadata, read 2026-09-09; flask==3.1.3 in requirements.txt", "Verified", ""),
}
MARKERS = [("Pedregosa et al., 2011", "Pedregosa"), ("Gorman et al. (2014)", "Gorman"),
           ("skops developers, n.d.", "skops developers"), ("Liu et al., 2008", "Liu et al., 2008"), ("Lei et al., 2018", "Lei et al., 2018"),
           ("World Wide Web Consortium, 2024", "World Wide Web Consortium"), ("Efron et al., 2004", "Efron et al., 2004"),
           ("Fisher, 1936", "Fisher, 1936"), ("Pallets, 2026", "Pallets")]

wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Citation audit"
head = ["#", "In-text marker", "Exact document sentence (read from the built HTML)", "Full APA reference", "Source type",
        "Open access", "Verbatim supporting quote", "Source location", "Verification status", "Notes"]
ws.append(head)
n = 0
for marker, key in MARKERS:
    hits = [x for x in sentences if marker in x]
    assert hits, marker
    for sent in hits:
        n += 1
        ref = REFS[key]
        ws.append([n, f"({marker})" if "(" not in marker else marker, sent.strip(), ref[0], ref[1], ref[2], ref[3], ref[4], ref[5], ref[6]])
for c in ws[1]:
    c.font = Font(bold=True)
widths = [4, 24, 70, 60, 20, 14, 60, 40, 22, 50]
for i, w in enumerate(widths):
    ws.column_dimensions[chr(65 + i)].width = w
for row in ws.iter_rows(min_row=2):
    for c in row:
        c.alignment = Alignment(wrap_text=True, vertical="top")
ws.freeze_panes = "A2"

rm = wb.create_sheet("Read me")
for r in [("Document", "docs/digiterra_guide_2026-09-25.html (DiGiTerra 0.1.0 guide)"),
          ("Built by", "build_guide.py; this workbook by make_cite_workbook.py from the built HTML"),
          ("Rows", n), ("Date", "2026-09-25"),
          ("Reused rows", "Pedregosa, Gorman, skops, Pallets and Fisher quotes and locations from DiGiTerra_Citation_Audit_2026-09-09_v3.xlsx"),
          ("New rows", "Liu et al. 2008, Lei et al. 2018, W3C 2024 read on 2026-09-25; Efron et al. 2004 metadata only")]:
    rm.append(list(r))
rm["A1"].font = Font(bold=True); rm.column_dimensions["A"].width = 16; rm.column_dimensions["B"].width = 100
wb.save(out); print(out, n, "rows")
