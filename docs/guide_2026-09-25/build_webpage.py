"""Restyle DiGiTerra_Web_Page_2026-09-25.html to the guide's tokens.

Input: the pandoc build of docs/audit_2026-09-25/web_page.tex (images embedded),
the guide's stylesheet (read from digiterra_guide_2026-09-25.html), static/doc_banner.jpg.
The body is kept as pandoc wrote it except: headings move down one level under the
masthead, the table of contents goes into the contents box, the header row that
longtable repeats as the first body row is removed, and the TL;DR is replaced with
the text of web_page.tex after its 2026-09-25 revision.
Run: python3 build_webpage.py SRC.html OUT.html
"""
import re, sys
src, out = sys.argv[1], sys.argv[2]
s = open(src).read()
guide = open("docs/digiterra_guide_2026-09-25.html").read()
css = re.search(r"<style>(.*?)</style>", guide, re.S).group(1)
body = s[s.find("<body>") + 6 : s.rfind("</body>")]

# drop pandoc's title block; the masthead replaces it
body = re.sub(r'<header id="title-block-header">.*?</header>', "", body, flags=re.S)
# table of contents into the contents box
body = re.sub(r'<nav id="TOC" role="doc-toc">(.*?)</nav>',
              r'<div class="contents"><strong>Contents</strong>\1</div>', body, flags=re.S)
# TL;DR heading and paragraph
tldr_new = ("<p><strong>TL;DR.</strong> On the regression, nearest-neighbor and descriptive-statistics "
            "cases run through every program, DiGiTerra 0.1.0 agreed with a plain scikit-learn script to the "
            "precision each program prints, and R, Excel, Orange and Altair AI Studio agreed with the same script "
            "within a relative difference of 1e-6. A review that set out to break the application reproduced 25 defects, and each "
            "fix has a regression test. Among eleven other no-code tools, DiGiTerra is the one that writes a run "
            "record with the input hash and package versions and signs its saved models. DiGiTerra does not join "
            "tables, filter rows by formula or read databases, and the node-graph tools do. Every desktop tool "
            "compared is free, and the cloud tools bill by the hour.</p>")
body, n = re.subn(r'<h4 id="tldr\.">TL;DR\.</h4>\s*<p>.*?</p>', "", body, count=1, flags=re.S)
assert n == 1
body = tldr_new + "\n" + body
# headings down one level (h4 first so nothing is moved twice)
for a, b in [("h4", "h5"), ("h3", "h4"), ("h2", "h3"), ("h1", "h2")]:
    body = re.sub(rf"<{a}(\s|>)", rf"<{b}\1", body); body = body.replace(f"</{a}>", f"</{b}>")
# tables take the guide's plain style; remove a body row that repeats the header
def fix_table(m):
    t = m.group(0).replace("<table>", '<table class="plain">', 1)
    hdr = re.search(r"<thead>.*?</thead>", t, re.S)
    if hdr:
        cells = re.findall(r"<th(?:\s[^>]*)?>(.*?)</th>", hdr.group(0), re.S)
        first = re.search(r"<tbody>\s*(<tr[^>]*>.*?</tr>)", t, re.S)
        if first and [c.strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", first.group(1), re.S)] == [c.strip() for c in cells]:
            t = t.replace(first.group(1), "", 1)
    return t
body = re.sub(r"<table>.*?</table>", fix_table, body, flags=re.S)
body = re.sub(r' style="text-align: left;"', "", body)

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DiGiTerra 0.1.0 Verification</title>
<style>{css}
.main-content table.plain td {{ vertical-align: top; }}
.main-content table.plain caption {{ caption-side: top; text-align: left; font-size: 0.9rem; color: #6c7069; margin-bottom: 0.5rem; }}
.main-content figure img {{ background: #fff; }}
.main-content td code {{ overflow-wrap: normal; }}
.main-content a.uri {{ overflow-wrap: anywhere; }}
</style>
</head>
<body>
<section class="page-header">
<h1 class="title toc-ignore project-name">DiGiTerra 0.1.0: Verification Record and Comparison with Other Tools</h1>
<h4 class="author project-author">DiGiTerra project, 2026-09-25</h4>
</section>
<section class="main-content">
{body}
</section>
</body>
</html>
"""
open(out, "w").write(page)
print(out, len(page))
