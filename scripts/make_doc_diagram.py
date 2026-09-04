"""Emit the run-flow diagram as an inline SVG for the Documentation view.

Same layout as paper/build_figure1.js, redrawn for the screen: classes instead of
baked-in fills, so the app stylesheet themes it in light and dark, and the
interface's font stack instead of Times.

Writes two files:
  doc_flow_diagram.svg   the fragment inlined into templates/index.html
  doc_flow_diagram.css   the rules appended to static/style.css

The diagram is inlined rather than referenced with <img> so the stylesheet reaches
it and it follows the light and dark palettes. Regenerate it here rather than
editing the SVG in the template by hand, or the two drift apart.

Run from docs/:  python ../scripts/make_doc_diagram.py
"""

from xml.sax.saxutils import escape

SCALE = 100  # px per inch, matching the paper figure's coordinate system
W, H = 1080, 1150

LEFT_CX, APP_CX = 300, 670
LANE_SPLIT = 505
BOX_W, BOX_H = 290, 46
DIA_W, DIA_H = 290, 62
TERM_X, TERM_W = 855, 210
BAND_X, BAND_W = 10, 1060
LABEL_W = 82
R0, PITCH = 85, 72

# The label strip is 82px wide and SVG text does not wrap, so each band name
# carries its own line breaks.
BANDS = [
    (["Upload"], 55, 337, "upload"),
    (["Data", "Exploration"], 337, 409, "explore"),
    (["Model", "Preprocessing"], 409, 553, "prep"),
    (["Modeling"], 553, 841, "model"),
    (["Inference on", "New Data"], 841, 1125, "infer"),
]

out = []
add = out.append


def row(n):
    return R0 + (n - 1) * PITCH


def box(cx, cy, text, w=BOX_W, cls="dg-data", size=13, stadium=False):
    x, y = cx - w / 2, cy - BOX_H / 2
    r = BOX_H / 2 if stadium else 2
    add(f'  <rect class="{cls}" x="{x:g}" y="{y:g}" width="{w:g}" height="{BOX_H}" rx="{r:g}"/>')
    add(f'  <text class="dg-label" x="{cx:g}" y="{cy:g}" font-size="{size}">{escape(text)}</text>')


def diamond(cx, cy, text):
    pts = f"{cx},{cy - DIA_H / 2:g} {cx + DIA_W / 2:g},{cy:g} {cx},{cy + DIA_H / 2:g} {cx - DIA_W / 2:g},{cy:g}"
    add(f'  <polygon class="dg-decision" points="{pts}"/>')
    add(f'  <text class="dg-label" x="{cx:g}" y="{cy:g}" font-size="12">{escape(text)}</text>')


def vline(x, y1, y2, arrow=True, cls="dg-edge"):
    ref = "dgArrowStop" if "--stop" in cls else "dgArrow"
    marker = f' marker-end="url(#{ref})"' if arrow else ""
    add(f'  <line class="{cls}" x1="{x:g}" y1="{y1:g}" x2="{x:g}" y2="{y2:g}"{marker}/>')


def hline(y, x1, x2, arrow=True, cls="dg-edge"):
    ref = "dgArrowStop" if "--stop" in cls else "dgArrow"
    marker = f' marker-end="url(#{ref})"' if arrow else ""
    add(f'  <line class="{cls}" x1="{x1:g}" y1="{y:g}" x2="{x2:g}" y2="{y:g}"{marker}/>')


def cross(from_cx, from_bottom, to_cx, to_top):
    mid = (from_bottom + to_top) / 2
    vline(from_cx, from_bottom, mid, arrow=False)
    hline(mid, from_cx, to_cx, arrow=False)
    vline(to_cx, mid, to_top)


def tag(text, x, y, cls="dg-tag"):
    add(f'  <text class="{cls}" x="{x:g}" y="{y:g}">{escape(text)}</text>')


hb, hd = BOX_H / 2, DIA_H / 2

add(
    f'<svg class="dg-flow" viewBox="0 0 {W} {H}" role="img" '
    'aria-labelledby="dgFlowTitle dgFlowDesc" xmlns="http://www.w3.org/2000/svg">'
)
add('  <title id="dgFlowTitle">How one DiGiTerra run moves through the application</title>')
add(
    '  <desc id="dgFlowDesc">A two-lane flow chart. The left lane holds the steps a user '
    "takes and the right lane holds the steps the application takes. A shaded band behind "
    "each group of steps names the page they belong to: Upload, Data Exploration, Model "
    "Preprocessing, Modeling, and Inference on New Data. Rectangles are steps and diamonds "
    "are decisions. A blocking data check stops the run, a run with no free training slot "
    "is queued, and a prediction file whose columns do not match is refused.</desc>"
)
add("  <defs>")
for mid, head_cls in (("dgArrow", "dg-arrowhead"), ("dgArrowStop", "dg-arrowhead dg-arrowhead--stop")):
    add(
        f'    <marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse">'
    )
    add(f'      <path class="{head_cls}" d="M 0 0 L 10 5 L 0 10 z"/>')
    add("    </marker>")
add("  </defs>")

for lines, y0, y1, key in BANDS:
    add(f'  <rect class="dg-band dg-band--{key}" x="{BAND_X}" y="{y0}" width="{BAND_W}" height="{y1 - y0}"/>')
    cx = BAND_X + LABEL_W / 2 + 4
    cy = (y0 + y1) / 2 - (len(lines) - 1) * 6.5
    spans = "".join(f'<tspan x="{cx:g}" dy="{0 if i == 0 else 13}">{escape(t)}</tspan>' for i, t in enumerate(lines))
    add(f'  <text class="dg-band-label" x="{cx:g}" y="{cy:g}">{spans}</text>')

add(f'  <line class="dg-divider" x1="{LANE_SPLIT}" y1="55" x2="{LANE_SPLIT}" y2="1125"/>')
add('  <line class="dg-divider" x1="92" y1="55" x2="92" y2="1125"/>')
add(f'  <text class="dg-lane" x="{LEFT_CX}" y="36">User</text>')
add(f'  <text class="dg-lane" x="{APP_CX}" y="36">DiGiTerra</text>')

box(LEFT_CX, row(1), "Start", w=160, stadium=True)
vline(LEFT_CX, row(1) + hb, row(2) - hb)

box(LEFT_CX, row(2), "Upload a file (CSV, XLSX, Parquet)")
cross(LEFT_CX, row(2) + hb, APP_CX, row(3) - hb)

box(APP_CX, row(3), "Parse in an isolated process; run the data checks", size=12)
vline(APP_CX, row(3) + hb, row(4) - hd)

diamond(APP_CX, row(4), "Blocking finding?")
hline(row(4), APP_CX + DIA_W / 2, TERM_X, cls="dg-edge dg-edge--stop")
tag("Y", APP_CX + DIA_W / 2 + 18, row(4) - 8, cls="dg-tag dg-tag--stop")
box(TERM_X + TERM_W / 2, row(4), "Run stops; reasons returned", w=TERM_W, cls="dg-terminal", size=12)
vline(APP_CX, row(4) + hd, row(5) - hb)
tag("N", APP_CX + 12, row(4) + hd + 14)

box(APP_CX, row(5), "Compute statistics, correlations, pair plots", size=12)
cross(APP_CX, row(5) + hb, LEFT_CX, row(6) - hb)

box(LEFT_CX, row(6), "Choose predictors, target, encoding, scaling, holdout", size=11)
cross(LEFT_CX, row(6) + hb, APP_CX, row(7) - hb)

box(APP_CX, row(7), "Build the design matrix and split")
cross(APP_CX, row(7) + hb, LEFT_CX, row(8) - hb)

box(LEFT_CX, row(8), "Choose AutoML, Simple or Advanced; set the estimator", size=11)
cross(LEFT_CX, row(8) + hb, APP_CX, row(9) - hd)

diamond(APP_CX, row(9), "Training slot free?")
hline(row(9), APP_CX + DIA_W / 2, TERM_X)
tag("N", APP_CX + DIA_W / 2 + 18, row(9) - 8)
box(TERM_X + TERM_W / 2, row(9), "Queued; position returned", w=TERM_W, size=12)
vline(TERM_X + TERM_W / 2, row(9) + hb, row(10), arrow=False)
hline(row(10), TERM_X + TERM_W / 2, APP_CX + BOX_W / 2)
vline(APP_CX, row(9) + hd, row(10) - hb)
tag("Y", APP_CX - 30, row(9) + hd + 14)

box(APP_CX, row(10), "Fit the model and score against a baseline", size=12)
vline(APP_CX, row(10) + hb, row(11) - hb)

box(APP_CX, row(11), "Write the manifest, model card and signed bundle", size=12)
cross(APP_CX, row(11) + hb, LEFT_CX, row(12) - hb)

box(LEFT_CX, row(12), "Upload a second file")
cross(LEFT_CX, row(12) + hb, APP_CX, row(13) - hd)

diamond(APP_CX, row(13), "Columns match?")
hline(row(13), APP_CX + DIA_W / 2, TERM_X, cls="dg-edge dg-edge--stop")
tag("N", APP_CX + DIA_W / 2 + 18, row(13) - 8, cls="dg-tag dg-tag--stop")
box(TERM_X + TERM_W / 2, row(13), "Prediction refused", w=TERM_W, cls="dg-terminal", size=12)
vline(APP_CX, row(13) + hd, row(14) - hb)
tag("Y", APP_CX + 12, row(13) + hd + 14)

box(APP_CX, row(14), "Write predictions (CSV, XLSX)", size=12)
cross(APP_CX, row(14) + hb, LEFT_CX, row(15) - hb)

box(LEFT_CX, row(15), "End", w=160, stadium=True)
add("</svg>")

CSS = """
/* --- Documentation flow diagram -------------------------------------------
   The diagram is inlined rather than loaded as an image so these rules reach
   it, which is what lets it follow the light and dark palettes. Geometry lives
   in the SVG; only colour and type live here. */

.dg-flow {
    display: block;
    width: 100%;
    height: auto;
    max-width: 900px;
    margin: 16px auto 24px;
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}

.dg-flow text {
    fill: var(--doc-ink);
    text-anchor: middle;
    dominant-baseline: middle;
}

.dg-band { fill: var(--doc-panel); stroke: var(--doc-rule); stroke-width: 1; }
.dg-band--upload  { fill: #f1f5f0; }
.dg-band--explore { fill: #f6f2e8; }
.dg-band--prep    { fill: #edf2f2; }
.dg-band--model   { fill: #f3f0f4; }
.dg-band--infer   { fill: #f6f0ef; }

.dg-band-label { font-size: 11px; font-weight: 700; }
.dg-lane       { font-size: 15px; font-weight: 700; }
.dg-tag        { font-size: 12px; font-style: italic; }
.dg-tag--stop  { fill: var(--doc-stop, #6b2e2e); }

.dg-data,
.dg-decision { fill: var(--doc-paper); stroke: var(--doc-ink); stroke-width: 1; }
.dg-terminal { fill: var(--doc-paper); stroke: var(--doc-stop, #6b2e2e);
               stroke-width: 1; stroke-dasharray: 5 4; }

.dg-edge      { stroke: var(--doc-ink); stroke-width: 1; fill: none; }
.dg-edge--stop { stroke: var(--doc-stop, #6b2e2e); }
.dg-arrowhead       { fill: var(--doc-ink); }
.dg-arrowhead--stop { fill: var(--doc-stop, #6b2e2e); }
.dg-divider   { stroke: var(--doc-rule); stroke-width: 1; }

/* A band tint that reads on paper is invisible on a dark ground, so the bands
   become a faint lift of the panel colour and the labels carry the meaning. */
body.dark-mode .dg-band,
body.dark-mode .dg-band--upload,
body.dark-mode .dg-band--explore,
body.dark-mode .dg-band--prep,
body.dark-mode .dg-band--model,
body.dark-mode .dg-band--infer {
    fill: var(--doc-panel);
}

body.dark-mode .dg-band--explore { fill: color-mix(in srgb, var(--doc-panel) 88%, #b08a2a); }
body.dark-mode .dg-band--prep    { fill: color-mix(in srgb, var(--doc-panel) 88%, #3f8f92); }
body.dark-mode .dg-band--model   { fill: color-mix(in srgb, var(--doc-panel) 88%, #8a6ba0); }
body.dark-mode .dg-band--infer   { fill: color-mix(in srgb, var(--doc-panel) 88%, #a05a5a); }

@media (max-width: 720px) {
    .dg-flow { font-size: 15px; }
}

@media print {
    .dg-flow { max-width: 100%; }
}
"""

with open("doc_flow_diagram.svg", "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
with open("doc_flow_diagram.css", "w", encoding="utf-8") as f:
    f.write(CSS.lstrip("\n"))

print(f"wrote doc_flow_diagram.svg ({len(out)} lines) and doc_flow_diagram.css")
