"""Figures for the DiGiTerra audit documents, from the repository's own records.

Input (paths relative to the repo root passed as argv[1]):
  docs/differential_test_E1/RESULTS.csv     app, case, term, oracle, value, cls, reldiff
  docs/every_model_sweep_2026-09-24.json    one dict per registered model, status pass/skip
  docs/artifact_roundtrip_2026-09-24.json   summary dict and rows
  docs/price_survey_2026-09-25.txt          read by hand; the price scenario is coded below
Output: figures/*.pdf and *.png, white ground, black and one accent, hatched fills.
Run: python3 make_figures.py <repo_root>
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(sys.argv[1]); OUT = Path(__file__).parent / "figures"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "hatch.linewidth": 0.5, "legend.frameon": False})
ACCENT = "#4f6d4a"   # moss
def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight"); plt.close(fig)

# ---- Figure 1: E1 agreement by application and case
r = pd.read_csv(ROOT / "docs/differential_test_E1/RESULTS.csv")
apps = ["digiterra", "r", "excel", "orange", "aistudio"]
labels = {"digiterra": "DiGiTerra", "r": "R 4.5.1", "excel": "Excel 16.03", "orange": "Orange 3.39.0", "aistudio": "AI Studio 2026.0.1"}
cases = ["D1", "D2", "R1", "C1", "C2"]
classes = [("same", "white", ""), ("last digit", "white", "////"), ("rounded 6 dp", "white", "...."),
           ("near zero", "#d9d9d9", ""), ("other", ACCENT, "")]
fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.2), sharey=True)
for ax, case in zip(axes, cases):
    sub = r[r["case"] == case]
    y = np.arange(len(apps))
    left = np.zeros(len(apps))
    for cls, col, hatch in classes:
        vals = np.array([(sub[(sub.app == a) & (sub.cls == cls)]).shape[0] for a in apps], float)
        ax.barh(y, vals, left=left, color=col, edgecolor="black", linewidth=0.6, hatch=hatch, height=0.6, label=cls)
        left += vals
    ax.set_yticks(y); ax.set_yticklabels([labels[a] for a in apps]); ax.invert_yaxis()
    ax.set_title(case, fontsize=8, loc="left")
    for i, a in enumerate(apps):
        n = int(left[i]); ax.text(left[i] + 2, i, ("not run" if case != "C2" else "excluded by plan") if n == 0 else str(n), va="center", fontsize=6 if n == 0 else 6.5)
    ax.set_xlim(0, max(left.max() * 1.3, 10))
h, l = axes[0].get_legend_handles_labels()
fig.supxlabel("compared cells (C1 and C2 bars: predicted-class rows and probability cells)", fontsize=7.5, y=-0.06)
fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.2), fontsize=7)
save(fig, "fig_e1_agreement")

# ---- Figure 2: relative differences, R1 predictions and coefficients, per application
fig, ax = plt.subplots(figsize=(3.4, 2.4))
sub = r[(r["case"] == "R1") & (r.term.str.startswith("pred.") | r.term.isin(["intercept","age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]))]
data, names = [], []
for a in apps:
    d = sub[sub.app == a]["reldiff"].to_numpy(float); d = d[np.isfinite(d)]
    if len(d): data.append(np.maximum(d, 1e-18)); names.append(labels[a])
bp = ax.boxplot(data, vert=False, widths=0.5, patch_artist=True, showfliers=True,
                flierprops=dict(marker="o", markersize=2, markerfacecolor="black", markeredgecolor="none"),
                medianprops=dict(color="black"), boxprops=dict(facecolor="white", linewidth=0.6), whiskerprops=dict(linewidth=0.6), capprops=dict(linewidth=0.6))
ax.set_yticks(range(1, len(names) + 1)); ax.set_yticklabels(names); ax.set_xscale("log")
ax.axvline(1e-6, color=ACCENT, linewidth=0.8, linestyle="--"); ax.text(1e-6, len(names) + 0.55, "tolerance 1e-6", color=ACCENT, fontsize=6.5, ha="center")
ax.set_xlabel("relative difference from the scikit-learn arbiter (log)"); ax.set_xlim(1e-18, 1e-3)
save(fig, "fig_e1_r1_reldiff")

# ---- Figure 3: Orange C2 probability gap
sub = r[(r.app == "orange") & (r["case"] == "C2") & (r.cls == "other")]
fig, ax = plt.subplots(figsize=(3.4, 2.2))
ax.hist(sub["reldiff"], bins=np.logspace(-6, -1.5, 19), color="white", edgecolor="black", hatch="////", linewidth=0.6)
ax.set_xscale("log"); ax.set_xlabel("relative difference, class probability"); ax.set_ylabel("cells")
ax.axvline(1e-6, color=ACCENT, linewidth=0.8, linestyle="--")
ax.set_title(f"Orange C2 logistic regression, {len(sub)} of 180 probability cells", fontsize=8, loc="left")
save(fig, "fig_e1_orange_c2")

# ---- Figure 4: sweep and round trip
s = json.load(open(ROOT / "docs/every_model_sweep_2026-09-24.json"))
rt = json.load(open(ROOT / "docs/artifact_roundtrip_2026-09-24.json"))
tasks = ["regression", "classification", "cluster"]
cnt = {t: [sum(1 for x in s if x["task"] == t and x["status"] == "pass"), sum(1 for x in s if x["task"] == t and x["status"] == "skip")] for t in tasks}
rows = rt["rows"]; rtc = {t: [sum(1 for x in rows if x["task"] == t and x["status"] == "ok"), sum(1 for x in rows if x["task"] == t and x["status"] != "ok")] for t in tasks}
fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.0), sharey=True)
for ax, (title, c, lab) in zip(axes, [("Sweep: trains with an empty form", cnt, "registered without a trainer"), ("Round trip through the bundle", rtc, "not writable safely")]):
    y = np.arange(3)
    ok = [c[t][0] for t in tasks]; bad = [c[t][1] for t in tasks]
    ax.barh(y, ok, color="white", edgecolor="black", linewidth=0.6, height=0.6, label="pass")
    ax.barh(y, bad, left=ok, color=ACCENT, edgecolor="black", linewidth=0.6, height=0.6, label=lab)
    for i in range(3): ax.text(ok[i] + bad[i] + 0.5, i, f"{ok[i]} of {ok[i]+bad[i]}", va="center", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels(["regression", "classification", "clustering"]); ax.invert_yaxis()
    ax.set_title(title, fontsize=8, loc="left"); ax.set_xlim(0, 42); ax.set_xlabel("estimators"); ax.legend(fontsize=6.5, loc="lower right")
save(fig, "fig_sweep_roundtrip")

# ---- Figure 5: sweep training time per estimator
fig, ax = plt.subplots(figsize=(7.2, 2.4))
ss = sorted([x for x in s if x["status"] == "pass"], key=lambda x: -x["seconds"])
ax.bar(range(len(ss)), [x["seconds"] for x in ss], color="white", edgecolor="black", linewidth=0.5, hatch="////", width=0.7)
ax.set_xticks(range(len(ss))); ax.set_xticklabels([x["model"] for x in ss], rotation=90, fontsize=5.5)
ax.set_ylabel("seconds"); ax.set_yscale("symlog", linthresh=1); ax.set_title("Sweep: seconds to train each estimator on its example dataset", fontsize=8, loc="left")
save(fig, "fig_sweep_seconds")

# ---- Figure 6: price scenario, one user, one year, 100 hours of modelling
# Values are the stated prices in docs/price_survey_2026-09-25.txt; the scenario assumptions are in the caption.
items = [("DiGiTerra", 0, "MIT"), ("Orange", 0, "GPL"), ("KNIME Analytics Platform", 0, "free tier"), ("Weka", 0, "GPL"), ("JASP", 0, "GPL"), ("jamovi", 0, "free"),
         ("H2O-3 Flow", 0, "Apache 2.0"), ("AI Studio, non-commercial", 0, "free one-year license"),
         ("Excel, Microsoft 365 Personal", 99.99, "$99.99 per year"), ("Excel, Office Home 2024", 179.99, "$179.99 once"),
         ("KNIME Pro", 19 * 12, "$19 per month"), ("SageMaker Canvas, 100 session hours", 1.9 * 100, "$1.90 per hour"),
         ("Vertex AI AutoML, 10 training hours", 21.252 * 10, "$21.252 per hour")]
fig, ax = plt.subplots(figsize=(5.4, 3.0))
y = np.arange(len(items)); v = [i[1] for i in items]
ax.barh(y, v, color=["white" if x == 0 else ACCENT for x in v], edgecolor="black", linewidth=0.6, height=0.6)
for i, (n, val, note) in enumerate(items):
    ax.text(max(val, 0) + 4, i, f"${val:,.0f}  ({note})", va="center", fontsize=6.5, parse_math=False)
ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items]); ax.invert_yaxis(); ax.set_xlim(0, 330)
ax.set_xlabel("US dollars, one user, one year"); save(fig, "fig_price_scenario")
print("figures written to", OUT)
