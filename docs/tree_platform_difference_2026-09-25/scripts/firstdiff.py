"""Run: python firstdiff.py LINUX_DIR MAC_DIR (outputs of tree_probe_2026-09-25.py with prefixes linux, linux3, mac3).

First node where the Linux and Mac trees disagree, and the two splits' improvements there."""
import sys, numpy as np
from fractions import Fraction
from sklearn.ensemble._forest import _generate_sample_indices, _get_n_samples_bootstrap
import json
LDIR, mac_dir = sys.argv[1], sys.argv[2]
L = {r['model']: r for r in json.load(open(LDIR + '/linux.json'))}

def rows_at(T, X, node, keep):
    """Training rows that reach `node`, following thresholds (x <= thr goes left)."""
    parent = {}
    for p in range(len(T['feature'])):
        for c in (T['children_left'][p], T['children_right'][p]):
            if c != -1: parent[c] = p
    r = keep.copy(); chain = []
    n = node
    while n in parent:
        p = parent[n]; chain.append((p, T['children_left'][p] == n)); n = p
    for p, left in reversed(chain):
        go = X[r, T['feature'][p]] <= T['threshold'][p]
        r = r[go] if left else r[~go]
    return r

def proxy_exact(X, y, w, rows, f, thr):
    L_ = rows[X[rows, f] <= thr]; R = rows[X[rows, f] > thr]
    def part(ix):
        W = sum(Fraction(int(w[i])) for i in ix); S = sum(Fraction(int(w[i])) * Fraction(float(y[i])) for i in ix)
        return S * S / W
    return part(L_) + part(R), set(L_.tolist())

from sklearn.ensemble import BaggingRegressor
_d = np.load(LDIR + '/linux_Bagging.npz')
_p = {k: eval(v) for k, v in L['Bagging']['params'].items()}
BAG = BaggingRegressor(**_p).fit(_d['X'], _d['y'].ravel())
_lt = np.load(LDIR + '/linux3_Bagging_trees.npz')
assert all(np.array_equal(BAG.estimators_[t].tree_.threshold, _lt[f"t{t}_threshold"]) for t in range(len(BAG.estimators_))), "refit Bagging does not match the pipeline"
for m in ("RF", "Bagging", "ExtraTrees"):
    d = np.load(f'{LDIR}/linux_{m}.npz'); X, y = d['X'].astype(np.float32), d['y'].ravel(); n = len(X)  # trees split on float32 X
    lt = np.load(f'{LDIR}/linux3_{m}_trees.npz'); mt = np.load(f'{mac_dir}/mac3_{m}_trees.npz')
    ntrees = len(L[m]['trees']); summary = []
    for t in range(ntrees):
        A = {k: lt[f"t{t}_{k}"] for k in ("feature", "threshold", "children_left", "children_right")}
        B = {k: mt[f"t{t}_{k}"] for k in ("feature", "threshold", "children_left", "children_right")}
        k = min(len(A['feature']), len(B['feature']))
        bad = [i for i in range(k) if A['feature'][i] != B['feature'][i] or A['threshold'][i] != B['threshold'][i]]
        if not bad and len(A['feature']) == len(B['feature']): continue
        node = bad[0]
        if m == "RF":
            idx = _generate_sample_indices(L[m]['trees'][t]['seed'], n, _get_n_samples_bootstrap(n, None, None), None)
            w = np.bincount(idx, minlength=n)
        elif m == "Bagging":
            w = np.bincount(BAG.estimators_samples_[t], minlength=n)
        else:
            w = np.ones(n, int)
        keep = np.arange(n) if w is None else np.nonzero(w)[0]
        if w is None: w = np.ones(n, int)
        rows = rows_at(A, X, node, keep)
        if A['feature'][node] < 0 or B['feature'][node] < 0:
            summary.append((t, node, len(rows), int(A['feature'][node]), int(B['feature'][node]), None, None)); continue
        pa, sa = proxy_exact(X, y, w, rows, A['feature'][node], A['threshold'][node])
        pb, sb = proxy_exact(X, y, w, rows, B['feature'][node], B['threshold'][node])
        summary.append((t, node, len(rows), int(A['feature'][node]), int(B['feature'][node]), pa == pb, sa == sb or sa == set(rows.tolist()) - sb))
    print(f"== {m}: {len(summary)} trees differ")
    for t, node, nr, fa, fb, tie, samepart in summary:
        print(f"   tree {t:3d} first diff node {node:4d} ({nr} rows): linux feature {fa}, mac feature {fb}; exact tie {tie}; same partition {samepart}")
    print(f"   exact ties at first differing node: {sum(s[5] is True for s in summary)} of {len(summary)}")
