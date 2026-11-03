"""How often the Linux and Mac trees predict differently. Run: python effect.py LINUX_DIR MAC_DIR."""
import sys, numpy as np
LDIR, D = sys.argv[1], sys.argv[2]
def predict(T, X):
    out = np.empty(len(X))
    for j, x in enumerate(X):
        n = 0
        while T['children_left'][n] != -1:
            n = T['children_left'][n] if x[T['feature'][n]] <= T['threshold'][n] else T['children_right'][n]
        out[j] = T['value'][n].ravel()[0]
    return out
rng = np.random.default_rng(1)
for m, nt in (("RF", 20), ("Bagging", 10), ("ExtraTrees", 100)):
    d = np.load(f'{LDIR}/linux_{m}.npz'); X = d['X'].astype(np.float32); y = d['y'].ravel()
    # off-training points: each column drawn independently from that column's training values
    Q = np.column_stack([rng.choice(X[:, f], 2000) for f in range(X.shape[1])]).astype(np.float32)
    lt, mt = np.load(f'{LDIR}/linux3_{m}_trees.npz'), np.load(f'{D}/mac3_{m}_trees.npz')
    tr = off = 0; ens_l = np.zeros(len(Q)); ens_m = np.zeros(len(Q))
    for t in range(nt):
        A = {k: lt[f"t{t}_{k}"] for k in ("feature","threshold","children_left","children_right","value","impurity","n_node_samples")}
        B = {k: mt[f"t{t}_{k}"] for k in ("feature","threshold","children_left","children_right","value","impurity","n_node_samples")}
        a, b = predict(A, X), predict(B, X); tr += np.sum(a != b)
        qa, qb = predict(A, Q), predict(B, Q); off += np.sum(qa != qb); ens_l += qa; ens_m += qb
        if m == "ExtraTrees" and not (len(A['feature']) == len(B['feature']) and np.array_equal(A['feature'], B['feature'])):
            k = min(len(A['feature']), len(B['feature']))
            node = next(i for i in range(k) if A['feature'][i] != B['feature'][i])
            leaf, other = (A, B) if A['feature'][node] < 0 else (B, A)
            print(f"   ExtraTrees tree {t} node {node}: leaf-side impurity {leaf['impurity'][node]!r}, split-side impurity {other['impurity'][node]!r}, samples {leaf['n_node_samples'][node]}")
    rel = np.abs(ens_l - ens_m) / np.abs(ens_l)
    print(f"{m}: training-row tree predictions differing {tr} of {nt*len(X)}; off-training {off} of {nt*len(Q)}; "
          f"ensemble off-training points changed {np.sum(rel>0)} of {len(Q)}, max rel {rel.max():.2e}")
