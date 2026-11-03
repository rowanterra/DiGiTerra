"""Compare two platform_experiment outputs. Run: python compare_experiment.py PREFIX_A PREFIX_B (each PREFIX.json and PREFIX_preds.npz)."""
import json, sys, numpy as np
la, lb = sys.argv[1], sys.argv[2]   # prefixes: linux, mac
A, B = json.load(open(la + ".json")), json.load(open(lb + ".json"))
PA, PB = np.load(la + "_preds.npz"), np.load(lb + "_preds.npz")
print(f"A {A['machine']} sklearn {A['sklearn']}  B {B['machine']} sklearn {B['sklearn']}  X same {A['X']==B['X']}  Q same {A['Q']==B['Q']}")
for key in A["fits"]:
    ta, tb = A["fits"][key]["trees"], B["fits"][key]["trees"]
    same = sum(x == y for x, y in zip(ta, tb))
    pa, pb = PA[key], PB[key]
    d = np.abs(pa - pb)
    print(f"  {key:32s} trees equal {same:3d}/{len(ta):3d}  query rows changed {int(np.sum(d > 0)):4d}/2000  max abs {d.max():.3g}  (target sd units)")
for key in [k for k in PB.files if k.startswith("artifact")]:
    if key in PA.files:
        d = np.abs(PA[key] - PB[key])
        print(f"  {key:32s} saved model, predictions changed {int(np.sum(d > 0))}/2000, max abs {d.max():.3g}")
