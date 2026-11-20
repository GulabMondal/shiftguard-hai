"""Level channel on the 59 variables shared by HAI 21.03/22.04/23.05, matching the v2 cross-release files, so that
ShiftGuard can be combined with the Transformer (whose differenced-input scores were saved in v2)."""
import os, sys, pickle, numpy as np
sys.path.insert(0, "src")
from shiftguard import LevelChannel
V = ["21.03", "22.04", "23.05"]; OUT = "results/v4/level59"; os.makedirs(OUT, exist_ok=True)
allv = pickle.load(open("cache/all4_f32.pkl", "rb"))
common = sorted(set.intersection(*[set(allv[v]["cols"]) for v in V])); print(len(common), "shared variables")
arr = lambda v, part: [(X[:, [allv[v]["cols"].index(c) for c in common]], y) for X, y in allv[v][part]]
for src in V:
    tr = [x for x, _ in arr(src, "train")]; ks = [int(len(x) * 0.9) for x in tr]
    L = LevelChannel().fit([x[:k] for x, k in zip(tr, ks)])
    vs = [L.score(x)[k:] for x, k in zip(tr, ks)]
    for tgt in V:
        te = arr(tgt, "test")
        np.savez_compressed(f"{OUT}/Level_{src}_{tgt}.npz", **{f"v{i}": a for i, a in enumerate(vs)},
                            **{f"t{i}": L.score(x) for i, (x, _) in enumerate(te)}, **{f"y{i}": y for i, (_, y) in enumerate(te)})
        print(src, tgt, flush=True)
