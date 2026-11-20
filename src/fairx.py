"""Cross-release ShiftGuard with the Transformer: Transformer-diff (v2 scores, 59 shared variables) OR Level (59 shared
variables), on the 3x3 matrix of HAI 21.03/22.04/23.05; PCA counterpart for reference. Smoothing 10 s and 60 s."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from metrics import evaluate
R2 = "results/v2/raw"; L = "results/v4/level59"; V = ["21.03", "22.04", "23.05"]


def load(f, sm):
    d = np.load(f); k = lambda p: sorted([x for x in d.files if x[0] == p], key=lambda x: int(x[1:]))
    s = lambda a: pd.Series(a).rolling(sm, min_periods=1).mean().to_numpy()
    return np.concatenate([s(d[x]) for x in k("v")]), [s(d[x]) for x in k("t")], np.concatenate([d[x] for x in k("y")]).astype(np.int8)


al = lambda c, q: np.concatenate([t > np.percentile(c[0], q) for t in c[1]]).astype(np.int8)
rows = []
for sm in (10, 60):
    for s in V:
        for t in V:
            tag = "within-common" if s == t else "cross-common"; lev = load(f"{L}/Level_{s}_{t}.npz", sm)
            for m in ("PCA", "Transformer"):
                c = load(f"{R2}/{m}_diff_s0_{s}_{t}_{tag}.npz", sm); assert len(c[2]) == len(lev[2]) and len(c[0]) == len(lev[0])
                for name, yp in ((f"{m}-diff", al(c, 99.9)), (f"ShiftGuard-{m}", al(c, 99.95) | al(lev, 99.95)), ("Level", al(lev, 99.9))):
                    rows.append(dict(sm=sm, src=s, tgt=t, method=name, **evaluate(c[2], yp)))
D = pd.DataFrame(rows).drop_duplicates(["sm", "src", "tgt", "method"]); D.to_csv("results/v4/fair_cross59.csv", index=False)
x = D[D.src != D.tgt].groupby(["sm", "method"]).agg(eTaF1=("eTaF1", "mean"), worst=("eTaF1", "min"), FPR=("FPR", "mean"), maxFA=("fa_per_day", "max"))
print(x.round(3))
