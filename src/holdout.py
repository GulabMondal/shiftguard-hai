"""Hold-out protocol: ShiftGuard's free parameters (level window W, per-channel percentiles) are selected on HAI 21.03
(within-release) only; the selected configuration is then reported on 20.07, 22.04, 23.05 and the cross-release pairs
that do not involve 21.03. Smoothing fixed at 60 s."""
import sys, pickle, itertools, numpy as np, pandas as pd
sys.path.insert(0, "src")
import shiftguard as SG
from metrics import evaluate
S = "results/v3/scores2"; OUT = "results/v4"; SM = 60
allv = pickle.load(open("cache/all4_f32.pkl", "rb"))


def segs(d):
    k = lambda p: sorted([x for x in d.files if x[0] == p], key=lambda x: int(x[1:]))
    s = lambda a: pd.Series(a).rolling(SM, min_periods=1).mean().to_numpy()
    return np.concatenate([s(d[x]) for x in k("v")]), [s(d[x]) for x in k("t")], np.concatenate([d[x] for x in k("y")]).astype(np.int8)


al = lambda c, q: np.concatenate([t > np.percentile(c[0], q) for t in c[1]]).astype(np.int8)
v = "21.03"; dif = segs(np.load(f"{S}/PCA-diff_s0_{v}_{v}_full.npz"))
tr = [x for x, _ in allv[v]["train"]]; keep = np.concatenate([t[::5] for t in tr]).std(0) > 1e-6
tr = [x[:, keep] for x in tr]; ks = [int(len(x) * 0.9) for x in tr]; te = [x[:, keep] for x, _ in allv[v]["test"]]
rows = []
for Wh in (1, 3, 6, 12):
    SG.W_SEC = Wh * 3600
    L = SG.LevelChannel().fit([x[:k] for x, k in zip(tr, ks)])
    raw = {"v": [L.score(x)[k:] for x, k in zip(tr, ks)], "t": [L.score(x) for x in te]}
    s = lambda a: pd.Series(a).rolling(SM, min_periods=1).mean().to_numpy()
    lev = (np.concatenate([s(a) for a in raw["v"]]), [s(a) for a in raw["t"]], dif[2])
    for qd, ql in itertools.product((99.9, 99.95, 99.99), (99.9, 99.95, 99.99)):
        e = evaluate(dif[2], al(dif, qd) | al(lev, ql)); rows.append(dict(W_h=Wh, qd=qd, ql=ql, eTaF1=e["eTaF1"], fa=e["fa_per_day"]))
    print(Wh, flush=True)
G = pd.DataFrame(rows).sort_values("eTaF1", ascending=False); G.to_csv(f"{OUT}/holdout_grid_2103.csv", index=False)
print(G.head(10).round(3).to_string())
