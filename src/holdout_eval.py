"""Evaluate the configuration selected on HAI 21.03 (holdout.py) on the held-out releases 20.07, 22.04, 23.05:
within-release (full features) and the six cross pairs among them (38 shared variables)."""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "src")
import shiftguard as SG
from metrics import evaluate
S = "results/v3/scores2"; OUT = "results/v4"; SM = 60
G = pd.read_csv(f"{OUT}/holdout_grid_2103.csv").iloc[0]; Wh, QD, QL = int(G.W_h), G.qd, G.ql; print("selected", Wh, QD, QL)
SG.W_SEC = Wh * 3600
allv = pickle.load(open("cache/all4_f32.pkl", "rb"))
H = ["20.07", "22.04", "23.05"]; common = sorted(set.intersection(*[set(allv[v]["cols"]) for v in allv]))
sm = lambda a: pd.Series(a).rolling(SM, min_periods=1).mean().to_numpy()


def segs(d):
    k = lambda p: sorted([x for x in d.files if x[0] == p], key=lambda x: int(x[1:]))
    return np.concatenate([sm(d[x]) for x in k("v")]), [sm(d[x]) for x in k("t")], np.concatenate([d[x] for x in k("y")]).astype(np.int8)


al = lambda c, q: np.concatenate([t > np.percentile(c[0], q) for t in c[1]]).astype(np.int8)
rows = []
for src in H:
    for fs, feats, tgts in (("full", allv[src]["cols"], [src]), ("common", common, H)):
        idx = [allv[src]["cols"].index(c) for c in feats]
        tr = [x[:, idx] for x, _ in allv[src]["train"]]; keep = np.concatenate([t[::5] for t in tr]).std(0) > 1e-6
        tr = [x[:, keep] for x in tr]; ks = [int(len(x) * 0.9) for x in tr]
        L = SG.LevelChannel().fit([x[:k] for x, k in zip(tr, ks)]); lv_v = np.concatenate([sm(L.score(x)[k:]) for x, k in zip(tr, ks)])
        for tgt in tgts:
            ti = [allv[tgt]["cols"].index(c) for c in np.array(feats)[keep]]
            lev = (lv_v, [sm(L.score(x[:, ti])) for x, _ in allv[tgt]["test"]], None)
            dif = segs(np.load(f"{S}/PCA-diff_s0_{src}_{tgt}_{fs}.npz")); old = segs(np.load(f"{S}/Level_s0_{src}_{tgt}_{fs}.npz"))
            assert len(dif[0]) == len(lv_v)
            for name, yp in (("PCA-diff", al(dif, 99.9)), ("ShiftGuard (v3 config)", al(dif, 99.95) | al(old, 99.95)),
                             ("ShiftGuard (selected on 21.03)", al(dif, QD) | al(lev, QL))):
                rows.append(dict(src=src, tgt=tgt, setting="within" if fs == "full" else ("diag" if src == tgt else "cross"), method=name, **evaluate(dif[2], yp)))
            print(src, tgt, fs, flush=True)
D = pd.DataFrame(rows); D.to_csv(f"{OUT}/holdout_eval.csv", index=False)
print(D[D.setting == "within"].pivot_table(index="method", columns="src", values="eTaF1").assign(mean=lambda x: x.mean(1)).round(3))
print(D[D.setting == "within"].groupby("method").fa_per_day.mean().round(2))
print(D[D.setting == "cross"].groupby("method").agg(e=("eTaF1", "mean"), worst=("eTaF1", "min"), fpr=("FPR", "mean"), maxfa=("fa_per_day", "max")).round(3))
