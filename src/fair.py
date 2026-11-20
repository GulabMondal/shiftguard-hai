"""Fair within-release comparison on HAI 21.03/22.04/23.05: every detector re-thresholded from its saved unsmoothed scores
at the same smoothing (10 s and 60 s). Adds ShiftGuard-Transformer = Transformer-diff (p99.95) OR Level (p99.95)."""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, "src")
from metrics import evaluate, ranges
R2 = "results/v2/raw"; S3 = "results/v3/scores2"; OUT = "results/v4"
os.makedirs(OUT, exist_ok=True)
V = ["21.03", "22.04", "23.05"]


def load(f, sm):
    d = np.load(f); k = lambda p: sorted([x for x in d.files if x[0] == p], key=lambda x: int(x[1:]))
    s = lambda a: pd.Series(a).rolling(sm, min_periods=1).mean().to_numpy()
    return np.concatenate([s(d[x]) for x in k("v")]), [s(d[x]) for x in k("t")], np.concatenate([d[x] for x in k("y")]).astype(np.int8)


def alarm(c, q):
    thr = np.percentile(c[0], q); return np.concatenate([t > thr for t in c[1]]).astype(np.int8)


rows, att = [], []
for sm in (10, 60):
    for v in V:
        lev = load(f"{S3}/Level_s0_{v}_{v}_full.npz", sm)
        src = {"PCA-raw": glob.glob(f"{R2}/PCA_none_s0_{v}_{v}_within-full.npz"),
               "PCA-diff": glob.glob(f"{R2}/PCA_diff_s0_{v}_{v}_within-full.npz"),
               "Transformer-raw": sorted(glob.glob(f"{R2}/Transformer_none_s*_{v}_{v}_within-full.npz")),
               "Transformer-diff": glob.glob(f"{R2}/Transformer_diff_s0_{v}_{v}_within-full.npz"),
               "TranAD-raw": sorted(glob.glob(f"{R2}/TranAD_none_s*_{v}_{v}_within-full.npz"))}
        for m, files in src.items():
            for f in files:
                c = load(f, sm); y = c[2]; assert len(y) == len(lev[2])
                seed = f.split("_s")[1].split("_")[0] if "_s" in f else "0"
                preds = {m: alarm(c, 99.9)}
                if m == "PCA-diff": preds["ShiftGuard-PCA"] = alarm(c, 99.95) | alarm(lev, 99.95)
                if m == "Transformer-diff": preds["ShiftGuard-Transformer"] = alarm(c, 99.95) | alarm(lev, 99.95)
                for mm, yp in preds.items():
                    e = evaluate(y, yp); rows.append(dict(sm=sm, release=v, method=mm, seed=seed, **e))
                    if sm == 60:
                        for i, (a0, a1) in enumerate(ranges(y)):
                            hit = np.flatnonzero(yp[a0:a1 + 1])
                            att.append(dict(method=mm, seed=seed, release=v, attack=i, detected=len(hit) > 0, delay=hit[0] if len(hit) else np.nan))
        print(sm, v, flush=True)
D = pd.DataFrame(rows); D.to_csv(f"{OUT}/fair_within.csv", index=False)
pd.DataFrame(att).to_csv(f"{OUT}/fair_attacks_sm60.csv", index=False)
g = D.groupby(["sm", "method", "release"])[["eTaF1", "fa_per_day", "FPR", "event_recall"]].mean().reset_index()
print(g.pivot_table(index=["sm", "method"], columns="release", values="eTaF1").assign(mean=lambda x: x.mean(1)).round(3))
print(g.groupby(["sm", "method"])[["fa_per_day", "FPR", "event_recall"]].mean().round(3))
