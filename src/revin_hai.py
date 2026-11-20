"""RevIN-style instance-normalisation baseline on the four HAI releases: PCA on causal per-variable z-scores over the
preceding W seconds (W=60 and W=600). Same splits and output format as run3.py (results/v3/scores2)."""
import os, sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "src")
import models as M
V = ["20.07", "21.03", "22.04", "23.05"]; OUT = "results/v3/scores2"
allv = pickle.load(open("cache/all4_f32.pkl", "rb"))
cols = {v: allv[v]["cols"] for v in V}; common = sorted(set.intersection(*[set(c) for c in cols.values()]))
arrays = lambda v, feats, part: [(X[:, [allv[v]["cols"].index(c) for c in feats]], y) for X, y in allv[v][part]]


def revin(a, w):
    d = pd.DataFrame(a); m = d.rolling(w, min_periods=1).mean(); s = d.rolling(w, min_periods=2).std().fillna(0)
    return ((d - m) / (s + 1e-3)).to_numpy(np.float32)


def save(comp, src, tgt, fs, vs, ts, ys):
    np.savez_compressed(f"{OUT}/{comp}_s0_{src}_{tgt}_{fs}.npz", **{f"v{i}": v.astype(np.float32) for i, v in enumerate(vs)},
                        **{f"t{i}": t.astype(np.float32) for i, t in enumerate(ts)}, **{f"y{i}": y for i, y in enumerate(ys)})


for w in (60, 600):
    f = lambda a: revin(a, w)
    for src in V:
        for fs, feats, tgts in (("full", cols[src], [src]), ("common", common, V)):
            tr = [x for x, _ in arrays(src, feats, "train")]; keep = np.concatenate(tr).std(0) > 1e-6
            tr = [x[:, keep] for x in tr]; ks = [int(len(x) * 0.9) for x in tr]
            Ftr = [f(x) for x in tr]; F = np.concatenate([a[:k] for a, k in zip(Ftr, ks)])
            lo, hi = F.min(0), F.max(0); r = np.where(hi - lo > 0, hi - lo, 1); norm = lambda A: (A - lo) / r
            m = M.PCA_().fit(norm(F[::5])) if hasattr(M, "PCA_") else [c for c in M.CLASSICAL if c.name == "PCA"][0]().fit(norm(F[::5]))
            vs = [m.score(norm(a))[k:] for a, k in zip(Ftr, ks)]; del Ftr
            for tgt in tgts:
                te = [(x[:, keep], y) for x, y in arrays(tgt, feats, "test")]
                save(f"PCA-revin{w}", src, tgt, fs, vs, [m.score(norm(f(x))) for x, _ in te], [y for _, y in te])
            print(w, src, fs, flush=True)
