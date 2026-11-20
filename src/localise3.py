"""(a) Attack-shape taxonomy and detection by shape; (b) root-cause localisation (HAI 20.07 and 21.03, per-process labels).

(a) For each attack, a detector-independent signal analysis on the release's own data:
    baseline_j = median of variable j over the 10 min before the attack; scale_j = IQR/1.349 of j in normal training data;
    d_j(t) = (x_j(t) - baseline_j) / scale_j inside the attack; dominant variable j* = argmax_j max_t |d_j(t)| (non-binary j);
    excursion E = max_t |d_j*(t)|; steepness = max over t of |d_j*(t) - d_j*(t-60)| / E.
    Class: subtle if E < 3; else abrupt if steepness >= 0.5; else gradual.
(b) At the first alarm inside a detected attack, the variable with the largest contribution (PCA: squared residual;
    Level: |z|) is mapped to its process (P1-P4 prefix); correct if it matches a labelled attacked process.
"""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "src")
from metrics import ranges
import shiftguard as SG
from sklearn.decomposition import PCA

V = ["20.07", "21.03", "22.04", "23.05"]; OUT = "results/v3"
allv = pickle.load(open("cache/all4_f32.pkl", "rb"))

# ---------------- (b) localisation ----------------
sm = lambda s: pd.Series(s).rolling(60, min_periods=1).mean().to_numpy()
loc = []
for v, labf in (("20.07", "cache/labels_2007.csv"), ("21.03", None)):
    cols = np.array(allv[v]["cols"]); tr = [x for x, _ in allv[v]["train"]]
    keep = np.concatenate(tr).std(0) > 1e-6; tr = [x[:, keep] for x in tr]; cols = cols[keep]
    proc = np.array([c.split("_")[0] for c in cols])
    if labf: lab = pd.read_csv(labf)
    else: lab = pd.concat([pd.read_csv(f"data/21.03/test{i}.csv.gz", usecols=["attack", "attack_P1", "attack_P2", "attack_P3"]) for i in range(1, 6)], ignore_index=True)
    X = np.concatenate([x[:, keep] for x, _ in allv[v]["test"]]); y = lab.attack.values
    fit = [x[:int(len(x) * .9)] for x in tr]; val = [x[int(len(x) * .9):] for x in tr]
    for rep in ("raw", "diff"):
        f = (lambda a: a) if rep == "raw" else (lambda a: np.vstack([np.zeros((1, a.shape[1]), a.dtype), np.diff(a, axis=0)]))
        F = np.concatenate([f(x) for x in fit]); lo, hi = F.min(0), F.max(0); r = np.where(hi - lo > 0, hi - lo, 1)
        p = PCA(0.95).fit(((F - lo) / r)[::5]); P, mu = p.components_, p.mean_
        contrib = lambda A: (((A - lo) / r - mu) - (((A - lo) / r - mu) @ P.T) @ P) ** 2
        thr = np.percentile(np.concatenate([sm(contrib(f(x)).sum(1)) for x in tr])[-sum(len(x) for x in val):], 99.9)
        Cx = contrib(f(X)); s = sm(Cx.sum(1)); yp = s > thr
        for i, (a0, a1) in enumerate(ranges(y)):
            hit = np.where(yp[a0:a1 + 1])[0]
            if not len(hit): continue
            t = a0 + hit[0]; top = proc[Cx[t].argmax()]
            truth = {pp for pp in ("P1", "P2", "P3") if lab[f"attack_{pp}"].values[a0:a1 + 1].any()}
            loc.append(dict(release=v, method=f"PCA-{rep}", attack=i, top=top, truth="+".join(sorted(truth)), correct=top in truth))
    L = SG.LevelChannel().fit(fit); Xu = X[:, L.use]; med, mad = SG._rolling_stats(Xu)
    z = np.abs(Xu - med) / np.maximum(1.4826 * mad, L.floor[L.use]); z[np.isnan(z)] = 0
    thr = np.percentile(np.concatenate([sm(L.score(x))[int(len(x) * .9):] for x in tr]), 99.9)
    yp = sm(z.max(1)) > thr; procu = proc[L.use]
    for i, (a0, a1) in enumerate(ranges(y)):
        hit = np.where(yp[a0:a1 + 1])[0]
        if not len(hit): continue
        t = a0 + hit[0]; top = procu[z[t].argmax()]
        truth = {pp for pp in ("P1", "P2", "P3") if lab[f"attack_{pp}"].values[a0:a1 + 1].any()}
        loc.append(dict(release=v, method="Level", attack=i, top=top, truth="+".join(sorted(truth)), correct=top in truth))
Lc = pd.DataFrame(loc); Lc.to_csv(f"{OUT}/localisation.csv", index=False)
print(Lc.groupby(["release", "method"]).correct.agg(["mean", "count"]).round(3))
print(Lc.groupby("method").correct.agg(["mean", "count"]).round(3))
