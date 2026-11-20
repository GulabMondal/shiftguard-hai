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

# ---------------- (a) attack shapes ----------------
rows = []
for v in V:
    cols = allv[v]["cols"]; tr = np.concatenate([x[::5] for x, _ in allv[v]["train"]])
    scale = (np.percentile(tr, 75, 0) - np.percentile(tr, 25, 0)) / 1.349
    nonbin = np.array([len(np.unique(tr[:, j])) > 10 for j in range(tr.shape[1])]) & (scale > 1e-6)
    X = np.concatenate([x for x, _ in allv[v]["test"]]); y = np.concatenate([y for _, y in allv[v]["test"]])
    for i, (a0, a1) in enumerate(ranges(y)):
        b0 = max(0, a0 - 600); base = np.median(X[b0:a0], 0) if a0 - b0 >= 60 else X[a0]
        d = (X[a0:a1 + 1][:, nonbin] - base[nonbin]) / scale[nonbin]
        j = np.abs(d).max(0).argmax(); dj = d[:, j]; E = np.abs(dj).max()
        steep = (np.abs(dj[60:] - dj[:-60]).max() if len(dj) > 60 else np.abs(dj - dj[0]).max()) / max(E, 1e-9)
        cls = "subtle" if E < 3 else ("abrupt" if steep >= 0.5 else "gradual")
        rows.append(dict(release=v, attack=i, length_s=a1 - a0 + 1, dominant=np.array(cols)[nonbin][j], excursion=E, steepness=steep, shape=cls))
S = pd.DataFrame(rows); S.to_csv(f"{OUT}/attack_shapes.csv", index=False)
print(S.groupby(["release", "shape"]).size().unstack(fill_value=0)); print(S["shape"].value_counts())

A = pd.read_csv(f"{OUT}/attacks3_sm60_s0.csv", dtype={"release": str})
M = A.merge(S[["release", "attack", "shape"]], on=["release", "attack"])
tab = M.groupby(["shape", "method"]).detected.mean().unstack().round(3)
tab["n"] = M[M.method == M.method.iloc[0]].groupby("shape").size()
tab.to_csv(f"{OUT}/detection_by_shape.csv"); print(tab)

