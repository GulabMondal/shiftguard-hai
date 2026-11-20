"""(v4: delta estimated from the first H hours of target normal data; observed on the rest)
Numerical check of Propositions 1-2 on HAI (PCA, 38 variables shared by all four releases).
Prop 1: for an operating-point offset delta, E[s(x+delta)] = E[s(x)] + ||(I-P)delta||^2 and
        s(x+delta) >= (||(I-P)delta|| - ||r(x)||)^2, so the false-alarm probability tends to 1 once ||(I-P)delta|| exceeds
        sqrt(tau) plus the normal residual norm.
Prop 2: with differenced inputs a constant offset cancels exactly, so the score distribution on shifted normal data is unchanged.
We compare predicted and observed mean scores on the target's normal test data."""
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
D = "data"
TR = {"20.07": [f"{D}/20.07/train{i}.csv.gz" for i in (1, 2)], "21.03": [f"{D}/21.03/train{i}.csv.gz" for i in (1, 2, 3)],
      "22.04": [f"{D}/22.04/train{i}.csv" for i in range(1, 7)], "23.05": [f"{D}/23.05/hai-train{i}.csv" for i in range(1, 5)]}
sep = lambda v: ";" if v == "20.07" else ","
cols = {v: set(pd.read_csv(TR[v][0], sep=sep(v), nrows=1).columns) for v in TR}
common = sorted(set.intersection(*cols.values()) - {"time", "timestamp", "attack", "Attack"})
def load(v, lab=None):
    X = pd.concat([pd.read_csv(f, sep=sep(v), usecols=common + ([lab] if lab else [])) for f in TR[v]], ignore_index=True)
    if v == "20.07":
        a = pd.concat([pd.read_csv(f, sep=";", usecols=["attack"]) for f in TR[v]], ignore_index=True).attack.values
        X = X[a == 0]
    return X[common].to_numpy(np.float64)
data = {v: load(v) for v in TR}
rows = []
for src in TR:
    A = data[src]; keep = A.std(0) > 1e-6; A = A[:, keep]; lo, hi = A.min(0), A.max(0); r = np.where(hi - lo > 0, hi - lo, 1)
    Z = (A - lo) / r; k90 = int(len(Z) * 0.9)
    pca = PCA(0.95).fit(Z[:k90:5]); Pc = pca.components_; mu = pca.mean_
    res = lambda X: (X - mu) - ((X - mu) @ Pc.T) @ Pc
    s_src = (res(Z[k90:]) ** 2).sum(1); tau = np.percentile(s_src, 99.9)
    dZ = np.diff(Z, axis=0); pd_ = PCA(0.95).fit(dZ[:k90:5]); Pd, mud = pd_.components_, pd_.mean_
    resd = lambda X: (X - mud) - ((X - mud) @ Pd.T) @ Pd
    sd_src = (resd(dZ[k90:]) ** 2).sum(1); taud = np.percentile(sd_src, 99.9)
    for tgt in TR:
      if tgt == src: continue
      Ball = (data[tgt][:, keep] - lo) / r
      for H in (1, 6, "all"):
        if H == "all": B = Ball; delta = Ball.mean(0) - Z[:k90].mean(0)
        else: B = Ball[H * 3600:]; delta = Ball[:H * 3600].mean(0) - Z[:k90].mean(0)
        rdel = delta - (delta @ Pc.T) @ Pc
        pred = s_src.mean() + (rdel ** 2).sum()
        s_t = (res(B) ** 2).sum(1)
        sd_t = (resd(np.diff(B, axis=0)) ** 2).sum(1)
        rows.append(dict(H=H, src=src, tgt=tgt, offset_norm=np.linalg.norm(delta), resid_offset=np.sqrt((rdel ** 2).sum()), sqrt_tau=np.sqrt(tau),
                         pred_mean=pred, obs_mean=s_t.mean(), obs_over_pred=s_t.mean() / pred, fa_raw=(s_t > tau).mean(),
                         fa_diff=(sd_t > taud).mean(), src_mean=s_src.mean()))
df = pd.DataFrame(rows); df.to_csv("results/v4/theory_check_v4.csv", index=False)
from scipy.stats import spearmanr
df["H"] = df.H.astype(str)
for H, g in df.groupby("H", sort=False):
    rho = spearmanr(g.pred_mean, g.obs_mean)[0]; ratio = g.resid_offset / g.sqrt_tau
    acc = ((ratio > 1) == (g.fa_raw > 0.5)).mean()
    loro = [spearmanr(g[(g.src != v) & (g.tgt != v)].pred_mean, g[(g.src != v) & (g.tgt != v)].obs_mean)[0] for v in TR]
    print(H, "rho", round(rho, 3), "obs/pred", round(g.obs_over_pred.min(), 2), round(g.obs_over_pred.max(), 2),
          "blind-prediction accuracy", round(acc, 3), "LORO rho", [round(x, 2) for x in loro], "max fa_diff", round(g.fa_diff.max(), 3),
          "ratio>1 fa range", round(g[ratio > 1].fa_raw.min(), 2), "ratio<=1 max", round(g[ratio <= 1].fa_raw.max(), 2), "n>1", int((ratio > 1).sum()))
