"""External check on SKAB (Skoltech Anomaly Benchmark, water-circulation testbed, 8 sensors, 1 Hz, 34 labelled experiments).

Shift protocol: detectors are fitted on the anomaly-free recording (a different day from the experiments), thresholds are set
on its last 10% (normal only), and every experiment file is scored unchanged. Reference protocol: the SKAB convention of
fitting on the first 400 (normal) rows of each experiment file. Smoothing 60 s for all methods, as on HAI.
The level channel's window is scaled to the short SKAB files (W = 10 min, 5-s grid, 2-min warm-up), fixed a priori.
"""
import glob, os, sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
import shiftguard as SG
from metrics import evaluate
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
SG.W_SEC, SG.STEP, SG.WARMUP = 600, 5, 120
D = "data/skab/data"; OUT = "results/v4"; SM = int(os.environ.get("SM", 60))
rd = lambda f: pd.read_csv(f, sep=";", index_col="datetime", parse_dates=True)
files = sorted(glob.glob(f"{D}/valve1/*.csv") + glob.glob(f"{D}/valve2/*.csv") + glob.glob(f"{D}/other/*.csv"))
free = rd(f"{D}/anomaly-free/anomaly-free.csv"); cols = [c for c in free.columns if c not in ("anomaly", "changepoint")]
diff = lambda a: np.vstack([np.zeros((1, a.shape[1])), np.diff(a, axis=0)])
sm = lambda s: pd.Series(s).rolling(SM, min_periods=1).mean().to_numpy()


class PCAdet:
    def fit(self, F):
        self.lo, r = F.min(0), F.max(0) - F.min(0); self.r = np.where(r > 0, r, 1); Z = (F - self.lo) / self.r
        self.mu = Z.mean(0); p = PCA(0.95).fit(Z - self.mu); self.P = p.components_.T @ p.components_; return self
    def resid(self, X): Z = (X - self.lo) / self.r - self.mu; return Z - Z @ self.P
    def score(self, X): return (self.resid(X) ** 2).sum(1)


def revin(a, w=60):
    """Instance normalisation in the spirit of RevIN: causal per-variable z-score over the preceding w seconds."""
    d = pd.DataFrame(a); m = d.rolling(w, min_periods=1).mean(); s = d.rolling(w, min_periods=2).std().fillna(0)
    return ((d - m) / (s + 1e-3)).to_numpy()


X0 = free[cols].to_numpy(float); k = int(len(X0) * 0.9)
reps = {"raw": lambda a: a, "diff": diff, "revin": revin}
det = {r: PCAdet().fit(f(X0[:k])) for r, f in reps.items()}
thr = {r: np.percentile(sm(det[r].score(f(X0))[k:]), 99.9) for r, f in reps.items()}
thr95 = {"diff": np.percentile(sm(det["diff"].score(diff(X0))[k:]), 99.95)}
L = SG.LevelChannel().fit([X0[:k]]); lv = sm(L.score(X0)[k:])
thrL, thrL95 = np.percentile(lv, 99.9), np.percentile(lv, 99.95)
sqrt_tau = np.sqrt(thr["raw"])

ys, preds, theo = [], {m: [] for m in ["PCA-raw", "PCA-diff", "PCA-RevIN", "Level", "ShiftGuard-PCA", "PCA-raw (per-file fit)"]}, []
for f in files:
    d = rd(f); X = d[cols].to_numpy(float); y = d["anomaly"].to_numpy().astype(np.int8); ys.append(y)
    s = {r: sm(det[r].score(fn(X))) for r, fn in reps.items()}; lvl = sm(L.score(X))
    preds["PCA-raw"].append(s["raw"] > thr["raw"]); preds["PCA-diff"].append(s["diff"] > thr["diff"])
    preds["PCA-RevIN"].append(s["revin"] > thr["revin"]); preds["Level"].append(lvl > thrL)
    preds["ShiftGuard-PCA"].append((s["diff"] > thr95["diff"]) | (lvl > thrL95))
    # reference: SKAB convention, fit on first 400 normal rows of the same file (no shift); threshold on rows 360-400
    pf = PCAdet().fit(X[:360]); ps = sm(pf.score(X)); preds["PCA-raw (per-file fit)"].append(ps > np.percentile(ps[360:400], 99.9))
    # theory check on the file's normal rows: offset of the normal part vs anomaly-free, out of PCA subspace
    n = y == 0; delta = (X[n].mean(0) - X0[:k].mean(0)) / det["raw"].r
    rd_ = delta - delta @ det["raw"].P
    base = det["raw"].score(X0[:k]).mean()
    theo.append(dict(file=f.split("data/")[1], resid_offset=np.linalg.norm(rd_), ratio=np.linalg.norm(rd_) / sqrt_tau,
                     pred=base + (rd_ ** 2).sum(), obs=det["raw"].score(X[n]).mean(),
                     fa_raw=(s["raw"][n] > thr["raw"]).mean(), fa_diff=(s["diff"][n] > thr["diff"]).mean()))
y = np.concatenate(ys); rows = []
for m, p in preds.items():
    e = evaluate(y, np.concatenate(p).astype(np.int8)); rows.append(dict(method=m, **e))
    # per-file eTaF1 too (files are separate experiments)
    rows[-1]["eTaF1_file_mean"] = np.mean([evaluate(yy, pp.astype(np.int8))["eTaF1"] for yy, pp in zip(ys, p)])
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/skab_sm{SM}.csv", index=False)
T = pd.DataFrame(theo); T.to_csv(f"{OUT}/skab_theory.csv", index=False)
print(R[["method", "eTaF1", "eTaF1_file_mean", "F1", "FPR", "event_recall", "fa_per_day"]].round(3).to_string())
print("files", len(files), "anomaly share", y.mean().round(3), "events", R.n_events.iloc[0])
print("theory: spearman(pred,obs)", spearmanr(T.pred, T.obs)[0].round(3), " ratio>1:", (T.ratio > 1).sum(),
      "FA raw ratio>1 median", T[T.ratio > 1].fa_raw.median().round(3), "ratio<=1 median", T[T.ratio <= 1].fa_raw.median().round(3),
      " max FA diff", T.fa_diff.max().round(3))
