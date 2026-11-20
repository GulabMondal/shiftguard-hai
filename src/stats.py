"""Seed aggregation with 95% CIs, and paired significance tests on per-attack detection (Fix M3).

* eTaF1 (train-only policy, within-release, full features): mean and 95% t-interval over all available seeds
  (seeds 0-1 from run.py and seeds 2-4 from run2.py; same protocol).
* Per-attack detection (160 attacks pooled over the three releases): Cochran's Q across all detectors,
  and exact McNemar tests of PCA against every other detector, per seed; Holm correction across detectors.
"""
import glob, os, re, numpy as np, pandas as pd
from scipy import stats
R = "results"; V2 = f"{R}/v2"
ORDER = ["Envelope", "PCA", "IForest", "kNN", "OC-SVM", "MLP-AE", "LSTM-AE", "Transformer", "USAD", "TranAD", "GDN"]
VERS = ["21.03", "22.04", "23.05"]

# ---------- seed-level results, within-release, train-only ----------
old = [pd.read_csv(f, dtype={"train": str, "test": str}) for f in glob.glob(f"{R}/main_*.csv") + glob.glob(f"{R}/seeds/main_*.csv")]
old = pd.concat(old); old["seed"] = old["seed"].fillna(0).astype(int) if "seed" in old else 0
old["transform"] = "none"
new = pd.concat([pd.read_csv(f, dtype={"train": str, "test": str}) for f in glob.glob(f"{V2}/main_*_none_s*.csv")])
allr = pd.concat([old, new], ignore_index=True)
w = allr[(allr.setting == "within-full") & (allr.policy == "train-only") & (allr["transform"] == "none")]
w = w.drop_duplicates(["model", "test", "seed"], keep="last")
rows = []
for m in ORDER:
    for v in VERS:
        x = w[(w.model == m) & (w.test == v)].eTaF1.values
        if len(x) == 0: continue
        half = (stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 2 else (np.ptp(x) / 2 if len(x) == 2 else 0.0))
        rows.append(dict(model=m, release=v, n_seeds=len(x), eTaF1_mean=x.mean(), ci95_half=half, sd=x.std(ddof=1) if len(x) > 1 else 0.0))
ci = pd.DataFrame(rows); ci.to_csv(f"{V2}/seed_ci.csv", index=False)

# ---------- per-attack paired tests ----------
A = {}
for f in glob.glob(f"{V2}/attacks/*_none_s*_within-full.npy"):
    b = os.path.basename(f)[:-4]; m, _, s, src = b.split("_")[:4]
    A.setdefault((m, int(s[1:])), {})[src] = np.load(f)
vec = {k: np.concatenate([d[v] for v in VERS]) for k, d in A.items() if all(v in d for v in VERS)}
pca = vec[("PCA", 0)]
tests = []
for (m, s), x in sorted(vec.items()):
    if m == "PCA": continue
    b = int(((pca == 1) & (x == 0)).sum()); c = int(((pca == 0) & (x == 1)).sum())
    p = stats.binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0          # exact McNemar
    tests.append(dict(model=m, seed=s, pca_only=b, other_only=c, pca_detected=int(pca.sum()), other_detected=int(x.sum()), p=p))
T = pd.DataFrame(tests)
# Holm correction within each seed index across models
T["p_holm"] = np.nan
for s, g in T.groupby("seed"):
    order = g.p.sort_values().index; k = len(order)
    adj = np.maximum.accumulate([min(1, (k - i) * T.loc[j, "p"]) for i, j in enumerate(order)])
    T.loc[order, "p_holm"] = adj
T.to_csv(f"{V2}/mcnemar_pca.csv", index=False)

# Cochran's Q on all detectors using the lowest available seed of each
first = {}
for (m, s) in vec:
    if m not in first or s < first[m]: first[m] = s
M = np.stack([vec[(m, first[m])] for m in ORDER if m in first], 1).astype(int)
k = M.shape[1]; Cj = M.sum(0); Ri = M.sum(1); N = M.sum()
Q = (k - 1) * (k * (Cj ** 2).sum() - N ** 2) / (k * N - (Ri ** 2).sum())
pQ = stats.chi2.sf(Q, k - 1)
with open(f"{V2}/cochran.txt", "w") as fh:
    fh.write(f"Cochran Q = {Q:.2f}, df = {k-1}, p = {pQ:.3g}, detectors = {[m for m in ORDER if m in first]}, attacks = {M.shape[0]}\n")
print(ci.round(3).to_string(index=False)); print(T.round(4).to_string(index=False)); print(open(f"{V2}/cochran.txt").read())

# ---------- LaTeX table ----------
NAME = {"IForest": "Isolation Forest", "kNN": "$k$NN", "OC-SVM": "One-class SVM"}
L = [r"\begin{table*}[t]", r"\centering",
     r"\caption{Deployable eTaF1 (train-only threshold) as mean $\pm$ 95\% confidence half-width over seeds ($n$; for $n=2$, half the range of the two runs), and paired comparison with PCA on the 160 attacks. Detected: attacks detected, mean over the seeds with per-attack outcomes. McNemar: exact test against PCA, Holm-corrected across detectors; the largest corrected $p$ over seeds is shown, with the direction of the difference.}",
     r"\label{tab:stats}", r"\footnotesize",
     r"\begin{tabular}{@{}lc|ccc|cc@{}}", r"\toprule",
     r"Detector & $n$ & 21.03 & 22.04 & 23.05 & Detected (of 160) & McNemar vs.\ PCA \\", r"\midrule"]
for m in ORDER:
    g = ci[ci.model == m].set_index("release")
    if g.empty: continue
    cells = []
    for v in VERS:
        r = g.loc[v]; cells.append(f"{r.eTaF1_mean:.2f}" + (f" $\\pm$ {r.ci95_half:.2f}" if r.n_seeds > 1 else ""))
    if m == "PCA":
        det = f"{int(pca.sum())}"; mc = "--"
    else:
        t = T[T.model == m]
        if t.empty:
            det, mc = "--", "--"
        else:
            det = f"{t.other_detected.mean():.0f}"; pmax = t.p_holm.max()
            more = (t.other_only > t.pca_only).all(); less = (t.other_only < t.pca_only).all()
            dirn = "more" if more else ("fewer" if less else "mixed")
            mc = (f"{dirn}, $p\\le${pmax:.3f}" if pmax < 0.05 else f"n.s.\\ ($p\\ge${t.p_holm.min():.2f})") if pmax >= 0.001 else f"{dirn}, $p<0.001$"
            if pmax >= 0.05 and t.p_holm.min() < 0.05: mc = f"{dirn}, sig.\\ in {(t.p_holm < 0.05).sum()}/{len(t)} seeds"
    if m in ("MLP-AE", "TranAD"): L.append(r"\midrule")
    L.append(f"{NAME.get(m, m)} & {int(g.n_seeds.max())} & " + " & ".join(cells) + f" & {det} & {mc} " + r"\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
open("paper/tables/tab_stats.tex", "w").write("\n".join(L))
