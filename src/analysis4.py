"""v4 analyses: RevIN baseline evaluation, detection delay, warm-up exposure, legitimate set-point change (release junction),
attack-shape threshold sensitivity, Holm-corrected McNemar, SKAB fluctuation analysis."""
import sys, pickle, itertools, numpy as np, pandas as pd
sys.path.insert(0, "src")
import shiftguard as SG, models as M
from metrics import evaluate, ranges
from scipy.stats import binomtest, spearmanr
S = "results/v3/scores2"; R3 = "results/v3"; OUT = "results/v4"; SM = 60
V = ["20.07", "21.03", "22.04", "23.05"]
sm = lambda a: pd.Series(a).rolling(SM, min_periods=1).mean().to_numpy()


def segs(f):
    d = np.load(f); k = lambda p: sorted([x for x in d.files if x[0] == p], key=lambda x: int(x[1:]))
    return np.concatenate([sm(d[x]) for x in k("v")]), [sm(d[x]) for x in k("t")], np.concatenate([d[x] for x in k("y")]).astype(np.int8)


al = lambda c, q: np.concatenate([t > np.percentile(c[0], q) for t in c[1]]).astype(np.int8)
out = open(f"{OUT}/analysis4.txt", "w")
def P(*a):
    print(*a, flush=True); print(*a, file=out, flush=True)

# (1) RevIN baseline
rows = []
for w in (60, 600):
    for s in V:
        for t, fs, tag in [(s, "full", "within")] + [(t, "common", "matrix") for t in V]:
            c = segs(f"{S}/PCA-revin{w}_s0_{s}_{t}_{fs}.npz"); rows.append(dict(method=f"PCA-RevIN{w}", train=s, test=t, setting=tag, **evaluate(c[2], al(c, 99.9))))
Rv = pd.DataFrame(rows); Rv.to_csv(f"{OUT}/revin_eval.csv", index=False)
P("RevIN within", Rv[Rv.setting == "within"].pivot_table(index="method", columns="train", values="eTaF1").assign(mean=lambda x: x.mean(1)).round(3).to_string())
P("RevIN within FA/d", Rv[Rv.setting == "within"].groupby("method").fa_per_day.mean().round(2).to_dict())
c = Rv[(Rv.setting == "matrix") & (Rv.train != Rv.test)]
P("RevIN cross", c.groupby("method").agg(e=("eTaF1", "mean"), worst=("eTaF1", "min"), fpr=("FPR", "mean"), maxfa=("fa_per_day", "max")).round(3).to_string())

# (2) detection delay (median over detected attacks, seconds) from eval3
E = pd.read_csv(f"{R3}/eval3_sm60_s0.csv", dtype={"train": str, "test": str})
P("delay_med within", E[E.setting == "within"].pivot_table(index="method", columns="train", values="delay_med").round(0).to_string())

# (3) warm-up exposure: attacks starting in the first 30 min of a test file
allv = pickle.load(open("cache/all4_f32.pkl", "rb"))
A = pd.read_csv(f"{R3}/attacks3_sm60_s0.csv", dtype={"release": str})
n_w, det_w = 0, []
for v in V:
    off = 0; starts = []
    for _, y in allv[v]["test"]:
        for a0, _ in ranges(y): starts.append(a0 + off) if a0 < SG.WARMUP else None
        off += len(y)
    y = np.concatenate([y for _, y in allv[v]["test"]]); rs = ranges(y)
    for i, (a0, _) in enumerate(rs):
        if a0 in starts:
            n_w += 1; det_w.append(A[(A.method == "ShiftGuard-PCA") & (A.release == v) & (A.attack == i)].detected.iloc[0])
P("attacks starting inside the 30-min warm-up:", n_w, "detected by ShiftGuard:", int(np.sum(det_w)))

# (4) legitimate change of operating point: last 6 h of source validation data followed by first 12 h of target normal data
common = sorted(set.intersection(*[set(allv[v]["cols"]) for v in V])); jr = []
diff = lambda a: np.vstack([np.zeros((1, a.shape[1]), a.dtype), np.diff(a, axis=0)])
for s in V:
    idx = [allv[s]["cols"].index(c) for c in common]
    tr = [x[:, idx] for x, _ in allv[s]["train"]]; keep = np.concatenate([t[::5] for t in tr]).std(0) > 1e-6
    tr = [x[:, keep] for x in tr]; ks = [int(len(x) * 0.9) for x in tr]
    F = np.concatenate([diff(x)[:k] for x, k in zip(tr, ks)]); lo, hi = F.min(0), F.max(0); r = np.where(hi - lo > 0, hi - lo, 1)
    pca = [c for c in M.CLASSICAL if c.name == "PCA"][0]().fit((F[::5] - lo) / r)
    vd = np.concatenate([sm(pca.score((diff(x) - lo) / r)[k:]) for x, k in zip(tr, ks)])
    SG.W_SEC = 6 * 3600; L = SG.LevelChannel().fit([x[:k] for x, k in zip(tr, ks)])
    vl = np.concatenate([sm(L.score(x)[k:]) for x, k in zip(tr, ks)])
    td, tl = np.percentile(vd, 99.95), np.percentile(vl, 99.95)
    pre = tr[-1][-6 * 3600:]
    for t in V:
        if t == s: continue
        ti = [allv[t]["cols"].index(c) for c in np.array(common)[keep]]
        post = allv[t]["train"][0][0][:12 * 3600][:, ti]; X = np.vstack([pre, post]); j = len(pre)
        ad = sm(pca.score((diff(X) - lo) / r)) > td; alv = sm(L.score(X)) > tl
        def stats(a):
            p = a[j:]; on = np.flatnonzero(p); return (p.mean(), (on[-1] / 3600) if len(on) else 0.0)
        (fd, ld), (fl, ll) = stats(ad), stats(alv)
        jr.append(dict(src=s, tgt=t, diff_alarm_share=fd, diff_last_alarm_h=ld, level_alarm_share=fl, level_last_alarm_h=ll,
                       level_alarm_h=alv[j:].sum() / 3600, pre_alarm_share_level=alv[:j].mean()))
J = pd.DataFrame(jr); J.to_csv(f"{OUT}/setpoint_junction.csv", index=False)
P("junction (12 h after change):", J.round(3).to_string())
P("level alarm hours: median", J.level_alarm_h.median().round(2), "max", J.level_alarm_h.max().round(2),
  "| last level alarm (h after change): median", J.level_last_alarm_h.median().round(2), "max", J.level_last_alarm_h.max().round(2),
  "| diff alarm share max", J.diff_alarm_share.max().round(4))

# (5) shape threshold sensitivity
Sh = pd.read_csv(f"{R3}/attack_shapes.csv", dtype={"release": str}); M_ = A.merge(Sh[["release", "attack", "excursion", "steepness"]], on=["release", "attack"])
sr = []
for Ec, st in itertools.product((2, 3, 5), (0.3, 0.5, 0.7)):
    cls = np.where(M_.excursion < Ec, "subtle", np.where(M_.steepness >= st, "abrupt", "gradual")); M_["cls"] = cls
    g = M_[M_.cls == "gradual"]; n = (g.method == "ShiftGuard-PCA").sum()
    d = g.groupby("method").detected.mean()
    sr.append(dict(E=Ec, steep=st, n_gradual=n, SG=d.get("ShiftGuard-PCA"), diff=d.get("PCA-diff"), raw=d.get("PCA-raw"), level=d.get("Level"),
                   SG_abrupt=M_[(M_.cls == "abrupt") & (M_.method == "ShiftGuard-PCA")].detected.mean(),
                   diff_abrupt=M_[(M_.cls == "abrupt") & (M_.method == "PCA-diff")].detected.mean()))
SR = pd.DataFrame(sr); SR.to_csv(f"{OUT}/shape_sensitivity.csv", index=False); P(SR.round(3).to_string())

# (6) Holm-corrected McNemar for ShiftGuard vs each component
w = A.pivot_table(index=["release", "attack"], columns="method", values="detected").astype(bool); ps = []
for m in ("PCA-raw", "PCA-diff", "Level"):
    b = int((w["ShiftGuard-PCA"] & ~w[m]).sum()); cc = int((~w["ShiftGuard-PCA"] & w[m]).sum())
    ps.append((m, b, cc, binomtest(b, b + cc, 0.5).pvalue if b + cc else 1.0))
ps.sort(key=lambda x: x[3]); k = len(ps); adj, run = [], 0
for i, (m, b, cc, p) in enumerate(ps): run = max(run, min(1, (k - i) * p)); adj.append((m, b, cc, p, run))
P("McNemar SG vs (discordant SG-only, other-only, p, Holm p):", [(m, b, cc, round(p, 4), round(h, 4)) for m, b, cc, p, h in adj])
pd.DataFrame(adj, columns=["vs", "sg_only", "other_only", "p", "p_holm"]).to_csv(f"{OUT}/mcnemar_sg_holm.csv", index=False)

# (7) SKAB: fluctuation-size change vs differenced false-alarm rate
import glob
D = "data/skab/data"; rd = lambda f: pd.read_csv(f, sep=";", index_col="datetime")
free = rd(f"{D}/anomaly-free/anomaly-free.csv"); cols = [c for c in free.columns if c not in ("anomaly", "changepoint")]
s0 = free[cols].diff().std(); T = pd.read_csv(f"{OUT}/skab_theory.csv")
fs = sorted(glob.glob(f"{D}/valve1/*.csv") + glob.glob(f"{D}/valve2/*.csv") + glob.glob(f"{D}/other/*.csv"))
T["fluct_ratio_max"] = [(rd(f).query("anomaly==0")[cols].diff().std() / s0).max() for f in fs]
T.to_csv(f"{OUT}/skab_theory.csv", index=False)
P("SKAB: spearman(max fluctuation ratio, diff FA)", round(spearmanr(T.fluct_ratio_max, T.fa_diff)[0], 3),
  "| files with ratio>2:", int((T.fluct_ratio_max > 2).sum()), "median diff FA there", T[T.fluct_ratio_max > 2].fa_diff.median().round(3),
  "else", T[T.fluct_ratio_max <= 2].fa_diff.median().round(3))
