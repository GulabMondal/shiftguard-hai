"""Revision runner.

Environment variables
  SEED       random seed (default 0)
  TRANSFORM  none | diff | rollbase   (input representation, applied per contiguous file before scaling)
  SETTINGS   within | all            (within: within-release on the full feature set only;
                                       all: also common-feature within-release and cross-release)
Writes  results/v2/main_<model>_<transform>_s<seed>.csv
        results/v2/raw/<model>_<transform>_s<seed>_<src>_<tgt>_<tag>.npz   (unsmoothed val/test scores per file)
        results/v2/attacks/<model>_<transform>_s<seed>_<src>_<tgt>_<tag>.npy  (per-attack detection, train-only policy)
"""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, "src")
from run import load, VERS, Q_DEPLOY, FIT_STRIDE
from metrics import evaluate, ranges
import models as M, deep as DM, official as O

ALL = {c.name: c for c in M.CLASSICAL + DM.DEEP + O.OFFICIAL}
SEED = os.environ.get("SEED", "0"); TRANSFORM = os.environ.get("TRANSFORM", "none")
SETTINGS = os.environ.get("SETTINGS", "all"); SMOOTH = 10; ROLL = 600
OUT = "results/v2"
for d in ("", "/raw", "/attacks"): os.makedirs(OUT + d, exist_ok=True)


def transform(a):
    if TRANSFORM == "none": return a
    if TRANSFORM == "diff":                                   # first difference: invariant to constant offsets
        return np.vstack([np.zeros((1, a.shape[1]), a.dtype), np.diff(a, axis=0)])
    if TRANSFORM == "rollbase":                               # deviation from causal 10-min rolling mean
        return (a - pd.DataFrame(a).rolling(ROLL, min_periods=1).mean().to_numpy()).astype(np.float32)
    raise ValueError(TRANSFORM)


def split(allv, v, feats):
    fit, val = [], []
    for X, _ in allv[v]["train"]:
        a = transform(X[feats].to_numpy(np.float32)); k = int(len(a) * 0.9)
        fit.append(a[:k]); val.append(a[k:])
    Xt = [transform(X[feats].to_numpy(np.float32)) for X, _ in allv[v]["test"]]
    yt = [y.astype(np.int8) for _, y in allv[v]["test"]]
    return fit, val, Xt, yt


def sm(s): return pd.Series(s).rolling(SMOOTH, min_periods=1).mean().to_numpy()


def fit_source(allv, src, feats, cls):
    fit, val, _, _ = split(allv, src, feats); F = np.concatenate(fit)
    keep = F.std(0) > 1e-6
    lo, hi = F[:, keep].min(0), F[:, keep].max(0); r = np.where(hi - lo > 0, hi - lo, 1)
    norm = lambda A: (A[:, keep] - lo) / r
    t0 = time.time(); m = cls().fit(norm(F[::FIT_STRIDE])); t_fit = time.time() - t0
    vraw = [m.score(norm(v)) for v in val]
    return dict(m=m, norm=norm, vraw=vraw, sv=np.concatenate([sm(v) for v in vraw]), t_fit=t_fit, n_feat=int(keep.sum()))


def eval_target(allv, S, name, src, tgt, feats, tag):
    _, _, Xt, yt = split(allv, tgt, feats); m, norm = S["m"], S["norm"]
    t0 = time.time(); traw = [m.score(norm(x)) for x in Xt]; t_inf = time.time() - t0
    s = np.concatenate([sm(t) for t in traw]); y = np.concatenate(yt)
    key = f"{name}_{TRANSFORM}_s{SEED}_{src}_{tgt}_{tag}"
    np.savez_compressed(f"{OUT}/raw/{key}.npz", **{f"v{i}": v.astype(np.float32) for i, v in enumerate(S["vraw"])},
                        **{f"t{i}": t.astype(np.float32) for i, t in enumerate(traw)}, **{f"y{i}": yy for i, yy in enumerate(yt)})
    thr = np.percentile(S["sv"], Q_DEPLOY); yp = (s > thr).astype(np.int8)
    np.save(f"{OUT}/attacks/{key}.npy", np.array([yp[a0:a1 + 1].any() for a0, a1 in ranges(y)]))
    rows = [dict(policy="train-only", **evaluate(y, yp))]
    best_f = best_pa = None
    for t in np.unique(np.percentile(s, np.linspace(80, 99.99, 120))):
        e = evaluate(y, (s > t).astype(np.int8))
        if best_f is None or e["F1"] > best_f["F1"]: best_f = e
        if best_pa is None or e["PA_F1"] > best_pa["PA_F1"]: best_pa = e
    rows += [dict(policy="oracle-F1", **best_f), dict(policy="oracle-PA", **best_pa)]
    for r in rows:
        r.update(model=name, transform=TRANSFORM, seed=int(SEED), train=src, test=tgt, setting=tag, n_feat=S["n_feat"],
                 fit_s=round(S["t_fit"], 1), infer_us_per_row=round(1e6 * t_inf / len(y), 2))
    return rows


if __name__ == "__main__":
    allv, cols = load(); common = sorted(set.intersection(*[set(c) for c in cols.values()]))
    for name in sys.argv[1].split(","):
        cls = ALL[name]; out = []
        for src in VERS:
            plans = [("full", cols[src], [src])]
            if SETTINGS == "all": plans.append(("common", common, VERS))
            for fs, feats, tgts in plans:
                t0 = time.time(); S = fit_source(allv, src, feats, cls)
                for tgt in tgts:
                    tag = "within-full" if fs == "full" else ("within-common" if tgt == src else "cross-common")
                    out += eval_target(allv, S, name, src, tgt, feats, tag)
                    print(name, TRANSFORM, f"s{SEED}", src, "->", tgt, tag, f"{time.time()-t0:.0f}s",
                          {k: round(float(out[-3][k]), 3) for k in ("F1", "PA_F1", "eTaF1", "event_recall", "FPR")}, flush=True)
        pd.DataFrame(out).to_csv(f"{OUT}/main_{name}_{TRANSFORM}_s{SEED}.csv", index=False)
