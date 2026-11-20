"""Experiment runner: within-version and cross-version evaluation with train-only thresholds."""
import pickle, json, time, sys, os, numpy as np, pandas as pd
sys.path.insert(0, "src")
from metrics import evaluate
import models as M
import deep as DM
ALL = M.CLASSICAL + DM.DEEP

C = "cache"; R = "results"
VERS = ["21.03", "22.04", "23.05"]
SMOOTH = 10        # causal moving-average window (s)
Q_DEPLOY = 99.9    # train-only threshold percentile on held-out normal data
FIT_STRIDE = 5     # subsample normal data for fitting


def load():
    allv = pickle.load(open(f"{C}/all.pkl", "rb"))
    cols = {v: list(allv[v]["train"][0][0].columns) for v in VERS}
    return allv, cols


def split(allv, v, feats):
    fit, val = [], []
    for X, _ in allv[v]["train"]:
        a = X[feats].to_numpy(np.float32); k = int(len(a) * 0.9)
        fit.append(a[:k]); val.append(a[k:])
    Xt = [X[feats].to_numpy(np.float32) for X, _ in allv[v]["test"]]
    yt = [y.astype(np.int8) for _, y in allv[v]["test"]]
    return fit, val, Xt, yt


def smooth(s):
    return pd.Series(s).rolling(SMOOTH, min_periods=1).mean().to_numpy()


def fit_source(allv, src, feats, model_cls):
    fit, val, _, _ = split(allv, src, feats)
    F = np.concatenate(fit)
    keep = F.std(0) > 1e-6                                 # drop sensors constant in training
    lo, hi = F[:, keep].min(0), F[:, keep].max(0); rng_ = np.where(hi - lo > 0, hi - lo, 1)
    norm = lambda A: (A[:, keep] - lo) / rng_
    t0 = time.time(); m = model_cls().fit(norm(F[::FIT_STRIDE])); t_fit = time.time() - t0
    sv = np.concatenate([smooth(m.score(norm(v))) for v in val])
    return dict(m=m, norm=norm, sv=sv, t_fit=t_fit, n_feat=int(keep.sum()))


def eval_target(allv, S, src, tgt, feats, tag):
    _, _, Xt, yt = split(allv, tgt, feats)
    m, norm = S["m"], S["norm"]
    t0 = time.time(); st = [smooth(m.score(norm(x))) for x in Xt]; t_inf = time.time() - t0
    s = np.concatenate(st); y = np.concatenate(yt)
    rows = []
    thr = np.percentile(S["sv"], Q_DEPLOY)            # (1) train-only threshold
    rows.append(dict(policy="train-only", **evaluate(y, (s > thr).astype(np.int8))))
    qs = np.unique(np.percentile(s, np.linspace(80, 99.99, 120)))   # (2) oracle thresholds on test labels
    best_f, best_pa = None, None
    for t in qs:
        e = evaluate(y, (s > t).astype(np.int8))
        if best_f is None or e["F1"] > best_f["F1"]: best_f = e
        if best_pa is None or e["PA_F1"] > best_pa["PA_F1"]: best_pa = e
    rows.append(dict(policy="oracle-F1", **best_f)); rows.append(dict(policy="oracle-PA", **best_pa))
    for r in rows:
        r.update(model=m.name, train=src, test=tgt, setting=tag, n_feat=S["n_feat"],
                 fit_s=round(S["t_fit"], 1), infer_us_per_row=round(1e6 * t_inf / len(y), 2))
    np.save(f"{R}/scores/{m.name}_{src}_{tgt}_{tag}" + ("" if os.environ.get("SEED", "0") == "0" else "_s" + os.environ["SEED"]) + ".npy", s.astype(np.float32))
    return rows


if __name__ == "__main__":
    import os; os.makedirs(R, exist_ok=True)
    allv, cols = load()
    common = sorted(set.intersection(*[set(c) for c in cols.values()]))
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else [c.name for c in M.CLASSICAL]
    todo = [c for c in ALL if c.name in names]
    os.makedirs(f"{R}/scores", exist_ok=True); os.makedirs(f"{R}/seeds", exist_ok=True)
    for cls in todo:
        out = []
        for src in VERS:
            for fs_name, feats in (("full", cols[src]), ("common", common)):
                t0 = time.time(); S = fit_source(allv, src, feats, cls)
                tgts = [src] if fs_name == "full" else VERS
                for tgt in tgts:
                    tag = "within-full" if fs_name == "full" else ("within-common" if tgt == src else "cross-common")
                    out += eval_target(allv, S, src, tgt, feats, tag)
                    print(cls.name, src, "->", tgt, tag, f"{time.time()-t0:.0f}s",
                          {k: round(float(out[-3][k]), 3) for k in ("F1", "PA_F1", "eTaF1", "event_recall", "FPR")}, flush=True)
        sd = os.environ.get("SEED", "0"); pd.DataFrame(out).assign(seed=int(sd)).to_csv(f"{R}/main_{cls.name}.csv" if sd == "0" else f"{R}/seeds/main_{cls.name}_s{sd}.csv", index=False)
