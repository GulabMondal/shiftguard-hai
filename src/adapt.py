"""Cross-version adaptation: how much fresh normal data from the changed plant restores detection?

Strategies (k = hours of target-version normal data, taken from the start of its first training file):
  thr-only : source model and scaler unchanged; threshold re-set on the k-hour window
  recenter : per-sensor mean shift aligns the target operating point to the source; threshold on k-hour window
  retrain  : model fitted only on the k-hour window (threshold on its last 10%)
  union    : model fitted on source training data + k-hour window (threshold on k-window last 10%)
"""
import sys, time, os, numpy as np, pandas as pd
sys.path.insert(0, "src")
from run import load, split, smooth, VERS, Q_DEPLOY, FIT_STRIDE, R
from metrics import evaluate
import models as M, deep as DM

ALL = {c.name: c for c in M.CLASSICAL + DM.DEEP}
KS = [1, 6, 24]


def scaler(F):
    keep = F.std(0) > 1e-6; lo, hi = F[:, keep].min(0), F[:, keep].max(0)
    r = np.where(hi - lo > 0, hi - lo, 1)
    return lambda A: (A[:, keep] - lo) / r


def evaluate_thr(m, norm, Xt, yt, s_calib):
    thr = np.percentile(s_calib, Q_DEPLOY)
    s = np.concatenate([smooth(m.score(norm(x))) for x in Xt]); y = np.concatenate(yt)
    return evaluate(y, (s > thr).astype(np.int8))


if __name__ == "__main__":
    allv, cols = load()
    common = sorted(set.intersection(*[set(c) for c in cols.values()]))
    names = sys.argv[1].split(",")
    os.makedirs(R, exist_ok=True)
    for name in names:
        cls = ALL[name]; out = []
        for src in VERS:
            fit_s, _, _, _ = split(allv, src, common); Fs = np.concatenate(fit_s)
            norm_s = scaler(Fs); m_s = cls().fit(norm_s(Fs[::FIT_STRIDE])); mu_s = Fs.mean(0)
            for tgt in VERS:
                if tgt == src: continue
                tr_t = allv[tgt]["train"][0][0][common].to_numpy(np.float32)
                _, _, Xt, yt = split(allv, tgt, common)
                for k in KS:
                    K = tr_t[: k * 3600]; kk = int(len(K) * 0.9)
                    t0 = time.time(); res = {}
                    # thr-only
                    res["thr-only"] = evaluate_thr(m_s, norm_s, Xt, yt, smooth(m_s.score(norm_s(K))))
                    # recenter
                    shift = K.mean(0) - mu_s
                    norm_rc = lambda A, sh=shift: norm_s(A - sh)
                    res["recenter"] = evaluate_thr(m_s, norm_rc, Xt, yt, smooth(m_s.score(norm_rc(K))))
                    # retrain on k-hour window only
                    nk = scaler(K[:kk]); mk = cls().fit(nk(K[:kk]))
                    res["retrain"] = evaluate_thr(mk, nk, Xt, yt, smooth(mk.score(nk(K[kk:]))))
                    # union of source + k-hour window
                    U = np.concatenate([Fs[::FIT_STRIDE], K[:kk]]); nu = scaler(U); mu = cls().fit(nu(U))
                    res["union"] = evaluate_thr(mu, nu, Xt, yt, smooth(mu.score(nu(K[kk:]))))
                    for strat, e in res.items():
                        out.append(dict(model=name, train=src, test=tgt, k_hours=k, strategy=strat, **e))
                    print(name, src, "->", tgt, f"k={k}h", f"{time.time()-t0:.0f}s",
                          {s: round(float(e["eTaF1"]), 3) for s, e in res.items()}, flush=True)
        pd.DataFrame(out).to_csv(f"{R}/adapt_{name}.csv", index=False)
