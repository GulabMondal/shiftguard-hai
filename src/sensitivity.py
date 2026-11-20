"""Threshold and smoothing sensitivity (Fix M4), using saved unsmoothed scores from run2.py (within-release, full features).
Policies: fixed percentile p of smoothed validation scores (p in 99, 99.5, 99.9, 99.99) x smoothing window (1, 10, 60 s),
plus POT (static) and SPOT (streaming) thresholds of Siffer et al. (KDD 2017) with risk q = 1e-4, initial level 98%."""
import glob, os, sys, numpy as np, pandas as pd
from scipy.stats import genpareto
sys.path.insert(0, "src")
from metrics import evaluate
RAW = "results/v2/raw"; OUT = "results/v2"


def sm(x, w): return x if w == 1 else pd.Series(x).rolling(w, min_periods=1).mean().to_numpy()


def _gpd_quantile(peaks, t, n, q):
    c, _, s = genpareto.fit(peaks, floc=0)
    r = q * n / len(peaks)
    return t + (s / c) * (r ** (-c) - 1) if abs(c) > 1e-6 else t - s * np.log(r)


def pot(calib, q=1e-4, level=0.98):
    t = np.quantile(calib, level); peaks = calib[calib > t] - t
    return _gpd_quantile(peaks, t, len(calib), q)


def spot(calib, stream, q=1e-4, level=0.98, refit_every=50):
    t = np.quantile(calib, level); peaks = list(calib[calib > t] - t); n = len(calib)
    z = _gpd_quantile(np.array(peaks), t, n, q); alarm = np.zeros(len(stream), np.int8); new = 0
    for i, x in enumerate(stream):
        if x > z: alarm[i] = 1; continue
        n += 1
        if x > t:
            peaks.append(x - t); new += 1
            if new >= refit_every: z = _gpd_quantile(np.array(peaks), t, n, q); new = 0
    return alarm


def load(f):
    d = np.load(f); v = sorted(k for k in d.files if k[0] == "v"); t = sorted(k for k in d.files if k[0] == "t")
    key = lambda k: int(k[1:])
    return [d[k] for k in sorted(v, key=key)], [d[k] for k in sorted(t, key=key)], [d["y" + k[1:]] for k in sorted(t, key=key)]


rows = []
files = [f for f in glob.glob(f"{RAW}/*_none_s*_within-full.npz")] + glob.glob(f"{RAW}/*_diff_s0_*_within-full.npz")
pick = {}
for f in files:                                   # one seed per model: the lowest seed available in run2 outputs
    b = os.path.basename(f).replace(".npz", ""); model, tr, seed, src = b.split("_")[:4]; model = model if tr == "none" else model + "+diff"
    if model not in pick or int(seed[1:]) < pick[model]: pick[model] = int(seed[1:])
for f in sorted(files):
    b = os.path.basename(f).replace(".npz", ""); model, tr, seed, src = b.split("_")[:4]; model = model if tr == "none" else model + "+diff"
    if int(seed[1:]) != pick[model]: continue
    V, T, Y = load(f); y = np.concatenate(Y)
    for w in (1, 10, 60):
        sv = np.concatenate([sm(v, w) for v in V]); st = np.concatenate([sm(t, w) for t in T])
        for p in (99.0, 99.5, 99.9, 99.99):
            e = evaluate(y, (st > np.percentile(sv, p)).astype(np.int8))
            rows.append(dict(model=model, release=src, seed=int(seed[1:]), smooth=w, policy=f"p{p}", **e))
        if w == 10 and (sv > np.quantile(sv, 0.98)).sum() >= 20:   # POT/SPOT need enough peaks (not met by the Envelope rule's many zero scores)
            e = evaluate(y, (st > pot(sv)).astype(np.int8)); rows.append(dict(model=model, release=src, seed=int(seed[1:]), smooth=w, policy="POT", **e))
            a = np.concatenate([spot(sv, sm(t, w)) for t in T])
            e = evaluate(y, a); rows.append(dict(model=model, release=src, seed=int(seed[1:]), smooth=w, policy="SPOT", **e))
    print(model, src, "done", flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/sensitivity.csv", index=False)
