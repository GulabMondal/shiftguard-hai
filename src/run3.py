"""Four-release study (HAI 20.07, 21.03, 22.04, 23.05): store unsmoothed component scores for ShiftGuard and baselines.

For every source release and feature set (full for within-release; the 38 variables shared by all four releases for the
complete 4x4 matrix), fit the components on the source's normal training data and score the validation part and every
target test set. Components: <MODEL>-raw, <MODEL>-diff (MODEL in PCA, Transformer, ...), and Level.
Output: results/v3/scores/<comp>_s<seed>_<src>_<tgt>_<fs>.npz  with v0.. (validation), t0.. (test), y0.. (labels)
Env: MODELS (default PCA), SEED, LEVEL=1 to also compute the level channel.
"""
import os, sys, time, pickle, numpy as np, pandas as pd
sys.path.insert(0, "src")
import models as M, deep as DM, official as O
from shiftguard import LevelChannel

V = ["20.07", "21.03", "22.04", "23.05"]; OUT = "results/v3/scores2"; os.makedirs(OUT, exist_ok=True)
ALL = {c.name: c for c in M.CLASSICAL + DM.DEEP + O.OFFICIAL}
SEED = os.environ.get("SEED", "0"); MODELS = os.environ.get("MODELS", "PCA").split(","); LEVEL = os.environ.get("LEVEL", "1") == "1"
FIT_STRIDE = 5

allv = pickle.load(open("cache/all4_f32.pkl", "rb"))
cols = {v: allv[v]["cols"] for v in V}
common = sorted(set.intersection(*[set(c) for c in cols.values()]))


def arrays(v, feats, part):
    idx = [allv[v]["cols"].index(c) for c in feats]
    return [(X[:, idx], y) for X, y in allv[v][part]]


def diff(a): return np.vstack([np.zeros((1, a.shape[1]), a.dtype), np.diff(a, axis=0)])


def ndiff(a, W=6 * 3600, step=10, warm=1800):
    """Self-normalized differences: Delta x divided by its causal rolling MAD (6 h window, 10-s grid), per variable."""
    d = diff(a); sub = pd.DataFrame(np.abs(d[::step]))
    mad = sub.rolling(W // step, min_periods=warm // step).median().fillna(0).to_numpy(np.float32)  # warm-up: floor only (causal)
    mad = np.repeat(mad, step, axis=0)[:len(d)]
    fl = sub.expanding(min_periods=1).mean().to_numpy(np.float32)  # causal floor: running mean |diff| up to t
    fl = np.maximum(np.repeat(fl, step, axis=0)[:len(d)], 1e-6)
    return d / np.maximum(1.4826 * mad, 0.25 * fl)


def save(comp, src, tgt, fs, vs, ts, ys):
    np.savez_compressed(f"{OUT}/{comp}_s{SEED}_{src}_{tgt}_{fs}.npz", **{f"v{i}": v.astype(np.float32) for i, v in enumerate(vs)},
                        **{f"t{i}": t.astype(np.float32) for i, t in enumerate(ts)}, **{f"y{i}": y for i, y in enumerate(ys)})


for src in V:
    for fs, feats, tgts in (("full", cols[src], [src]), ("common", common, V)):
        tr = [x for x, _ in arrays(src, feats, "train")]
        keep = np.concatenate(tr).std(0) > 1e-6
        tr = [x[:, keep] for x in tr]
        fit = [x[:int(len(x) * 0.9)] for x in tr]; valsplit = [int(len(x) * 0.9) for x in tr]
        tests = {t: [(x[:, keep], y) for x, y in arrays(t, feats, "test")] for t in tgts}
        for rep in os.environ.get("REPS", "raw,diff").split(","):
            f = {"raw": (lambda a: a), "diff": diff, "ndiff": ndiff}[rep]
            F = np.concatenate([f(x) for x in fit]); lo, hi = F.min(0), F.max(0); r = np.where(hi - lo > 0, hi - lo, 1)
            norm = lambda A: (A - lo) / r
            for name in MODELS:
                t0 = time.time(); m = ALL[name]().fit(norm(F[::FIT_STRIDE]))
                vs = [m.score(norm(f(x)))[k:] for x, k in zip(tr, valsplit)]     # transform whole file, keep validation tail
                for tgt, te in tests.items():
                    ts = [m.score(norm(f(x))) for x, _ in te]
                    save(f"{name}-{rep}", src, tgt, fs, vs, ts, [y for _, y in te])
                print(name, rep, src, fs, f"{time.time()-t0:.0f}s", flush=True)
        if LEVEL and SEED == "0" and os.environ.get("SKIPLEVEL") != "1":
            t0 = time.time(); L = LevelChannel().fit(fit)
            vs = [L.score(x)[k:] for x, k in zip(tr, valsplit)]
            for tgt, te in tests.items():
                save("Level", src, tgt, fs, vs, [L.score(x) for x, _ in te], [y for _, y in te])
            print("Level", src, fs, f"{time.time()-t0:.0f}s", flush=True)
