"""Evaluate ShiftGuard and baselines on the four-release study from stored component scores.

Methods (all thresholds from normal validation data only; smoothing SM seconds, causal):
  <M>-raw        : detector on raw inputs, threshold p99.9
  <M>-diff       : detector on differenced inputs, threshold p99.9
  Level          : self-re-baselining level channel, threshold p99.9
  <M>-ndiff      : detector on self-normalized differences (Delta x / causal rolling MAD), threshold p99.9
  ShiftGuard-<M> : <M>-diff OR Level, each channel at p99.95 (the false-alarm budget is split between the two channels)
  NDiffLevel-<M> : <M>-ndiff OR Level (ablation; rejected variant)
Settings: within-release (full features) and the 4x4 matrix on the 38 shared variables.
"""
import glob, os, sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from metrics import evaluate, ranges
S = "results/v3/scores2"; OUT = "results/v3"
V = ["20.07", "21.03", "22.04", "23.05"]
SM = int(os.environ.get("SM", 60)); SEED = os.environ.get("SEED", "0")


def load(comp, src, tgt, fs):
    f = f"{S}/{comp}_s{SEED if comp != 'Level' else 0}_{src}_{tgt}_{fs}.npz"
    if not os.path.exists(f): return None
    d = np.load(f); k = lambda p: sorted([x for x in d.files if x[0] == p], key=lambda x: int(x[1:]))
    sm = lambda s: pd.Series(s).rolling(SM, min_periods=1).mean().to_numpy()
    return (np.concatenate([sm(d[x]) for x in k("v")]), [sm(d[x]) for x in k("t")], [d[x] for x in k("y")])


def alarm(c, q):
    v, ts, _ = c; thr = np.percentile(v, q); return np.concatenate([(t > thr) for t in ts]).astype(np.int8)


rows, attacks = [], []
models = sorted({os.path.basename(f).split("-")[0] for f in glob.glob(f"{S}/*-raw_s{SEED}_*.npz")})
for m in models:
    for src in V:
        for tgt, fs, tag in [(src, "full", "within")] + [(t, "common", "matrix") for t in V]:
            raw, dif, lev = load(f"{m}-raw", src, tgt, fs), load(f"{m}-diff", src, tgt, fs), load("Level", src, tgt, fs)
            nd = load(f"{m}-ndiff", src, tgt, fs)
            if raw is None or dif is None or lev is None or nd is None: continue
            y = np.concatenate(raw[2]).astype(np.int8)
            preds = {f"{m}-raw": alarm(raw, 99.9), f"{m}-diff": alarm(dif, 99.9), "Level": alarm(lev, 99.9),
                     f"{m}-ndiff": alarm(nd, 99.9),
                     f"ShiftGuard-{m}": (alarm(dif, 99.95) | alarm(lev, 99.95)).astype(np.int8),
                     f"NDiffLevel-{m}": (alarm(nd, 99.95) | alarm(lev, 99.95)).astype(np.int8)}
            for meth, yp in preds.items():
                if meth == "Level" and m != models[0]: continue
                e = evaluate(y, yp)
                rows.append(dict(method=meth, base=m, train=src, test=tgt, setting=tag, cross=src != tgt, **e))
                if tag == "within":
                    for i, (a0, a1) in enumerate(ranges(y)):
                        attacks.append(dict(method=meth, release=src, attack=i, start=a0, end=a1, detected=bool(yp[a0:a1 + 1].any())))
            print(m, src, tgt, tag, {k: round(float(r["eTaF1"]), 3) for k, r in zip(preds, rows[-len(preds):])}, flush=True)
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/eval3_sm{SM}_s{SEED}.csv", index=False)
pd.DataFrame(attacks).to_csv(f"{OUT}/attacks3_sm{SM}_s{SEED}.csv", index=False)

# summary
w = R[R.setting == "within"].groupby("method")[["eTaF1", "event_recall", "FPR", "fa_per_day"]].mean()
c = R[(R.setting == "matrix") & R.cross].groupby("method")[["eTaF1", "event_recall", "FPR", "fa_per_day"]].agg(["mean", "min", "max"])
pd.set_option("display.width", 220)
print("\nWITHIN (mean over 4 releases)\n", w.round(3)); print("\nCROSS (12 pairs)\n", c.round(3))
