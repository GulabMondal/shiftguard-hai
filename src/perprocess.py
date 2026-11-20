"""Which attacked process (P1 boiler, P2 turbine, P3 water treatment) do detectors miss? HAI 21.03 only (per-process labels)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from run import load, fit_source, smooth, Q_DEPLOY
from metrics import ranges
import models as M, deep as DM
allv, cols = load(); feats = cols["21.03"]
raw = [pd.read_csv(f"data/21.03/test{i}.csv.gz", usecols=["attack", "attack_P1", "attack_P2", "attack_P3"]) for i in range(1, 6)]
lab = pd.concat(raw, ignore_index=True)
out = []
for cls in [M.PCARecon, M.IForest, DM.MLPAE, DM.LSTMAE]:
    S = fit_source(allv, "21.03", feats, cls)
    s = np.concatenate([smooth(S["m"].score(S["norm"](X[feats].to_numpy(np.float32)))) for X, _ in allv["21.03"]["test"]])
    yp = s > np.percentile(S["sv"], Q_DEPLOY)
    for a0, a1 in ranges(lab.attack.values):
        seg = lab.iloc[a0:a1 + 1]
        procs = "+".join(p for p in ("P1", "P2", "P3") if seg[f"attack_{p}"].any())
        out.append(dict(model=S["m"].name, start=a0, length=a1 - a0 + 1, process=procs, detected=bool(yp[a0:a1 + 1].any())))
    print(S["m"].name, "done", flush=True)
df = pd.DataFrame(out); df.to_csv("results/perprocess_2103.csv", index=False)
print(df.groupby(["process", "model"]).detected.agg(["mean", "count"]).round(2).unstack("model"))
