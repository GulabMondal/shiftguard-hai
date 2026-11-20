"""Figures and aggregated tables for the manuscript (mean over available seeds)."""
import glob, os, pickle, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src")

R = "results"; F = "paper/figs"; os.makedirs(F, exist_ok=True)
ORDER = ["Envelope", "PCA", "IForest", "kNN", "OC-SVM", "MLP-AE", "LSTM-AE", "Transformer", "USAD", "TranAD", "GDN"]
VERS = ["21.03", "22.04", "23.05"]
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300})


def load_main():
    fs = glob.glob(f"{R}/main_*.csv") + glob.glob(f"{R}/seeds/main_*.csv")
    df = pd.concat([pd.read_csv(f, dtype={"train": str, "test": str}) for f in fs], ignore_index=True)
    df["seed"] = df["seed"].fillna(0).astype(int) if "seed" in df else 0
    v2 = [pd.read_csv(f, dtype={"train": str, "test": str}) for f in glob.glob(f"{R}/v2/main_*_none_s*.csv")]
    if v2:
        df = pd.concat([df] + v2, ignore_index=True)
    df = df.drop_duplicates(["model", "seed", "setting", "train", "test", "policy"], keep="first")
    return df


def agg(df, keys, val):
    g = df.groupby(keys)[val]
    return g.mean().rename("mean").to_frame().join(g.std().fillna(0).rename("std")).join(g.size().rename("n"))


df = load_main()
dt = df[df.setting == "within-full"]

# ---------- Table: inflation (mean over seeds) ----------
rows = []
for m in ORDER:
    for v in VERS:
        g = dt[(dt.model == m) & (dt.test == v)]
        if g.empty: continue
        pa = g[g.policy == "oracle-PA"]; of = g[g.policy == "oracle-F1"]; tr = g[g.policy == "train-only"]
        rows.append(dict(model=m, version=v, n_seeds=tr.seed.nunique(),
                         oracle_PA_F1=pa.PA_F1.mean(), oracle_F1=of.F1.mean(), deploy_F1=tr.F1.mean(),
                         deploy_eTaF1=tr.eTaF1.mean(), deploy_eTaF1_sd=tr.eTaF1.std(ddof=0),
                         deploy_event_recall=tr.event_recall.mean(), deploy_FPR=tr.FPR.mean(),
                         delay_s=tr.delay_med.mean(), fa_per_day=tr.fa_per_day.mean()))
T1 = pd.DataFrame(rows); T1.round(3).to_csv(f"{R}/T1_inflation_agg.csv", index=False)

# ---------- Fig 1: inflation bars ----------
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), sharey=True)
mets = [("oracle_PA_F1", "Point-adjust F1 (oracle thr.)"), ("oracle_F1", "Point-wise F1 (oracle thr.)"),
        ("deploy_F1", "Point-wise F1 (train-only thr.)"), ("deploy_eTaF1", "eTaF1 (train-only thr.)")]
ORD = [m for m in ORDER if m in set(T1.model)]; x = np.arange(len(ORD)); w = 0.2
for ax, v in zip(axes, VERS):
    t = T1[T1.version == v].set_index("model").reindex(ORD)
    for i, (k, lab) in enumerate(mets):
        ax.bar(x + (i - 1.5) * w, t[k], w * 0.9, color=C[i], label=lab, linewidth=0)
    ax.set_title(f"HAI {v}", color=INK, fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(ORD, rotation=55, ha="right", fontsize=7)
    ax.set_ylim(0, 1); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Score")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.08), fontsize=7)
fig.savefig(f"{F}/fig_inflation.pdf"); plt.close(fig)

# ---------- Fig 2: operating-point shift ----------
D = "data"
TRF = {"21.03": [f"{D}/21.03/train{i}.csv.gz" for i in (1, 2, 3)],
       "22.04": [f"{D}/22.04/train{i}.csv" for i in range(1, 7)],
       "23.05": [f"{D}/23.05/hai-train{i}.csv" for i in range(1, 5)]}
sens = [("P2_SIT01", "Turbine speed P2_SIT01 (RPM)"), ("P1_FT01", "Flow P1_FT01"), ("P1_PIT01", "Pressure P1_PIT01")]
SH = {v: pd.concat([pd.read_csv(f, usecols=[s for s, _ in sens]) for f in TRF[v]]) for v in VERS}
fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.9))
for ax, (s, lab) in zip(axes, sens):
    data = [SH[v][s].to_numpy()[::20] for v in VERS]
    bp = ax.boxplot(data, vert=False, widths=0.55, showfliers=False, patch_artist=True,
                    medianprops=dict(color=INK, linewidth=1), whiskerprops=dict(color=INK2, linewidth=0.8),
                    capprops=dict(color=INK2, linewidth=0.8))
    for p, c in zip(bp["boxes"], C[:3]): p.set_facecolor(c); p.set_edgecolor("white")
    ax.set_yticks([1, 2, 3]); ax.set_yticklabels([f"HAI {v}" for v in VERS]); ax.set_xlabel(lab, fontsize=7.5)
    ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(f"{F}/fig_shift.pdf"); plt.close(fig)

# ---------- Fig 3: cross-version heatmaps ----------
cc = df[df.setting.isin(["within-common", "cross-common"]) & (df.policy == "train-only")]
show = ["PCA", "LSTM-AE", "Transformer"]
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f0efec", "#9ec5f4", "#2a78d6", "#104281"])
for ax, m in zip(axes, show):
    e = cc[cc.model == m].groupby(["train", "test"]).eTaF1.mean().unstack().reindex(index=VERS, columns=VERS)
    f = cc[cc.model == m].groupby(["train", "test"]).FPR.mean().unstack().reindex(index=VERS, columns=VERS)
    ax.imshow(e.values, cmap=cmap, vmin=0, vmax=0.8)
    for i in range(3):
        for j in range(3):
            val = e.values[i, j]
            ax.text(j, i, f"{val:.2f}\nFPR {f.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if val > 0.45 else INK)
    ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(VERS); ax.set_yticklabels(VERS)
    ax.set_xlabel("Test version"); ax.set_title(m, color=INK, fontsize=8.5); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)
axes[0].set_ylabel("Training version")
fig.tight_layout(); fig.savefig(f"{F}/fig_cross.pdf"); plt.close(fig)

# ---------- Fig 4: adaptation ----------
ad = pd.concat([pd.read_csv(f, dtype={"train": str, "test": str}) for f in glob.glob(f"{R}/adapt_*.csv")], ignore_index=True)
ad.to_csv(f"{R}/adapt_all.csv", index=False)
ceil = cc[cc.setting == "within-common"].groupby(["model", "test"]).eTaF1.mean()
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.2))
am = [m for m in ORDER if m in set(ad.model)]
for i, m in enumerate(am):
    g = ad[ad.model == m]
    best = g.groupby(["train", "test", "k_hours"]).eTaF1.max().groupby("k_hours").mean()
    zero = cc[(cc.model == m) & (cc.setting == "cross-common")].eTaF1.mean()
    xs = [0] + list(best.index); ys = [zero] + list(best.values)
    axes[0].plot(range(len(xs)), ys, marker="o", ms=4, lw=1.5, color=C[i], label=m)
    pair_ceiling = np.mean([ceil.get((m, t), np.nan) for t in g.test])
    axes[0].hlines(pair_ceiling, 0, 3, colors=C[i], linestyles=":", lw=1)
axes[0].set_xticks(range(4)); axes[0].set_xticklabels(["0 h", "1 h", "6 h", "24 h"])
axes[0].set_xlabel("Hours of post-change normal data"); axes[0].set_ylabel("Mean eTaF1 (6 cross pairs)")
axes[0].legend(fontsize=6.5, ncol=5, loc="lower left", bbox_to_anchor=(0, 1.02), handlelength=1.5, columnspacing=0.8); axes[0].set_ylim(0, None)
strats = ["thr-only", "recenter", "retrain", "union"]
g = ad[ad.k_hours == 24].groupby(["model", "strategy"]).eTaF1.mean().unstack().reindex(index=am, columns=strats)
xx = np.arange(len(am))
for i, s in enumerate(strats):
    axes[1].bar(xx + (i - 1.5) * 0.2, g[s], 0.18, color=C[i], label=s, linewidth=0)
axes[1].set_xticks(xx); axes[1].set_xticklabels(am, rotation=30, ha="right", fontsize=7)
axes[1].set_ylabel("Mean eTaF1 at 24 h"); axes[1].legend(fontsize=6.5, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.02)); axes[1].grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(f"{F}/fig_adapt.pdf"); plt.close(fig)

# ---------- Fig 5: cost vs accuracy ----------
cost = dt.groupby("model")["infer_us_per_row"].mean()
acc = T1.groupby("model").deploy_eTaF1.mean()
fig, ax = plt.subplots(figsize=(3.5, 2.3))
for i, m in enumerate([m for m in ORDER if m in cost.index]):
    ax.scatter(cost[m], acc[m], s=28, color=C[0] if m in ("Envelope", "PCA", "IForest", "kNN", "OC-SVM") else (C[2] if m in ("TranAD", "GDN") else C[1]),
               edgecolor="white", linewidth=1, zorder=3)
    off = {"OC-SVM": (-34, -3), "MLP-AE": (6, 3), "Transformer": (-52, 2), "kNN": (5, -6), "LSTM-AE": (5, 3), "USAD": (5, -6), "TranAD": (-14, 6), "GDN": (-22, -9), "IForest": (5, -2), "Envelope": (5, 3), "PCA": (5, 3)}.get(m, (4, 3))
    ax.annotate(m, (cost[m], acc[m]), xytext=off, textcoords="offset points", fontsize=6.5, color=INK2)
ax.set_xscale("log"); ax.set_xlabel(r"Inference cost ($\mu$s per sample, CPU, log)"); ax.set_ylabel("Mean eTaF1 (3 versions)")
ax.scatter([], [], color=C[0], label="classical"); ax.scatter([], [], color=C[1], label="deep (ours)"); ax.scatter([], [], color=C[2], label="published"); ax.legend(fontsize=6.5, loc="lower right")
fig.savefig(f"{F}/fig_cost.pdf"); plt.close(fig)

# ---------- aggregated cross table ----------
T2 = cc.groupby(["model", "train", "test"])[["eTaF1", "FPR"]].mean().round(3)
T2.to_csv(f"{R}/T2_cross_agg.csv")
print(T1.round(3).to_string(index=False)); print("seeds per model:", dt.groupby("model").seed.nunique().to_dict())
