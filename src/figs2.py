"""Revision tables and figures: representations (RQ5), published baselines, seed CIs, sensitivity, significance."""
import glob, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "results"; V2 = f"{R}/v2"; F = "paper/figs"; T = "paper/tables"
VERS = ["21.03", "22.04", "23.05"]
PAIRS = [("21.03", "22.04"), ("21.03", "23.05"), ("22.04", "21.03"), ("22.04", "23.05"), ("23.05", "21.03"), ("23.05", "22.04")]
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]; INK = "#0b0b0b"
NAME = {"IForest": "Isolation Forest", "kNN": "$k$NN", "OC-SVM": "One-class SVM"}
f2 = lambda x: "--" if pd.isna(x) else f"{x:.2f}"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.5, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300})

old = pd.concat([pd.read_csv(f, dtype={"train": str, "test": str}) for f in glob.glob(f"{R}/main_*.csv")])
old["transform"] = "none"; old["seed"] = 0
v2 = pd.concat([pd.read_csv(f, dtype={"train": str, "test": str}) for f in glob.glob(f"{V2}/main_*.csv")])
D = pd.concat([old, v2], ignore_index=True)
D = D[D.policy == "train-only"].drop_duplicates(["model", "transform", "seed", "setting", "train", "test"], keep="first")

# ---------------- RQ5: representations ----------------
rows = []
for m in ["PCA", "MLP-AE", "Transformer"]:
    for tr, lab in (("none", "Raw"), ("diff", "Differenced"), ("rollbase", "Rolling baseline")):
        g = D[(D.model == m) & (D["transform"] == tr) & (D.seed == 0)]
        if g.empty: continue
        wf = g[g.setting == "within-full"].set_index("test").eTaF1
        cr = g[g.setting == "cross-common"].set_index(["train", "test"])
        rows.append(dict(model=m, repr=lab, **{f"within_{v}": wf.get(v) for v in VERS},
                         within_mean=wf.mean(), cross_mean=np.mean([cr.loc[p, "eTaF1"] for p in PAIRS]),
                         cross_min=np.min([cr.loc[p, "eTaF1"] for p in PAIRS]),
                         cross_fpr_mean=np.mean([cr.loc[p, "FPR"] for p in PAIRS]), cross_fpr_max=np.max([cr.loc[p, "FPR"] for p in PAIRS]),
                         from2103=np.mean([cr.loc[p, "eTaF1"] for p in PAIRS[:2]])))
RP = pd.DataFrame(rows); RP.round(3).to_csv(f"{V2}/T_repr.csv", index=False)
L = [r"\begin{table*}[t]", r"\centering",
     r"\caption{Effect of the input representation (seed 0, train-only threshold). Within: eTaF1 per release on the full feature set. Cross: eTaF1 averaged over the six cross-release pairs (common feature set), its worst pair, and the mean and worst-pair false-positive rate.}",
     r"\label{tab:repr}", r"\footnotesize",
     r"\begin{tabular}{@{}ll|ccc|cccc@{}}", r"\toprule",
     r" & & \multicolumn{3}{c|}{Within-release eTaF1} & \multicolumn{4}{c}{Cross-release} \\",
     r"Detector & Input & 21.03 & 22.04 & 23.05 & eTaF1 & worst & FPR & worst FPR \\", r"\midrule"]
for i, r in RP.iterrows():
    if i and r.model != RP.model[i - 1]: L.append(r"\midrule")
    L.append(f"{r.model} & {r["repr"]} & {f2(r['within_21.03'])} & {f2(r['within_22.04'])} & {f2(r['within_23.05'])} & "
             f"{f2(r.cross_mean)} & {f2(r.cross_min)} & {f2(r.cross_fpr_mean)} & {f2(r.cross_fpr_max)} " + r"\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
open(f"{T}/tab_repr.tex", "w").write("\n".join(L))

fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.2))
ms = list(RP.model.unique()); reps = ["Raw", "Differenced", "Rolling baseline"]; x = np.arange(len(ms))
for i, rp in enumerate(reps):
    g = RP[RP.repr == rp].set_index("model").reindex(ms)
    axes[0].bar(x + (i - 1) * 0.26, g.cross_mean, 0.24, color=C[i], label=rp, linewidth=0)
    axes[1].bar(x + (i - 1) * 0.26, g.cross_fpr_mean, 0.24, color=C[i], label=rp, linewidth=0)
axes[0].set_ylabel("Cross-release eTaF1 (mean of 6 pairs)"); axes[1].set_ylabel("Cross-release false-positive rate")
for a in axes: a.set_xticks(x); a.set_xticklabels(ms); a.grid(axis="x", visible=False)
axes[0].legend(fontsize=7, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02)); axes[1].set_ylim(0, 1)
fig.tight_layout(); fig.savefig(f"{F}/fig_repr.pdf"); plt.close(fig)
print(RP.round(3).to_string(index=False))
