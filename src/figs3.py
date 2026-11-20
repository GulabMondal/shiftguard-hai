"""Figures and tables for the theory check, ShiftGuard four-release study, attack shapes and localisation."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "results/v3"; F = "paper/figs"; T = "paper/tables"
V = ["20.07", "21.03", "22.04", "23.05"]
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]; INK = "#0b0b0b"; INK2 = "#52514e"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.5, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300})
f2 = lambda x: f"{x:.2f}"

# ---- theory figure ----
th = pd.read_csv("results/v3_theory_check.csv", dtype={"src": str, "tgt": str})
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4))
ax = axes[0]
ax.plot([0.08, 30], [0.08, 30], color="#c3c2b7", lw=0.8, ls="--", zorder=1)
sc = ax.scatter(th.pred_mean, th.obs_mean, c=th.fa_raw, cmap="Blues", vmin=0, vmax=1, s=32, edgecolor=INK2, linewidth=0.5, zorder=3)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"Predicted mean score  $E[s]+\|(I-P)\delta\|^2$"); ax.set_ylabel("Observed mean score on shifted normal data")
cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02); cb.set_label("False-alarm rate (raw)", fontsize=7)
ax.text(0.04, 0.92, r"Spearman $\rho$ = " + f"{th[['pred_mean','obs_mean']].corr('spearman').iloc[0,1]:.2f}", transform=ax.transAxes, fontsize=7.5)
ax = axes[1]
ratio = th.resid_offset / th.sqrt_tau
ax.scatter(ratio, th.fa_raw, s=30, color=C[0], label="raw inputs", zorder=3, edgecolor="white", linewidth=0.8)
ax.scatter(ratio, th.fa_diff, s=30, color=C[1], marker="s", label="differenced inputs", zorder=3, edgecolor="white", linewidth=0.8)
ax.axvline(1.0, color="#c3c2b7", lw=0.8, ls="--")
ax.set_xlabel(r"$\|(I-P)\delta\| / \sqrt{\tau}$"); ax.set_ylabel("False-alarm rate on shifted normal data"); ax.set_ylim(-0.03, 1.05)
ax.legend(fontsize=7, loc="center right")
fig.tight_layout(); fig.savefig(f"{F}/fig_theory.pdf"); plt.close(fig)

# ---- ShiftGuard table ----
E = pd.read_csv(f"{R}/eval3_sm60_s0.csv", dtype={"train": str, "test": str})
order = [("PCA-raw", "PCA, raw inputs"), ("PCA-diff", "PCA, differenced inputs"), ("Level", "Level channel only"),
         ("NDiffLevel-PCA", "Normalized diff.\\ + level (rejected)"), ("ShiftGuard-PCA", "\\textbf{ShiftGuard-PCA}")]
w = E[E.setting == "within"]; c = E[(E.setting == "matrix") & E.cross]
L = [r"\begin{table*}[t]", r"\centering",
     r"\caption{ShiftGuard on four HAI releases (PCA detector, 60-s smoothing, thresholds from normal validation data only). Within-release: full feature set, eTaF1 per release and mean false alarms per day (FA/d). Cross-release: the 12 ordered pairs on the 38 variables shared by all releases; mean and worst-pair eTaF1, mean false-positive rate, and worst-pair false alarms per day.}",
     r"\label{tab:shiftguard}", r"\footnotesize",
     r"\begin{tabular}{@{}l|cccc|cc|cccc@{}}", r"\toprule",
     r" & \multicolumn{4}{c|}{Within-release eTaF1} & \multicolumn{2}{c|}{Within mean} & \multicolumn{4}{c}{Cross-release (12 pairs)} \\",
     r"Method & 20.07 & 21.03 & 22.04 & 23.05 & eTaF1 & FA/d & eTaF1 & worst & FPR & worst FA/d \\", r"\midrule"]
for m, lab in order:
    ww = w[w.method == m].set_index("train"); cc = c[c.method == m]
    L.append(f"{lab} & " + " & ".join(f2(ww.loc[v, 'eTaF1']) for v in V) + f" & {f2(ww.eTaF1.mean())} & {ww.fa_per_day.mean():.1f} & "
             f"{f2(cc.eTaF1.mean())} & {f2(cc.eTaF1.min())} & {f2(cc.FPR.mean())} & {cc.fa_per_day.max():.0f} " + r"\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
open(f"{T}/tab_shiftguard.tex", "w").write("\n".join(L))

# ---- ShiftGuard cross matrix figure ----
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f0efec", "#9ec5f4", "#2a78d6", "#104281"])
for ax, (m, lab) in zip(axes, [("PCA-raw", "PCA, raw"), ("PCA-diff", "PCA, differenced"), ("ShiftGuard-PCA", "ShiftGuard-PCA")]):
    g = E[(E.setting == "matrix") & (E.method == m)]
    e = g.pivot_table(index="train", columns="test", values="eTaF1").reindex(index=V, columns=V)
    fa = g.pivot_table(index="train", columns="test", values="fa_per_day").reindex(index=V, columns=V)
    ax.imshow(e.values, cmap=cmap, vmin=0, vmax=0.65)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{e.values[i, j]:.2f}\n{fa.values[i, j]:.0f}/d", ha="center", va="center", fontsize=6, color="white" if e.values[i, j] > 0.4 else INK)
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(V, fontsize=6.5); ax.set_yticklabels(V, fontsize=6.5)
    ax.set_title(lab, fontsize=8.5, color=INK); ax.set_xlabel("Test release"); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)
axes[0].set_ylabel("Training release")
fig.tight_layout(); fig.savefig(f"{F}/fig_shiftguard.pdf"); plt.close(fig)

# ---- attack shapes / localisation tables ----
ds = pd.read_csv(f"{R}/detection_by_shape.csv", index_col=0)
L = [r"\begin{table}[t]", r"\centering",
     r"\caption{Share of attacks detected within release, by attack shape (198 attacks, four releases). Shapes are assigned from the sensor data alone (Section~\ref{sec:shapes}).}",
     r"\label{tab:shapes}", r"\footnotesize", r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
     r"Shape & $n$ & PCA raw & PCA diff. & Level & ShiftGuard \\", r"\midrule"]
for s in ["abrupt", "gradual", "subtle"]:
    if s in ds.index:
        r = ds.loc[s]; L.append(f"{s.capitalize()} & {int(r['n'])} & {f2(r['PCA-raw'])} & {f2(r['PCA-diff'])} & {f2(r['Level'])} & {f2(r['ShiftGuard-PCA'])} " + r"\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(f"{T}/tab_shapes.tex", "w").write("\n".join(L))

lc = pd.read_csv(f"{R}/localisation.csv")
L = [r"\begin{table}[t]", r"\centering",
     r"\caption{Root-cause localisation on HAI 20.07 and 21.03: share of detected attacks for which the variable with the largest contribution at the first alarm belongs to an attacked process. Chance: always naming the process attacked most often (P1).}",
     r"\label{tab:loc}", r"\footnotesize", r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
     r"Method & Detected attacks & Correct process & Chance \\", r"\midrule"]
for m, lab in [("PCA-raw", "PCA, raw"), ("PCA-diff", "PCA, differenced"), ("Level", "Level channel")]:
    g = lc[lc.method == m]; ch = g.truth.str.contains("P1").mean()
    L.append(f"{lab} & {len(g)} & {f2(g.correct.mean())} & {f2(ch)} " + r"\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(f"{T}/tab_loc.tex", "w").write("\n".join(L))
print("v3 figures and tables written")
