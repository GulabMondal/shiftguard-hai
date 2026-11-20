"""Generate LaTeX tables from result CSVs (run after figs.py)."""
import os, glob, numpy as np, pandas as pd
R = "results"; T = "paper/tables"; os.makedirs(T, exist_ok=True)
ORDER = ["Envelope", "PCA", "IForest", "kNN", "OC-SVM", "MLP-AE", "LSTM-AE", "Transformer", "USAD", "TranAD", "GDN"]
VERS = ["21.03", "22.04", "23.05"]
NAME = {"IForest": "Isolation Forest", "kNN": "$k$NN", "OC-SVM": "One-class SVM"}
f2 = lambda x: "--" if pd.isna(x) else f"{x:.2f}"

t1 = pd.read_csv(f"{R}/T1_inflation_agg.csv", dtype={"version": str})
# best deployable eTaF1 per version in bold
best = t1.groupby("version").deploy_eTaF1.max()
L = [r"\begin{table*}[t]", r"\centering",
     r"\caption{Detection quality on each HAI release (full feature set). PA-F1 and F1 (oracle) use thresholds tuned on the test labels; all other columns use the deployable train-only threshold. ER: event recall; FA/d: false-alarm intervals per day of normal operation; Delay: median seconds to first alarm; $n$: random seeds averaged. Best deployable eTaF1 per release in bold.}",
     r"\label{tab:inflation}", r"\footnotesize", r"\setlength{\tabcolsep}{2.6pt}",
     r"\begin{tabular}{@{}lc" + "|cccccc" * 3 + r"@{}}", r"\toprule",
     r" & & \multicolumn{6}{c|}{HAI 21.03} & \multicolumn{6}{c|}{HAI 22.04} & \multicolumn{6}{c}{HAI 23.05} \\",
     r"Detector & $n$ " + "& PA-F1 & F1$^{o}$ & eTaF1 & ER & FA/d & Delay " * 3 + r"\\", r"\midrule"]
for m in ORDER:
    g = t1[t1.model == m].set_index("version")
    if g.empty: continue
    cells = []
    for v in VERS:
        r = g.loc[v]
        e = f2(r.deploy_eTaF1)
        if abs(r.deploy_eTaF1 - best[v]) < 1e-9: e = r"\textbf{" + e + "}"
        cells += [f2(r.oracle_PA_F1), f2(r.oracle_F1), e, f2(r.deploy_event_recall), f"{r.fa_per_day:.1f}", f"{r.delay_s:.0f}"]
    if m in ("MLP-AE", "TranAD"): L.append(r"\midrule")
    L.append(f"{NAME.get(m, m)} & {int(g.n_seeds.max())} & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
open(f"{T}/tab_inflation.tex", "w").write("\n".join(L))

# Cross-release table
c = pd.read_csv(f"{R}/T2_cross_agg.csv", dtype={"train": str, "test": str})
pairs = [("21.03", "22.04"), ("21.03", "23.05"), ("22.04", "21.03"), ("22.04", "23.05"), ("23.05", "21.03"), ("23.05", "22.04")]
L = [r"\begin{table*}[t]", r"\centering",
     r"\caption{Cross-release eTaF1 with train-only thresholds (common feature set). ``Within'' is the mean of the three within-release results on the same feature set. FPR: mean share of normal seconds in alarm over the six cross-release pairs.}",
     r"\label{tab:cross}", r"\footnotesize",
     r"\begin{tabular}{@{}lc|cccccc|c@{}}", r"\toprule",
     r"Detector & Within & 21$\to$22 & 21$\to$23 & 22$\to$21 & 22$\to$23 & 23$\to$21 & 23$\to$22 & Cross FPR \\", r"\midrule"]
for m in ORDER:
    g = c[c.model == m].set_index(["train", "test"])
    if g.empty: continue
    within = np.mean([g.loc[(v, v), "eTaF1"] for v in VERS])
    vals = [f2(g.loc[p, "eTaF1"]) for p in pairs]
    fpr = np.mean([g.loc[p, "FPR"] for p in pairs])
    L.append(f"{NAME.get(m, m)} & {f2(within)} & " + " & ".join(vals) + f" & {f2(fpr)} " + r"\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
open(f"{T}/tab_cross.tex", "w").write("\n".join(L))

# Adaptation table
ad = pd.read_csv(f"{R}/adapt_all.csv", dtype={"train": str, "test": str})
cc = c.set_index(["model", "train", "test"])
L = [r"\begin{table}[t]", r"\centering",
     r"\caption{Recovery after a release change: mean eTaF1 over the six cross-release pairs, using the best strategy per pair. ``0 h'' is the unadapted detector; ``Target'' is the detector trained on the full target-release data.}",
     r"\label{tab:adapt}", r"\footnotesize",
     r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
     r"Detector & 0 h & 1 h & 6 h & 24 h & Target \\", r"\midrule"]
for m in [x for x in ORDER if x in set(ad.model)]:
    g = ad[ad.model == m]
    best_k = g.groupby(["train", "test", "k_hours"]).eTaF1.max().groupby("k_hours").mean()
    zero = np.mean([cc.loc[(m,) + p, "eTaF1"] for p in pairs])
    target = np.mean([cc.loc[(m, p[1], p[1]), "eTaF1"] for p in pairs])
    L.append(f"{NAME.get(m, m)} & {f2(zero)} & {f2(best_k.get(1))} & {f2(best_k.get(6))} & {f2(best_k.get(24))} & {f2(target)} " + r"\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(f"{T}/tab_adapt.tex", "w").write("\n".join(L))

# Per-process table
pp_path = f"{R}/perprocess_2103.csv"
if os.path.exists(pp_path):
    pp = pd.read_csv(pp_path)
    tab = pp.groupby(["process", "model"]).detected.mean().unstack("model")
    cnt = pp[pp.model == pp.model.iloc[0]].groupby("process").size()
    ms = [m for m in ORDER if m in tab.columns]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Share of HAI 21.03 attacks detected (train-only threshold, full feature set), by targeted process. P1: boiler; P2: turbine; P3: water treatment.}",
         r"\label{tab:perprocess}", r"\footnotesize",
         r"\begin{tabular}{@{}lc" + "c" * len(ms) + r"@{}}", r"\toprule",
         "Process & Attacks & " + " & ".join(NAME.get(m, m) for m in ms) + r" \\", r"\midrule"]
    for p in tab.index:
        L.append(f"{p.replace('+', '+')} & {cnt[p]} & " + " & ".join(f2(tab.loc[p, m]) for m in ms) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(f"{T}/tab_perprocess.tex", "w").write("\n".join(L))
print("tables written")
