"""v4 tables: fair comparison, four-release ShiftGuard table with RevIN rows, robustness summary, SKAB, theory-with-limited-data."""
import numpy as np, pandas as pd
R3 = "results/v3"; R4 = "results/v4"; T = "paper/tables"
f2 = lambda x: "--" if pd.isna(x) else f"{x:.2f}"
V4 = ["20.07", "21.03", "22.04", "23.05"]

# --- fair comparison (3 releases, within) ---
F = pd.read_csv(f"{R4}/fair_within.csv", dtype={"release": str})
g = F.groupby(["sm", "method", "release"])[["eTaF1", "fa_per_day"]].mean().reset_index()
cost = {"PCA-raw": "0.5", "PCA-diff": "0.5", "ShiftGuard-PCA": "6", "Transformer-raw": "90", "Transformer-diff": "90",
        "ShiftGuard-Transformer": "96", "TranAD-raw": "280"}
seeds = F.groupby("method").seed.nunique()
lab = {"PCA-raw": "PCA, raw", "PCA-diff": "PCA, differenced", "ShiftGuard-PCA": "\\textbf{ShiftGuard-PCA}",
       "Transformer-raw": "Transformer, raw", "Transformer-diff": "Transformer, differenced",
       "ShiftGuard-Transformer": "\\textbf{ShiftGuard-Transformer}", "TranAD-raw": "TranAD, raw"}
L = [r"\begin{table*}[t]", r"\centering",
     r"\caption{Within-release comparison of ShiftGuard with the strongest benchmark detectors on HAI 21.03, 22.04 and 23.05, all re-thresholded from saved scores under identical smoothing (train-only thresholds). eTaF1 per release at 10-s smoothing, mean eTaF1 and false alarms per day (FA/d) at 10 s and 60 s, seeds averaged, and inference cost per sample on a two-core CPU.}",
     r"\label{tab:fair}", r"\footnotesize", r"\begin{tabular}{@{}lc|ccc|cc|cc|r@{}}", r"\toprule",
     r" & & \multicolumn{3}{c|}{eTaF1, 10-s smoothing} & \multicolumn{2}{c|}{10 s} & \multicolumn{2}{c|}{60 s} & Cost \\",
     r"Method & Seeds & 21.03 & 22.04 & 23.05 & eTaF1 & FA/d & eTaF1 & FA/d & ($\mu$s) \\", r"\midrule"]
for m in lab:
    a = g[(g.sm == 10) & (g.method == m)].set_index("release"); b = g[(g.sm == 60) & (g.method == m)]
    L.append(f"{lab[m]} & {seeds[m]} & " + " & ".join(f2(a.loc[v, 'eTaF1']) for v in ["21.03", "22.04", "23.05"]) +
             f" & {f2(a.eTaF1.mean())} & {a.fa_per_day.mean():.1f} & {f2(b.eTaF1.mean())} & {b.fa_per_day.mean():.1f} & {cost[m]} " + r"\\")
    if m in ("ShiftGuard-PCA", "ShiftGuard-Transformer"): L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
open(f"{T}/tab_fair.tex", "w").write("\n".join(L))

# --- four-release table with RevIN rows ---
E = pd.read_csv(f"{R3}/eval3_sm60_s0.csv", dtype={"train": str, "test": str})
Rv = pd.read_csv(f"{R4}/revin_eval.csv", dtype={"train": str, "test": str}); Rv["cross"] = Rv.train != Rv.test
E = pd.concat([E, Rv], ignore_index=True)
order = [("PCA-raw", "PCA, raw inputs"), ("PCA-RevIN60", "PCA, instance-normalized (60 s)"), ("PCA-RevIN600", "PCA, instance-normalized (600 s)"),
         ("PCA-diff", "PCA, differenced inputs"), ("Level", "Level channel only"),
         ("NDiffLevel-PCA", "Normalized diff.\\ + level (rejected)"), ("ShiftGuard-PCA", "\\textbf{ShiftGuard-PCA}")]
w = E[E.setting == "within"]; c = E[(E.setting == "matrix") & E.cross]
L = [r"\begin{table*}[t]", r"\centering",
     r"\caption{ShiftGuard and its alternatives on four HAI releases (PCA detector, 60-s smoothing, thresholds from normal validation data only). Within-release: full feature set, eTaF1 per release and mean false alarms per day (FA/d). Cross-release: the 12 ordered pairs on the 38 variables shared by all releases; mean and worst-pair eTaF1, mean false-positive rate, and worst-pair FA/d. Instance normalization follows RevIN~\cite{kim2022revin}: each variable is z-scored over the preceding window.}",
     r"\label{tab:shiftguard}", r"\footnotesize", r"\begin{tabular}{@{}l|cccc|cc|cccc@{}}", r"\toprule",
     r" & \multicolumn{4}{c|}{Within-release eTaF1} & \multicolumn{2}{c|}{Within mean} & \multicolumn{4}{c}{Cross-release (12 pairs)} \\",
     r"Method & 20.07 & 21.03 & 22.04 & 23.05 & eTaF1 & FA/d & eTaF1 & worst & FPR & worst FA/d \\", r"\midrule"]
for m, lab_ in order:
    ww = w[w.method == m].set_index("train"); cc = c[c.method == m]
    L.append(f"{lab_} & " + " & ".join(f2(ww.loc[v, 'eTaF1']) for v in V4) + f" & {f2(ww.eTaF1.mean())} & {ww.fa_per_day.mean():.1f} & "
             f"{f2(cc.eTaF1.mean())} & {f2(cc.eTaF1.min())} & {f2(cc.FPR.mean())} & {cc.fa_per_day.max():.0f} " + r"\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
open(f"{T}/tab_shiftguard.tex", "w").write("\n".join(L))

# --- robustness summary across settings ---
X = pd.read_csv(f"{R4}/fair_cross59.csv", dtype={"src": str, "tgt": str}); X = X[(X.sm == 60) & (X.src != X.tgt)]
S = pd.read_csv(f"{R4}/skab_sm60.csv").set_index("method")
meth = [("PCA-raw", "PCA, raw"), ("PCA-RevIN", "PCA, instance-norm."), ("PCA-diff", "PCA, differenced"), ("Level", "Level only"), ("ShiftGuard-PCA", "\\textbf{ShiftGuard-PCA}")]
rows = {}
for m, _ in meth:
    rv = "PCA-RevIN600" if m == "PCA-RevIN" else m
    a = w[w.method == rv].eTaF1.mean(); b = c[c.method == rv].eTaF1.mean()
    x = X[X.method == m].eTaF1.mean() if m in set(X.method) else np.nan
    s = S.loc[m, "eTaF1"] if m in S.index else np.nan
    rows[m] = [a, b, x, s]
L = [r"\begin{table}[t]", r"\centering",
     r"\caption{Mean eTaF1 in four evaluation settings (train-only thresholds, 60-s smoothing). HAI-4: four releases (within: mean over releases; cross: 12 pairs, 38 variables). HAI-3 cross: six pairs among 21.03--23.05 on 59 variables. SKAB: fitted on the anomaly-free recording, applied to 34 experiments. Instance normalization uses a 600-s window on HAI and 60 s on SKAB. The last column is the worst of the four.}",
     r"\label{tab:robust}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
     r" & \multicolumn{2}{c}{HAI-4} & HAI-3 & & \\", r"Method & within & cross & cross & SKAB & worst \\", r"\midrule"]
for m, lab_ in meth:
    v = rows[m]; L.append(f"{lab_} & " + " & ".join(f2(x) for x in v) + f" & {f2(np.nanmin(v))} " + r"\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(f"{T}/tab_robust.tex", "w").write("\n".join(L))
pd.DataFrame(rows, index=["HAI4_within", "HAI4_cross", "HAI3_cross", "SKAB"]).T.to_csv(f"{R4}/robust_summary.csv")

# --- SKAB table ---
order = [("PCA-raw (per-file fit)", "PCA, raw, fitted per file (no shift)"), ("PCA-raw", "PCA, raw"), ("PCA-RevIN", "PCA, instance-normalized"),
         ("PCA-diff", "PCA, differenced"), ("Level", "Level channel only"), ("ShiftGuard-PCA", "\\textbf{ShiftGuard-PCA}")]
L = [r"\begin{table}[t]", r"\centering",
     r"\caption{External check on SKAB (34 experiments, 60-s smoothing). All detectors except the first are fitted on the anomaly-free recording and applied unchanged; thresholds from its last 10\%. The first row follows the SKAB convention of fitting on the first 400 normal rows of each file.}",
     r"\label{tab:skab}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
     r"Method & eTaF1 & F1 & FPR & Event recall \\", r"\midrule"]
for m, lab_ in order:
    r = S.loc[m]; L.append(f"{lab_} & {f2(r.eTaF1)} & {f2(r.F1)} & {f2(r.FPR)} & {f2(r.event_recall)} " + r"\\")
    if m == "PCA-raw (per-file fit)": L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(f"{T}/tab_skab.tex", "w").write("\n".join(L))
print(open(f"{T}/tab_robust.tex").read()); print(open(f"{T}/tab_fair.tex").read())
