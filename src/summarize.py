import glob, pandas as pd, numpy as np
R = "results"
df = pd.concat([pd.read_csv(f, dtype={"train": str, "test": str}) for f in glob.glob(f"{R}/main_*.csv")])
order = ["Envelope", "PCA", "IForest", "kNN", "OC-SVM", "MLP-AE", "LSTM-AE", "Transformer", "USAD"]
order = [m for m in order if m in set(df.model)]

# Table 1: the inflation gap on the standard (within-version, full features) setting
w = df[df.setting == "within-full"]
t1 = []
for m in order:
    for v in ["21.03", "22.04", "23.05"]:
        g = w[(w.model == m) & (w.test == v)].set_index("policy")
        if g.empty: continue
        t1.append(dict(model=m, version=v,
                       oracle_PA_F1=g.loc["oracle-PA", "PA_F1"],
                       oracle_F1=g.loc["oracle-F1", "F1"],
                       deploy_F1=g.loc["train-only", "F1"],
                       deploy_eTaF1=g.loc["train-only", "eTaF1"],
                       deploy_event_recall=g.loc["train-only", "event_recall"],
                       deploy_FPR=g.loc["train-only", "FPR"],
                       delay_s=g.loc["train-only", "delay_med"]))
t1 = pd.DataFrame(t1).round(3)
t1.to_csv(f"{R}/table1_inflation.csv", index=False)

# Table 2: cross-version eTaF1 matrix (common features, train-only threshold)
c = df[(df.setting.isin(["within-common", "cross-common"])) & (df.policy == "train-only")]
t2 = c.pivot_table(index=["model", "train"], columns="test", values="eTaF1").reindex(order, level=0).round(3)
t2.to_csv(f"{R}/table2_cross_eTaF1.csv")
t2f = c.pivot_table(index=["model", "train"], columns="test", values="FPR").reindex(order, level=0).round(3)
t2f.to_csv(f"{R}/table2_cross_FPR.csv")

# Compute cost
cost = df[df.setting == "within-full"].groupby("model")[["fit_s", "infer_us_per_row"]].mean().reindex(order).round(2)
cost.to_csv(f"{R}/table_cost.csv")

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(t1.to_string(index=False)); print(); print(t2); print(); print(t2f); print(); print(cost)
