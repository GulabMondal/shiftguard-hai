import pandas as pd, numpy as np, glob, os, pickle
D="data"; C="cache"
def _read(p): return pd.read_csv(p)
def load_version(v):
    if v=="21.03":
        tr=[_read(f"{D}/21.03/train{i}.csv.gz") for i in (1,2,3)]
        te=[_read(f"{D}/21.03/test{i}.csv.gz") for i in (1,2,3,4,5)]
        lab=lambda d:d["attack"].values
        drop=["time","attack","attack_P1","attack_P2","attack_P3"]
    elif v=="22.04":
        tr=[_read(f"{D}/22.04/train{i}.csv") for i in range(1,7)]
        te=[_read(f"{D}/22.04/test{i}.csv") for i in range(1,5)]
        lab=lambda d:d["Attack"].values; drop=["timestamp","Attack"]
    else:
        tr=[_read(f"{D}/23.05/hai-train{i}.csv") for i in range(1,5)]
        te=[]
        for i in (1,2):
            d=_read(f"{D}/23.05/hai-test{i}.csv"); l=_read(f"{D}/23.05/label-test{i}.csv")
            assert len(d)==len(l); td=pd.to_datetime(d.iloc[:,0]); tl=pd.to_datetime(l.timestamp, format='mixed'); assert ((td.dt.floor('min')==tl.dt.floor('min')).mean())>0.98, 'misaligned'
            d["label"]=l.label.values; te.append(d)
        lab=lambda d:d["label"].values; drop=["timestamp","label"]
    tcol=tr[0].columns[0]
    out={"train":[(d.drop(columns=[c for c in drop+[tcol] if c in d.columns]),lab(d) if (v!="23.05") else np.zeros(len(d),int)) for d in tr],
         "test":[(d.drop(columns=[c for c in drop+[tcol] if c in d.columns]),lab(d)) for d in te]}
    return out
if __name__=="__main__":
    allv={}
    for v in ("21.03","22.04","23.05"):
        o=load_version(v); allv[v]=o
        ntr=sum(len(x) for x,_ in o["train"]); nte=sum(len(x) for x,_ in o["test"])
        atr=sum(int(y.sum()) for _,y in o["train"]); ate=sum(int(y.sum()) for _,y in o["test"])
        # count attack segments
        seg=sum(int(((np.diff(np.r_[0,y,0]))==1).sum()) for _,y in o["test"])
        print(v,"cols",o["train"][0][0].shape[1],"train rows",ntr,"attack-in-train",atr,"test rows",nte,"attack rows",ate,"segments",seg)
    cols=[set(allv[v]["train"][0][0].columns) for v in allv]
    common=sorted(set.intersection(*cols)); print("common features",len(common))
    print("only 21.03:",sorted(cols[0]-set(common))); print("only 22.04:",sorted(cols[1]-set(common))[:20]); print("only 23.05:",sorted(cols[2]-set(common))[:30])
    pickle.dump(allv,open(f"{C}/all.pkl","wb"))
