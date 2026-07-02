import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score, roc_auc_score
from autocegar import CNN_RW_CEGAR
from rw.cnn_rw import CNN_RW

DD="/ocean/projects/cis260190p/yhwang2/data/TSB-AD-M/TSB-AD-M/"
f="001_Genesis_id_1_Sensor_tr_4055_1st_15538.csv"
df=pd.read_csv(os.path.join(DD,f)).dropna()
data=df.iloc[:,:-1].values.astype(float); label=df["Label"].astype(int).to_numpy()
print(f"data {data.shape}, anomalies {label.sum()}", flush=True)

for name,cls,kw in [("RW-1 (plain)",CNN_RW,{}),
                    ("RW-1+CEGAR",CNN_RW_CEGAR,{"lam":1.0})]:
    clf=cls(window_size=50, feats=data.shape[1], epochs=50, batch_size=256,
            l1_weight=0.001, **kw)
    s=clf.fit(data)
    print(f"{name:14s} AUC-PR={average_precision_score(label,s):.4f} "
          f"AUC-ROC={roc_auc_score(label,s):.4f}", flush=True)
print("SMOKE OK", flush=True)
