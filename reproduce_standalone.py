#!/usr/bin/env python3
"""Self-contained reproduction runner (no TSB-AD benchmark harness needed).

Runs RW / RW-1 / RW-1+CEGAR on TSB-AD-M CSV file(s) using ONLY this repo plus
its Python deps (requirements.txt), and reports AUC-PR / AUC-ROC. This is the
quick, portable path — for the exact paper-comparable numbers over all 199
datasets use the TSB-AD benchmark path (see REPRODUCE.md, "Path B").

Each TSB-AD-M CSV is `[T, n_features]` + a final `Label` column (0/1). The RW
family is unsupervised: it fits on the full series and the anomaly score is the
per-timestep correction magnitude.

Examples
--------
# one dataset, RW-1 at a fixed l1_weight
python reproduce_standalone.py --method rw1 --data_dir /path/TSB-AD-M \
       --file 001_Genesis_id_1_Sensor_tr_4055_1st_15538.csv --l1_weight 0.001

# RW-1 best-HP over the l1 grid for one dataset (max AUC-PR)
python reproduce_standalone.py --method rw1 --best_hp --data_dir /path/TSB-AD-M \
       --file 001_Genesis_id_1_Sensor_tr_4055_1st_15538.csv

# every dataset in a directory -> summary CSV
python reproduce_standalone.py --method rw --data_dir /path/TSB-AD-M --all --out rw_scores.csv
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # run from anywhere

from rw.cnn_uns import CNN_uns          # RW  (Algorithm 1)
from rw.cnn_rw import CNN_RW            # RW-1 (Algorithm 2)
from autocegar import CNN_RW_CEGAR      # RW-1 + CEGAR (extension)

L1_GRID = [1.0, 0.1, 0.01, 0.001]


def build(method, feats, epochs, l1_weight, cegar_kwargs):
    if method == "rw":
        return CNN_uns(window_size=50, feats=feats, epochs=epochs)
    if method == "rw1":
        return CNN_RW(window_size=50, feats=feats, epochs=epochs, l1_weight=l1_weight)
    if method == "cegar":
        return CNN_RW_CEGAR(window_size=50, feats=feats, epochs=epochs,
                            l1_weight=l1_weight, **cegar_kwargs)
    raise ValueError(f"unknown method {method}")


def run_one(path, method, epochs, l1_weight, best_hp, cegar_kwargs):
    df = pd.read_csv(path).dropna()
    data = df.iloc[:, :-1].values.astype(float)
    label = df["Label"].astype(int).to_numpy()
    feats = data.shape[1]

    def score_at(l1w):
        clf = build(method, feats, epochs, l1w, cegar_kwargs)
        s = clf.fit(data)
        return average_precision_score(label, s), roc_auc_score(label, s)

    if best_hp and method in ("rw1", "cegar"):
        results = {l1w: score_at(l1w) for l1w in L1_GRID}
        best_l1 = max(results, key=lambda k: results[k][0])  # max AUC-PR
        pr, roc = results[best_l1]
        return dict(file=os.path.basename(path), method=method, l1_weight=best_l1,
                    hp="best", **{"AUC-PR": pr, "AUC-ROC": roc})
    pr, roc = score_at(l1_weight)
    return dict(file=os.path.basename(path), method=method, l1_weight=l1_weight,
                hp="fixed", **{"AUC-PR": pr, "AUC-ROC": roc})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["rw", "rw1", "cegar"], required=True)
    ap.add_argument("--data_dir", required=True, help="dir with TSB-AD-M *.csv files")
    ap.add_argument("--file", help="single filename inside data_dir")
    ap.add_argument("--all", action="store_true", help="run every *.csv in data_dir")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--l1_weight", type=float, default=0.001)
    ap.add_argument("--best_hp", action="store_true", help="sweep the l1 grid, keep best AUC-PR")
    ap.add_argument("--lam", type=float, default=1.0, help="CEGAR gate strength")
    ap.add_argument("--warmup_epochs", type=int, default=0, help="CEGAR warm-up epochs")
    ap.add_argument("--out", help="write per-file results to this CSV")
    a = ap.parse_args()

    cegar_kwargs = dict(lam=a.lam, warmup_epochs=a.warmup_epochs,
                        correction_init="zero" if a.warmup_epochs > 0 else "neg_x")

    if a.all:
        files = sorted(glob.glob(os.path.join(a.data_dir, "*.csv")))
    elif a.file:
        files = [os.path.join(a.data_dir, a.file)]
    else:
        ap.error("give --file <name> or --all")

    rows = []
    for i, p in enumerate(files, 1):
        r = run_one(p, a.method, a.epochs, a.l1_weight, a.best_hp, cegar_kwargs)
        rows.append(r)
        print(f"[{i}/{len(files)}] {r['file']}: AUC-PR={r['AUC-PR']:.4f} "
              f"AUC-ROC={r['AUC-ROC']:.4f} (l1={r['l1_weight']})", flush=True)

    if rows:
        df = pd.DataFrame(rows)
        print(f"\n== {a.method} over {len(df)} dataset(s) ==")
        print(f"AUC-PR  mean={df['AUC-PR'].mean():.4f}  AUC-ROC mean={df['AUC-ROC'].mean():.4f}")
        if a.out:
            df.to_csv(a.out, index=False)
            print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
