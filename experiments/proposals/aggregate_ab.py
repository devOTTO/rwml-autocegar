#!/usr/bin/env python
"""A-vs-B config comparison for the P1-P5 re-run.

cfgA = zero-init + warmup 10 (current);  cfgB = neg_x-init + warmup 0 (thesis-faithful).
Reads results_p{1..5}.csv, splits each by config (cfgB rows carry
extra='correction_init=neg_x' + warmup 0; cfgA rows have extra='' + warmup 10), and
prints per-collection mean AUC-PR for both configs next to the reproduction RW-1.

    python experiments/proposals/aggregate_ab.py
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RW = {"OPPORTUNITY": 0.138, "GECCO": 0.639, "CreditCard": 0.111}
NAMES = {1: "P1 residual", 2: "P2 uncertainty", 3: "P3 consistency",
         4: "P4 dual-grad", 5: "P5 persistence"}


def main():
    for n in range(1, 6):
        csv = os.path.join(HERE, f"results_p{n}.csv")
        if not os.path.exists(csv):
            print(f"[P{n}] no results yet"); continue
        d = pd.read_csv(csv)
        d = d[pd.to_numeric(d["auc_pr"], errors="coerce").notna()].copy()
        d["auc_pr"] = d["auc_pr"].astype(float)
        d["extra"] = d["extra"].fillna("")
        d["warmup"] = pd.to_numeric(d["warmup"], errors="coerce")
        A = d[(d["extra"] == "") & (d["warmup"] == 10)]
        B = d[(d["extra"].str.contains("neg_x")) & (d["warmup"] == 0)]
        print(f"\n=== {NAMES[n]} ===")
        print(f"{'collection':12s} {'cfgA(zero+wu)':>14} {'cfgB(negx+nowu)':>16} {'RW-1':>7}  A/B beat?")
        for coll in ["OPPORTUNITY", "GECCO", "CreditCard"]:
            a = A[A.collection == coll]["auc_pr"].mean()
            b = B[B.collection == coll]["auc_pr"].mean()
            rw = RW[coll]
            aw = "Y" if a > rw else "n"
            bw = "Y" if b > rw else "n"
            print(f"{coll:12s} {a:>14.3f} {b:>16.3f} {rw:>7.3f}  A:{aw} B:{bw}")


if __name__ == "__main__":
    main()
