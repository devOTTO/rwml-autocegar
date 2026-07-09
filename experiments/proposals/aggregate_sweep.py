#!/usr/bin/env python
"""Per-config (stage) collection-mean table for a proposal's sweep.

Groups results_pN.csv by config (variant, tau, lam, tau_u) and shows the
per-collection mean AUC-PR for each config, plus whether any config beats the
reproduction RW-1 per-collection mean. Complements aggregate_collection.py
(which reports the default-config verdict only).

    python experiments/proposals/aggregate_sweep.py --proposal 2
"""
import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", type=int, default=2)
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(HERE, f"results_p{args.proposal}.csv"))
    df = df[pd.to_numeric(df["auc_pr"], errors="coerce").notna()].copy()
    df["auc_pr"] = df["auc_pr"].astype(float)
    df["tau_u"] = pd.to_numeric(df.get("tau_u"), errors="coerce").fillna(0.0)
    key = ["variant", "tau", "lam", "tau_u"]

    # per-config × collection mean AUC-PR
    piv = df.groupby(key + ["collection"])["auc_pr"].mean().unstack("collection").round(3)
    rw = pd.read_csv(os.path.join(ROOT, "rw/reproduction/summary_rw1_besthp.csv")) \
        .groupby("family")["best_pr"].mean().round(3)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(f"=== P{args.proposal} per-config collection-mean AUC-PR ===")
    print(piv.to_string())
    print("\nRW-1 (reproduction) per-collection AUC-PR:")
    print(rw.to_string())

    # does ANY config beat RW-1 on ANY collection?
    beats = []
    for cfg, row in piv.iterrows():
        for coll in piv.columns:
            v = row.get(coll)
            if pd.notna(v) and coll in rw.index and v > rw[coll]:
                beats.append((cfg, coll, round(v, 3), round(rw[coll], 3)))
    print(f"\nConfigs beating RW-1 (config, collection, P, RW-1): {beats if beats else 'NONE'}")


if __name__ == "__main__":
    main()
