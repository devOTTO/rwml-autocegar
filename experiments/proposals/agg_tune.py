#!/usr/bin/env python3
"""Aggregate the P5 tuning sweep into a like-for-like comparison vs RW-1 best-HP.

P5 best-HP = per-series max AUC-PR over the (l1_weight x lambda) grid, then the
collection mean (mirrors how RW-1 best_pr is a per-file best then family mean).
Also folds in the matched-config RW-1 control (l1=0.001) to show the SWaT config gap.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ORDER = ["GECCO", "OPPORTUNITY", "CreditCard", "TAO", "PSM", "MSL", "SWaT"]

# P5 tuning results
p5 = pd.read_csv(os.path.join(HERE, "results_p5_proposal5_tune.csv"))
p5 = p5[pd.to_numeric(p5["auc_pr"], errors="coerce").notna()].copy()
p5["auc_pr"] = p5["auc_pr"].astype(float)
# per-series best config (max AUC-PR over the 12 l1 x lambda cells)
idx = p5.groupby(["collection", "dataset"])["auc_pr"].idxmax()
best = p5.loc[idx, ["collection", "dataset", "auc_pr", "l1_weight", "lam", "extra"]]
p5_coll = best.groupby("collection")["auc_pr"].mean()

# RW-1 best-HP + DeepAnT reference
rw = pd.read_csv(os.path.join(ROOT, "rw/reproduction/summary_rw1_besthp.csv"))
rw_coll = rw.groupby("family")["best_pr"].mean()
da = pd.read_csv(os.path.join(ROOT, "deepant/reproduction/summary_per_dataset.csv"))
da_coll = da.groupby("family")["AUC-PR"].mean()

# matched-config RW-1 (l1=0.001) control, per collection per epochs
m = pd.read_csv(os.path.join(HERE, "results_rw1matched_proposal5_rw1matched.csv"))
m["auc_pr"] = pd.to_numeric(m["auc_pr"], errors="coerce")
m200 = m[m["epochs"] == 200].groupby("collection")["auc_pr"].mean()
m100 = m[m["epochs"] == 100].groupby("collection")["auc_pr"].mean()

print("=== P5 best-HP (per-series best over l1xlam, then collection mean) vs RW-1 best-HP ===")
print(f"{'collection':12} {'DeepAnT':>8} {'RW1-best':>9} {'P5-best':>8} {'delta':>8} {'RW1@l1.001-200':>15} {'RW1@l1.001-100':>15}")
wins = 0
for c in ORDER:
    da_v = float(da_coll.get(c, float('nan')))
    rw_v = float(rw_coll.get(c, float('nan')))
    p5_v = float(p5_coll.get(c, float('nan')))
    d = p5_v - rw_v
    if d > 0:
        wins += 1
    print(f"{c:12} {da_v:8.3f} {rw_v:9.3f} {p5_v:8.3f} {d:+8.3f} {float(m200.get(c,float('nan'))):15.3f} {float(m100.get(c,float('nan'))):15.3f}")
print(f"\nP5 best-HP beats RW-1 best-HP on {wins}/{len(ORDER)} collections (like-for-like).")

print("\n=== best config picked per series (l1_weight, lam) — does SWaT prefer heavy l1? ===")
for c in ORDER:
    sub = best[best["collection"] == c]
    combos = sub.apply(lambda r: f"l1={r['l1_weight']},lam={r['lam']}{'/auto' if str(r['extra'])=='lam_mode=auto_tr' else ''}", axis=1)
    print(f"  {c:12} n={len(sub):2}  best-configs: {list(combos)[:8]}")
