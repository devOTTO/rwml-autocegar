#!/usr/bin/env python3
"""Standalone P5-tuning aggregation for a LOCAL machine (no repo structure needed).

Put these 4 CSVs in the SAME folder as this script, then run `python agg_local.py`:
  - results_p5_proposal5_tune.csv              (P5 tuning sweep, 504 rows)
  - results_rw1matched_proposal5_rw1matched.csv (matched-config RW-1 control, 84 rows)
  - summary_rw1_besthp.csv                      (RW-1 best-HP reference, from rw/reproduction/)
  - summary_per_dataset.csv                     (DeepAnT reference, from deepant/reproduction/)

Only needs pandas:  pip install pandas
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
def P(name): return os.path.join(HERE, name)
ORDER = ["GECCO", "OPPORTUNITY", "CreditCard", "TAO", "PSM", "MSL", "SWaT"]

# P5 tuning: per-series best AUC-PR over the l1 x lambda grid, then collection mean
p5 = pd.read_csv(P("results_p5_proposal5_tune.csv"))
p5 = p5[pd.to_numeric(p5["auc_pr"], errors="coerce").notna()].copy()
p5["auc_pr"] = p5["auc_pr"].astype(float)
best = p5.loc[p5.groupby(["collection", "dataset"])["auc_pr"].idxmax(),
              ["collection", "dataset", "auc_pr", "l1_weight", "lam", "extra"]]
p5_coll = best.groupby("collection")["auc_pr"].mean()

rw = pd.read_csv(P("summary_rw1_besthp.csv")).groupby("family")["best_pr"].mean()
da = pd.read_csv(P("summary_per_dataset.csv")).groupby("family")["AUC-PR"].mean()

m = pd.read_csv(P("results_rw1matched_proposal5_rw1matched.csv"))
m["auc_pr"] = pd.to_numeric(m["auc_pr"], errors="coerce")
m200 = m[m["epochs"] == 200].groupby("collection")["auc_pr"].mean()
m100 = m[m["epochs"] == 100].groupby("collection")["auc_pr"].mean()

print(f"{'collection':12} {'DeepAnT':>8} {'RW1-best':>9} {'P5-best':>8} {'delta':>8} "
      f"{'RW1@l.001/200':>13} {'RW1@l.001/100':>13}")
wins = 0
for c in ORDER:
    p5v = float(p5_coll.get(c, float('nan'))); rwv = float(rw.get(c, float('nan')))
    d = p5v - rwv
    wins += int(d > 0)
    print(f"{c:12} {float(da.get(c,float('nan'))):8.3f} {rwv:9.3f} {p5v:8.3f} {d:+8.3f} "
          f"{float(m200.get(c,float('nan'))):13.3f} {float(m100.get(c,float('nan'))):13.3f}")
print(f"\nP5 best-HP beats RW-1 best-HP on {wins}/{len(ORDER)} collections (like-for-like).")
print("\nbest (l1,lam) per series:")
for c in ORDER:
    sub = best[best["collection"] == c]
    combos = [f"{r.dataset}:l1={r.l1_weight},lam={r.lam}{'+auto' if str(r.extra)=='lam_mode=auto_tr' else ''}"
              for r in sub.itertuples()]
    print(f"  {c:12} {combos}")
