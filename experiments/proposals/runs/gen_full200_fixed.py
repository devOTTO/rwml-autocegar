#!/usr/bin/env python3
"""(a) P5 full-benchmark with ONE fixed recipe: the (cr, l1) that maximizes mean AUC-PR
over the 7-collection crtune sweep, applied to ALL ~199 TSB-AD-M series.

    python experiments/proposals/runs/gen_full200_fixed.py
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

# pick the mean-best fixed (cr, l1) from the crtune sweep (not oracle per-series)
cr = pd.read_csv(os.path.join(ROOT, "experiments/proposals/results_p5_proposal5_crtune.csv"))
cr = cr[pd.to_numeric(cr["auc_pr"], errors="coerce").notna()].copy()
cr["auc_pr"] = cr["auc_pr"].astype(float)
means = cr.groupby(["correction_rate", "l1_weight"])["auc_pr"].mean()
best_cr, best_l1 = means.idxmax()
print(f"mean-best fixed recipe: correction_rate={best_cr}, l1_weight={best_l1} "
      f"(mean AUC-PR {means.max():.3f} over the 42 crtune series)")

files = pd.read_csv(os.path.join(ROOT, "rw/reproduction/summary_rw1_besthp.csv"))["file"].tolist()
lines = [f"--proposal 5 --variant h5 --dataset {f} --correction_rate {best_cr} "
         f"--l1_weight {best_l1} --epochs 200 --lam 1.0 --extra lam_mode=auto_tr "
         f"--wandb-group p5_full200_fixed --tag full200" for f in files]
with open(os.path.join(HERE, "p5_full200_fixed_grid.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"(a) full200 fixed: {len(lines)} runs -> p5_full200_fixed_grid.txt")
