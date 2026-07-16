#!/usr/bin/env python3
"""(b) P5 full-benchmark, per-file best-HP: sweep correction_rate x l1_weight on ALL ~199
TSB-AD-M series (take per-file best downstream). Matches the thesis best-HP methodology
so P5's overall mean is directly comparable to RW-1's paper number.

    python experiments/proposals/runs/gen_full200_besthp.py
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
CR = [0.001, 0.01, 0.1]
L1 = [0.001, 0.01, 0.1, 1.0]

files = pd.read_csv(os.path.join(ROOT, "rw/reproduction/summary_rw1_besthp.csv"))["file"].tolist()
lines = []
for f in files:
    for c in CR:
        for l1 in L1:
            lines.append(f"--proposal 5 --variant h5 --dataset {f} --correction_rate {c} "
                         f"--l1_weight {l1} --epochs 200 --lam 1.0 --extra lam_mode=auto_tr "
                         f"--wandb-group p5_full200_besthp --tag full200")
with open(os.path.join(HERE, "p5_full200_besthp_grid.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"(b) full200 best-HP: {len(files)} files x {len(CR)*len(L1)} = {len(lines)} runs "
      f"-> p5_full200_besthp_grid.txt")
