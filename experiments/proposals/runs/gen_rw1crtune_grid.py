#!/usr/bin/env python3
"""Generate the matched RW-1 correction_rate x l1_weight sweep (baseline-only), so the
P5 crtune "6/7" becomes a fair like-for-like — RW-1 gets the SAME cr x l1 grid P5 got.

    python experiments/proposals/runs/gen_rw1crtune_grid.py
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
FAMS = ["GECCO", "OPPORTUNITY", "CreditCard", "TAO", "PSM", "MSL", "SWaT"]
CR = [0.001, 0.01, 0.1]
L1 = [0.001, 0.01, 0.1, 1.0]
EPOCHS = 200

files = pd.read_csv(os.path.join(ROOT, "rw/reproduction/summary_rw1_besthp.csv"))
files = files[files["family"].isin(FAMS)]["file"].tolist()
print(f"series: {len(files)}")

lines = []
for f in files:
    for cr in CR:
        for l1 in L1:
            lines.append(
                f"--proposal 5 --dataset {f} --baseline-only --correction_rate {cr} "
                f"--l1_weight {l1} --epochs {EPOCHS} --wandb-group rw1_crtune --tag crtune")
with open(os.path.join(HERE, "rw1_crtune_grid.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"RW-1 crtune: {len(lines)} runs -> rw1_crtune_grid.txt")
