#!/usr/bin/env python3
"""Generate the P5 correction_rate x l1_weight sweep grid (thesis §6.3.2 axis).

correction_rate = the RMSprop step size for the correction tensor (Baldo tuned this;
we had it fixed at 0.1). Crossed with l1_weight (which we already found matters), at
auto-lambda / 200ep, on the same 42 series as the RW-1 best-HP baseline.

    python experiments/proposals/runs/gen_crtune_grid.py
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
FAMS = ["GECCO", "OPPORTUNITY", "CreditCard", "TAO", "PSM", "MSL", "SWaT"]
CR = [0.001, 0.01, 0.1]                       # thesis correction-rate sweep
L1 = [0.001, 0.01, 0.1, 1.0]                  # crossed with l1 (already known to matter)
EPOCHS = 200

files = pd.read_csv(os.path.join(ROOT, "rw/reproduction/summary_rw1_besthp.csv"))
files = files[files["family"].isin(FAMS)]["file"].tolist()
print(f"series: {len(files)}")

lines = []
for f in files:
    for cr in CR:
        for l1 in L1:
            lines.append(
                f"--proposal 5 --variant h5 --dataset {f} --correction_rate {cr} "
                f"--l1_weight {l1} --epochs {EPOCHS} --lam 1.0 --extra lam_mode=auto_tr "
                f"--wandb-group proposal5_crtune --tag crtune")
with open(os.path.join(HERE, "p5_crtune_grid.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"P5 crtune: {len(lines)} runs -> p5_crtune_grid.txt")
