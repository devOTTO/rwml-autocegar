#!/usr/bin/env python3
"""Generate cr x l1 sweep grids for P1-P4 (re-rank the gates under proper correction_rate
tuning, since the original screening fixed cr=0.1). One grid per proposal (504 lines each),
group proposalN_crtune.

    python experiments/proposals/runs/gen_p1234_crtune.py
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
FAMS = ["GECCO", "OPPORTUNITY", "CreditCard", "TAO", "PSM", "MSL", "SWaT"]
CR = [0.001, 0.01, 0.1]
L1 = [0.001, 0.01, 0.1, 1.0]
VARIANT = {1: "basic", 2: "mc5", 3: "full", 4: "gradnorm_wb"}  # P4 with docx write-back

files = pd.read_csv(os.path.join(ROOT, "rw/reproduction/summary_rw1_besthp.csv"))
files = files[files["family"].isin(FAMS)]["file"].tolist()
print(f"series: {len(files)}")

for n, v in VARIANT.items():
    lines = []
    for f in files:
        for cr in CR:
            for l1 in L1:
                lines.append(
                    f"--proposal {n} --variant {v} --dataset {f} --correction_rate {cr} "
                    f"--l1_weight {l1} --epochs 200 --lam 1.0 --extra lam_mode=auto_tr "
                    f"--wandb-group proposal{n}_crtune --tag crtune")
    with open(os.path.join(HERE, f"p{n}_crtune_grid.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"P{n} ({v}) crtune: {len(lines)} runs -> p{n}_crtune_grid.txt")
