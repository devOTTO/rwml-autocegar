#!/usr/bin/env python3
"""Generate the P5 hyperparameter-tuning sweep grids (like-for-like vs RW-1 best-HP).

Reads the exact per-collection series from the reproduction summary so the P5 tune
runs on the SAME files as the RW-1 best-HP baseline. Writes two grid files:
  - p5_tune_grid.txt        : Job A, group proposal5_tune
  - rw1_matched_grid.txt     : Job B, group proposal5_rw1matched (matched-config RW-1)

    python experiments/proposals/runs/gen_tune_grid.py
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))                 # .../experiments/proposals/runs
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))    # repo root
FAMS = ["GECCO", "OPPORTUNITY", "CreditCard", "TAO", "PSM", "MSL", "SWaT"]
L1 = [1.0, 0.1, 0.01, 0.001]                 # matches the RW-1 reproduction l1 sweep
# (lam, extra) : fixed lambda=1, fixed lambda=1.5, and the auto-lambda controller
LAMS = [("1.0", ""), ("1.5", ""), ("1.0", "--extra lam_mode=auto_tr")]
EPOCHS = 200                                 # match RW-1 best-HP (200ep) for like-for-like

df = pd.read_csv(os.path.join(ROOT, "rw/reproduction/summary_rw1_besthp.csv"))
files = df[df["family"].isin(FAMS)]["file"].tolist()
print(f"series: {len(files)} across {FAMS}")

# Job A — P5 tuning
a = []
for f in files:
    for l1 in L1:
        for lam, extra in LAMS:
            line = (f"--proposal 5 --variant h5 --dataset {f} --l1_weight {l1} "
                    f"--epochs {EPOCHS} --lam {lam} --wandb-group proposal5_tune --tag tune {extra}")
            a.append(line.rstrip())
with open(os.path.join(HERE, "p5_tune_grid.txt"), "w") as fh:
    fh.write("\n".join(a) + "\n")
print(f"Job A (p5_tune): {len(a)} runs -> p5_tune_grid.txt")

# Job B — matched-config RW-1 (l1=0.001) at 100 and 200 ep, to de-confound the deltas
b = []
for f in files:
    for ep in (100, 200):
        b.append(f"--proposal 5 --dataset {f} --baseline-only --l1_weight 0.001 "
                 f"--epochs {ep} --wandb-group proposal5_rw1matched --tag rw1matched")
with open(os.path.join(HERE, "rw1_matched_grid.txt"), "w") as fh:
    fh.write("\n".join(b) + "\n")
print(f"Job B (rw1_matched): {len(b)} runs -> rw1_matched_grid.txt")
