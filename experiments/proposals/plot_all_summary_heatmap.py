#!/usr/bin/env python
"""All-proposals overview for the ALL-summary run: a heatmap of Δ AUC-PR
(proposal fixed-λ − RW-1 best-HP) over the 5 proposals × 7 collections. Reads
results_p{N}.csv + the reproduction reference. Clean, no label overlap.

    python experiments/proposals/plot_all_summary_heatmap.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

plt.rcParams.update({"font.size": 13, "axes.titlesize": 15})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIG = os.path.join(HERE, "figures")
VARIANT = {1: "basic", 2: "mc5", 3: "full", 4: "gradnorm", 5: "h5"}
NAME = {1: "P1 residual", 2: "P2 uncertainty", 3: "P3 consistency",
        4: "P4 dual-grad", 5: "P5 persistence"}
COLS = ["OPPORTUNITY", "GECCO", "CreditCard", "TAO", "PSM", "MSL", "SWaT"]


def main():
    rw = (pd.read_csv(os.path.join(ROOT, "rw/reproduction/summary_rw1_besthp.csv"))
          .groupby("family")["best_pr"].mean())
    delta = np.full((5, len(COLS)), np.nan)
    for i, n in enumerate(range(1, 6)):
        df = pd.read_csv(os.path.join(HERE, f"results_p{n}.csv"))
        df = df[pd.to_numeric(df["auc_pr"], errors="coerce").notna()].copy()
        df["auc_pr"] = df["auc_pr"].astype(float)
        fx = df[(pd.to_numeric(df["tau"], errors="coerce") == 2.0)
                & (pd.to_numeric(df["lam"], errors="coerce") == 1.0)
                & (df["variant"] == VARIANT[n]) & (df["extra"].fillna("") == "")]
        cm = fx.groupby("collection")["auc_pr"].mean()
        for j, c in enumerate(COLS):
            if c in cm.index and c in rw.index:
                delta[i, j] = float(cm[c]) - float(rw[c])

    fig, ax = plt.subplots(figsize=(13, 6))
    # color capped at ±0.05 so small real wins/losses are visible; big losses saturate
    norm = TwoSlopeNorm(vmin=-0.05, vcenter=0.0, vmax=0.05)
    im = ax.imshow(np.clip(delta, -0.05, 0.05), cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(range(len(COLS))); ax.set_xticklabels(COLS, fontsize=12)
    ax.set_yticks(range(5)); ax.set_yticklabels([NAME[n] for n in range(1, 6)], fontsize=12)
    for i in range(5):
        for j in range(len(COLS)):
            v = delta[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=11,
                        color="black", fontweight="bold" if v > 0 else "normal")
    ax.set_title("Δ AUC-PR vs tuned RW-1 (proposal fixed-λ − RW-1 best-HP)\n"
                 "green = beats RW-1 · columns: 3 screening + 4 shape-extension collections")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Δ AUC-PR (capped ±0.05)")
    fig.tight_layout()
    out = os.path.join(FIG, "all_proposals_delta_heatmap.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
