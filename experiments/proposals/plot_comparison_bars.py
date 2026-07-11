#!/usr/bin/env python
"""Per-proposal AUC-PR comparison bar chart: DeepAnT / RW-1 / P{N} fixed / P{N} auto
across all tested collections. Reads results_p{N}.csv (local) + the reproduction
reference means — no GPU, no retraining.

    python experiments/proposals/plot_comparison_bars.py --proposal 5
    python experiments/proposals/plot_comparison_bars.py --all
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
    "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIG = os.path.join(HERE, "figures")
VARIANT = {1: "basic", 2: "mc5", 3: "full", 4: "gradnorm", 5: "h5"}
NAME = {1: "P1 residual", 2: "P2 uncertainty", 3: "P3 consistency",
        4: "P4 dual-grad", 5: "P5 persistence"}
# display order: verdict set then shape extension
ORDER = ["OPPORTUNITY", "GECCO", "CreditCard", "TAO", "PSM", "MSL", "SWaT"]


def ref_means():
    rw = pd.read_csv(os.path.join(ROOT, "rw/reproduction/summary_rw1_besthp.csv"))
    rw_g = rw.groupby("family").agg(rw=("best_pr", "mean"))
    da = pd.read_csv(os.path.join(ROOT, "deepant/reproduction/summary_per_dataset.csv"))
    da_g = da.groupby("family").agg(da=("AUC-PR", "mean"))
    return rw_g, da_g


def proposal_means(n):
    df = pd.read_csv(os.path.join(HERE, f"results_p{n}.csv"))
    df = df[pd.to_numeric(df["auc_pr"], errors="coerce").notna()].copy()
    df["auc_pr"] = df["auc_pr"].astype(float)
    df = df[(pd.to_numeric(df["tau"], errors="coerce") == 2.0)
            & (pd.to_numeric(df["lam"], errors="coerce") == 1.0)
            & (df["variant"] == VARIANT[n])]
    ex = df["extra"].fillna("")
    fixed = df[ex == ""].groupby("collection")["auc_pr"].mean()
    auto = df[ex.str.contains("auto_tr")].groupby("collection")["auc_pr"].mean()
    return fixed, auto


def make_chart(n):
    fixed, auto = proposal_means(n)
    rw_g, da_g = ref_means()
    colls = [c for c in ORDER if c in fixed.index]
    da = [float(da_g.loc[c, "da"]) if c in da_g.index else np.nan for c in colls]
    rw = [float(rw_g.loc[c, "rw"]) if c in rw_g.index else np.nan for c in colls]
    pf = [float(fixed.get(c, np.nan)) for c in colls]
    pa = [float(auto.get(c, np.nan)) for c in colls]

    x = np.arange(len(colls)); w = 0.2
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.bar(x - 1.5 * w, da, w, label="DeepAnT (best-HP)", color="#b0b7c0")
    ax.bar(x - 0.5 * w, rw, w, label="RW-1 (best-HP/200ep)", color="#4c78a8")
    ax.bar(x + 0.5 * w, pf, w, label=f"{NAME[n]} (fixed λ)", color="#f58518")
    ax.bar(x + 1.5 * w, pa, w, label=f"{NAME[n]} (auto-λ)", color="#54a24b")
    # mark only a MEANINGFUL win (auto-λ over RW-1 by > 0.003; ignores trivial ties)
    for i in range(len(colls)):
        if not np.isnan(pa[i]) and not np.isnan(rw[i]) and pa[i] > rw[i] + 0.003:
            ax.text(x[i] + 1.5 * w, pa[i] + 0.02, "beats\nRW-1", ha="center",
                    va="bottom", fontsize=10, color="#54a24b", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(colls, rotation=0)
    ax.set_ylabel("AUC-PR (whole-collection mean)")
    ax.set_ylim(0, 1.18)
    ax.set_title(f"{NAME[n]}: AUC-PR by collection vs RW-1 / DeepAnT")
    ax.legend(loc="upper left", ncol=1, framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = os.path.join(FIG, f"P{n}_comparison_bars.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"P{n}: saved -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", type=int, default=5)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    for n in (range(1, 6) if args.all else [args.proposal]):
        make_chart(n)


if __name__ == "__main__":
    main()
