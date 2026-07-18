#!/usr/bin/env python3
"""Aggregate + log the cr x l1 RE-RANKING of P1-P5 (the correction_rate follow-up to the
original screening, which fixed cr=0.1), plus the P5 full-200 benchmark.

For each proposal N, reads results_p{N}_proposal{N}_crtune.csv (504 runs = 7 collections x
per-series x cr{0.001,0.01,0.1} x l1{0.001,0.01,0.1,1.0} at 200 epochs). Per-series best
over the cr x l1 grid, then collection mean = "P{N} crtune-best". Compared to RW-1 best-HP
(summary_rw1_besthp.csv, l1-only sweep). Emits a clean grouped bar chart + a wandb summary
run. Also reports P5 full-200 (per-file best over cr x l1) vs the thesis RW-1 0.28.

IMPORTANT caveat baked into the outputs: RW-1 best-HP was tuned over l1 ONLY (cr fixed),
so P1-P5 here have one extra tuning axis -> the "wins vs RW-1" count is NOT like-for-like.
The defensible finding is that the five gates cluster tightly (gate choice ~ negligible)
and correction_rate is the dominant, collection-dependent lever.

    python experiments/proposals/log_crtune_rerank.py            # log to wandb
    python experiments/proposals/log_crtune_rerank.py --no-wandb # figure + txt only
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ORDER = ["GECCO", "OPPORTUNITY", "CreditCard", "TAO", "PSM", "MSL", "SWaT"]
NAMES = {1: "P1-residual", 2: "P2-uncertainty", 3: "P3-consistency",
         4: "P4-dualgrad(wb)", 5: "P5-persistence"}
CSV = {n: f"results_p{n}_proposal{n}_crtune.csv" for n in range(1, 6)}


def rw_means():
    rw = pd.read_csv(os.path.join(ROOT, "rw/reproduction/summary_rw1_besthp.csv"))
    return rw.groupby("family")["best_pr"].mean()


def best_by_collection(path):
    d = pd.read_csv(path)
    d = d[pd.to_numeric(d["auc_pr"], errors="coerce").notna()].copy()
    d["auc_pr"] = d["auc_pr"].astype(float)
    b = d.loc[d.groupby(["collection", "dataset"])["auc_pr"].idxmax()]
    return b.groupby("collection")["auc_pr"].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    rw = rw_means()
    tab = {n: best_by_collection(os.path.join(HERE, CSV[n])) for n in range(1, 6)}

    # -- combined table + txt ------------------------------------------------
    rows = []
    for c in ORDER:
        row = {"collection": c, "RW1_best": round(float(rw.get(c, np.nan)), 4)}
        for n in range(1, 6):
            row[NAMES[n]] = round(float(tab[n].get(c, np.nan)), 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    means = {"collection": "MEAN(7)", "RW1_best": round(float(rw.reindex(ORDER).mean()), 4)}
    for n in range(1, 6):
        means[NAMES[n]] = round(float(tab[n].reindex(ORDER).mean()), 4)
    df = pd.concat([df, pd.DataFrame([means])], ignore_index=True)
    wins = {NAMES[n]: int((tab[n].reindex(ORDER) > rw.reindex(ORDER)).sum()) for n in range(1, 6)}

    # -- P5 full-200 ---------------------------------------------------------
    f = pd.read_csv(os.path.join(HERE, "results_p5_p5_full200_besthp.csv"))
    f = f[pd.to_numeric(f["auc_pr"], errors="coerce").notna()].copy()
    f["auc_pr"] = f["auc_pr"].astype(float)
    per = f.groupby("dataset")["auc_pr"].max()
    full200_mean, full200_n = float(per.mean()), int(len(per))

    txt = os.path.join(HERE, "docs", "crtune_rerank.txt")
    with open(txt, "w") as fh:
        fh.write("cr x l1 RE-RANK (per-series best-HP, collection mean)\n")
        fh.write(df.to_string(index=False) + "\n\n")
        fh.write(f"wins vs RW-1 (of 7): {wins}\n")
        fh.write("CAVEAT: RW-1 best-HP swept l1 only (cr fixed); P1-P5 got the extra cr axis "
                 "-> wins are NOT like-for-like.\n\n")
        fh.write(f"P5 full-200 (per-file best over cr x l1) = {full200_mean:.4f} over {full200_n} series\n")
        fh.write("  vs thesis RW-1 0.28 / reproduction RW-1 best-HP 0.289\n")
    print(open(txt).read())

    # -- figure: grouped bar chart ------------------------------------------
    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(figsize=(15, 7))
    labels = ["RW1_best"] + [NAMES[n] for n in range(1, 6)]
    colors = ["#555555", "#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
    x = np.arange(len(ORDER)); w = 0.13
    for i, lab in enumerate(labels):
        vals = [df[df.collection == c][lab].values[0] for c in ORDER]
        ax.bar(x + (i - 2.5) * w, vals, w, label=lab, color=colors[i])
    ax.set_xticks(x); ax.set_xticklabels(ORDER, rotation=20, ha="right")
    ax.set_ylabel("AUC-PR (per-series best over cr x l1)")
    ax.set_title("cr x l1 re-ranking: P1-P5 vs RW-1 best-HP  (gates cluster tightly; "
                 "cr is the dominant lever)")
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    figp = os.path.join(HERE, "figures", "crtune_rerank.png")
    fig.savefig(figp, dpi=150, bbox_inches="tight")
    print(f"figure -> {figp}")

    if args.no_wandb:
        return

    import wandb
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, ".env"))
    except Exception:
        pass
    run = wandb.init(project="rwml-autocegar", group="crtune_rerank",
                     name="crtune-rerank-P1toP5", job_type="summary", reinit=True)
    wandb.log({"crtune_rerank_table": wandb.Table(dataframe=df),
               "crtune_rerank_chart": wandb.Image(figp)})
    for n in range(1, 6):
        wandb.summary[f"{NAMES[n]}_mean7"] = means[NAMES[n]]
        wandb.summary[f"{NAMES[n]}_wins_of7"] = wins[NAMES[n]]
    wandb.summary["RW1_mean7"] = means["RW1_best"]
    wandb.summary["P5_full200_mean"] = round(full200_mean, 4)
    wandb.summary["P5_full200_n"] = full200_n
    wandb.summary["thesis_rw1_full"] = 0.28
    run.finish()
    print("wandb summary logged:", run.url if hasattr(run, "url") else "(offline)")


if __name__ == "__main__":
    main()
