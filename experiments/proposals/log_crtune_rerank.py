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

# Thesis (Baldo, Table 6.2) per-dataset AUC-PR for method "RW 1" = the method we reproduce.
# This is Baldo's own properly-tuned (cr-tuned, §6.3.2) RW-1, so it is the fair *deployable*
# baseline. NOTE: it is on the thesis's protocol, not ours -> protocol-confounded (e.g. thesis
# RW1 SWaT 0.227 vs our reproduction RW-1 SWaT 0.444 for the SAME method).
THESIS_RW1 = {"GECCO": 0.621, "OPPORTUNITY": 0.059, "CreditCard": 0.173,
              "TAO": 1.000, "PSM": 0.238, "MSL": 0.086, "SWaT": 0.227}


def rw_means():
    rw = pd.read_csv(os.path.join(ROOT, "rw/reproduction/summary_rw1_besthp.csv"))
    return rw.groupby("family")["best_pr"].mean()


def _load(path):
    d = pd.read_csv(path)
    d = d[pd.to_numeric(d["auc_pr"], errors="coerce").notna()].copy()
    d["auc_pr"] = d["auc_pr"].astype(float)
    return d


def best_by_collection(path):
    """ORACLE: per-series best over the cr x l1 grid, then collection mean (upper bound)."""
    d = _load(path)
    b = d.loc[d.groupby(["collection", "dataset"])["auc_pr"].idxmax()]
    return b.groupby("collection")["auc_pr"].mean()


def fixed_config_by_collection(path):
    """DEPLOYABLE: pick the SINGLE (cr, l1) that maximizes mean-over-collections AUC-PR, then
    report that one config's per-collection values (no per-series oracle). Returns (series, cfg)."""
    d = _load(path)
    g = d.groupby(["correction_rate", "l1_weight", "collection"])["auc_pr"].mean().reset_index()
    piv = g.pivot_table(index=["correction_rate", "l1_weight"],
                        columns="collection", values="auc_pr").reindex(columns=ORDER)
    best = piv.mean(axis=1).idxmax()
    return piv.loc[best], best


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

    # -- FIXED-CONFIG (deployable) vs thesis RW 1 baseline -------------------
    fx, cfg = {}, {}
    for n in range(1, 6):
        fx[n], cfg[n] = fixed_config_by_collection(os.path.join(HERE, CSV[n]))
    frows = []
    for c in ORDER:
        r = {"collection": c, "thesis_RW1": THESIS_RW1[c]}
        for n in range(1, 6):
            r[NAMES[n]] = round(float(fx[n][c]), 4)
        frows.append(r)
    fmeans = {"collection": "MEAN(7)", "thesis_RW1": round(np.mean(list(THESIS_RW1.values())), 4)}
    for n in range(1, 6):
        fmeans[NAMES[n]] = round(float(np.mean([fx[n][c] for c in ORDER])), 4)
    fdf = pd.concat([pd.DataFrame(frows), pd.DataFrame([fmeans])], ignore_index=True)
    fwins = {NAMES[n]: int(sum(fx[n][c] > THESIS_RW1[c] for c in ORDER)) for n in range(1, 6)}
    cfgstr = {NAMES[n]: f"cr{cfg[n][0]}/l1{cfg[n][1]}" for n in range(1, 6)}

    txt = os.path.join(HERE, "docs", "crtune_rerank.txt")
    with open(txt, "w") as fh:
        fh.write("=== ORACLE: per-series best over cr x l1, collection mean ===\n")
        fh.write(df.to_string(index=False) + "\n\n")
        fh.write(f"wins vs reproduction RW-1 (of 7): {wins}\n")
        fh.write("CAVEAT: reproduction RW-1 swept l1 only (cr fixed at 0.1); this ORACLE table gives "
                 "P1-P5 the extra cr axis AND per-series HP -> not deployable, over-optimistic.\n\n")
        fh.write("=== DEPLOYABLE: single (cr,l1) per proposal vs thesis RW 1 (Table 6.2, Baldo-tuned) ===\n")
        fh.write(fdf.to_string(index=False) + "\n\n")
        fh.write(f"chosen single config: {cfgstr}\n")
        fh.write(f"wins vs thesis RW 1 (of 7): {fwins}\n")
        fh.write("READ: oracle 6/7 collapses to 3/7 with one deployable config. The remaining wins "
                 "(OPPORTUNITY/SWaT) are protocol-confounded -- our gate-off reproduction RW-1 already "
                 "gets SWaT 0.444 vs thesis RW1 0.227 for the SAME method -> gate contribution ~ 0.\n\n")
        fh.write(f"P5 full-200 (per-file best over cr x l1) = {full200_mean:.4f} over {full200_n} series\n")
        fh.write("  vs thesis RW-1 0.28 (Table 6.1) -- oracle upper bound; RW-1 not re-run on full-200.\n")
    print(open(txt).read())

    # -- figures: two grouped bar charts (oracle + deployable) --------------
    plt.rcParams.update({"font.size": 13})
    colors = ["#555555", "#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]

    def barchart(frame, base_col, base_label, ylab, title, out):
        fig, ax = plt.subplots(figsize=(15, 7))
        labels = [base_col] + [NAMES[n] for n in range(1, 6)]
        legend = [base_label] + [NAMES[n] for n in range(1, 6)]
        x = np.arange(len(ORDER)); w = 0.13
        for i, lab in enumerate(labels):
            vals = [frame[frame.collection == c][lab].values[0] for c in ORDER]
            ax.bar(x + (i - 2.5) * w, vals, w, label=legend[i], color=colors[i])
        ax.set_xticks(x); ax.set_xticklabels(ORDER, rotation=20, ha="right")
        ax.set_ylabel(ylab); ax.set_title(title)
        ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p = os.path.join(HERE, "figures", out)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"figure -> {p}")
        return p

    figp = barchart(df, "RW1_best", "RW-1 best (repro)",
                    "AUC-PR (per-series best over cr x l1)",
                    "ORACLE cr x l1 re-ranking: P1-P5 vs reproduction RW-1 (over-optimistic upper bound)",
                    "crtune_rerank.png")
    figp2 = barchart(fdf, "thesis_RW1", "thesis RW 1 (Table 6.2)",
                     "AUC-PR (single deployable cr x l1 config)",
                     "DEPLOYABLE single-config: P1-P5 vs thesis RW 1  (oracle 6/7 -> 3/7; gates still tied)",
                     "crtune_fixed_vs_thesis.png")

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
    wandb.log({"oracle_table": wandb.Table(dataframe=df),
               "oracle_chart": wandb.Image(figp),
               "deployable_vs_thesis_table": wandb.Table(dataframe=fdf),
               "deployable_vs_thesis_chart": wandb.Image(figp2)})
    for n in range(1, 6):
        wandb.summary[f"{NAMES[n]}_oracle_mean7"] = means[NAMES[n]]
        wandb.summary[f"{NAMES[n]}_oracle_wins_of7"] = wins[NAMES[n]]
        wandb.summary[f"{NAMES[n]}_fixed_mean7"] = fmeans[NAMES[n]]
        wandb.summary[f"{NAMES[n]}_fixed_wins_vs_thesis"] = fwins[NAMES[n]]
        wandb.summary[f"{NAMES[n]}_fixed_config"] = cfgstr[NAMES[n]]
    wandb.summary["RW1_repro_oracle_mean7"] = means["RW1_best"]
    wandb.summary["thesis_RW1_mean7"] = fmeans["thesis_RW1"]
    wandb.summary["P5_full200_mean"] = round(full200_mean, 4)
    wandb.summary["P5_full200_n"] = full200_n
    wandb.summary["thesis_rw1_full"] = 0.28
    run.finish()
    print("wandb summary logged:", run.url if hasattr(run, "url") else "(offline)")


if __name__ == "__main__":
    main()
