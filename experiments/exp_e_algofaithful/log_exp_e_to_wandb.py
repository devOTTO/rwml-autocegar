#!/usr/bin/env python3
"""Aggregate EXP-E (algorithm-faithful full-run) results and log them to W&B.

EXP-E is the full 180-dataset TSB-AD-M run of the algorithm-faithful rewrite
of RW (CNN_UNS, Algorithm 1) and RW-1 (CNN_RW, Algorithm 2). See
experiments/exp_e_algofaithful/README.md for the full context.

This script:
  * loads every per-dataset result CSV that Run_Detector_M wrote,
  * computes summary statistics per algorithm (mean / median over datasets),
  * counts the "freeze" symptom (AUC-ROC << 0.5) that RW-1 showed on Genesis,
  * logs a separate W&B run per algorithm: config, summary metrics, a table
    of every dataset's metrics, and histograms of AUC-PR / AUC-ROC.

Run inside the xlstmad_env venv (has wandb 0.27.1, pandas 3.x):
    source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
    python experiments/exp_e_algofaithful/log_exp_e_to_wandb.py
"""
import glob
import os

import numpy as np
import pandas as pd
import wandb

RESULTS_ROOT = (
    "/ocean/projects/cis260190p/yhwang2/TSB-AD/eval/metrics/multi_algofaithful"
)

# (algorithm folder, paper method name, paper's reported best AUC-PR)
ALGOS = [
    ("CNN_UNS", "RW", "Algorithm 1", 0.34),
    ("CNN_RW", "RW-1", "Algorithm 2", 0.35),
]

METRIC_COLS = [
    "AUC-PR", "AUC-ROC", "VUS-PR", "VUS-ROC",
    "Standard-F1", "PA-F1", "Event-based-F1", "R-based-F1", "Affiliation-F",
]

# Shared knobs of the algorithm-faithful rewrite (paper's best config).
COMMON_CONFIG = {
    "experiment": "EXP-E algo-faithful-full-run",
    "code_version": "03_rewritten_algorithm_faithful",
    "benchmark": "TSB-AD-M",
    "correction_optimizer": "RMSprop",
    "loss": "RMSE",
    "correction_rate_lr": 0.1,
    "update_cadence": "epoch-wise",
    "epochs": 200,
    "savgol_smoothing": False,
    "score": "abs(correction) averaged over feature channels",
    "slurm_partition": "GPU-shared",
    "gpu": "v100-32:1",
}


def load_algo(folder):
    """Concatenate every per-dataset result CSV for one algorithm."""
    paths = sorted(glob.glob(os.path.join(RESULTS_ROOT, folder, "*.csv")))
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception as e:  # keep going; note the bad file
            print(f"  !! failed to read {os.path.basename(p)}: {e}")
    df = pd.concat(frames, ignore_index=True)
    # numeric coercion; some metrics can be blank on degenerate datasets
    for c in METRIC_COLS + ["Time"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    entity = os.environ.get("WANDB_ENTITY", "yoonmeeh-cmu")
    project = os.environ.get("WANDB_PROJECT", "rwml-autocegar")

    print(f"Logging to W&B: {entity}/{project} (group=EXP-E-algofaithful)\n")

    overall = {}
    for folder, method, algo_ref, paper_pr in ALGOS:
        df = load_algo(folder)
        n = len(df)
        aucpr_mean = float(df["AUC-PR"].mean())
        aucpr_median = float(df["AUC-PR"].median())
        aucroc_mean = float(df["AUC-ROC"].mean())
        # "freeze" symptom: AUC-ROC well below random (see README / Genesis case)
        freeze = df[df["AUC-ROC"] < 0.4]
        n_freeze = int(len(freeze))

        print(f"== {method} ({folder}, {algo_ref}) ==")
        print(f"   datasets            : {n}")
        print(f"   AUC-PR  mean/median : {aucpr_mean:.4f} / {aucpr_median:.4f}  "
              f"(paper {paper_pr})")
        print(f"   AUC-ROC mean        : {aucroc_mean:.4f}")
        print(f"   freeze (AUC-ROC<0.4): {n_freeze}")
        if n_freeze:
            names = ", ".join(freeze["file"].astype(str).head(10))
            print(f"     -> {names}")
        print()

        run = wandb.init(
            entity=entity,
            project=project,
            group="EXP-E-algofaithful",
            name=f"EXP-E-{method}-full",
            job_type="eval-aggregate",
            tags=["EXP-E", "algo-faithful", method, "full-run"],
            config={**COMMON_CONFIG, "method": method,
                    "paper_algorithm": algo_ref,
                    "paper_best_auc_pr": paper_pr,
                    "ad_name": folder, "n_datasets": n},
            reinit=True,
        )

        summary = {"n_datasets": n, "n_freeze": n_freeze,
                   "paper_best_auc_pr": paper_pr}
        for c in METRIC_COLS:
            summary[f"{c}/mean"] = float(df[c].mean())
            summary[f"{c}/median"] = float(df[c].median())
        summary["Time/mean_s"] = float(df["Time"].mean())
        summary["AUC-PR/vs_paper"] = aucpr_mean - paper_pr
        run.summary.update(summary)

        # per-dataset table (sortable/filterable in the W&B UI)
        run.log({"per_dataset": wandb.Table(dataframe=df)})
        # distributions
        run.log({
            "hist/AUC-PR": wandb.Histogram(df["AUC-PR"].dropna().tolist()),
            "hist/AUC-ROC": wandb.Histogram(df["AUC-ROC"].dropna().tolist()),
        })
        run.finish()

        overall[method] = {
            "n": n, "aucpr_mean": aucpr_mean, "aucpr_median": aucpr_median,
            "aucroc_mean": aucroc_mean, "n_freeze": n_freeze,
            "paper_pr": paper_pr,
        }

    print("Done. Summary:")
    for m, s in overall.items():
        print(f"  {m}: AUC-PR mean {s['aucpr_mean']:.4f} "
              f"(paper {s['paper_pr']}), median {s['aucpr_median']:.4f}, "
              f"freeze {s['n_freeze']}/{s['n']}")


if __name__ == "__main__":
    main()
