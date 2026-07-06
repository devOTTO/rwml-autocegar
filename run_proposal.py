#!/usr/bin/env python
"""Fail-fast runner for the Auto-CEGAR proposals (versioned).

Tests ONE proposal on the datasets chosen in the week-8 meeting with Luis. The
three are topically disconnected and span the RW-vs-DeepAnT performance range, so
if a method wins on all three it is a good candidate for a full run:

    opportunity  (activity recognition)  -- DeepAnT tends to beat RW here
    gecco        (water quality sensor)  -- RW tends to be strong here
    creditcard   (finance)               -- middle, with headroom (not ~1.0)

This script DOES NOT submit any Slurm job. Run it directly on a node that already
has a GPU (e.g. inside an interactive `interact`/`salloc` session):

    # one proposal, one dataset, quick loop
    python run_proposal.py --proposal 1 --dataset opportunity --variant basic

    # all three selected datasets, with the plain RW-1 baseline for the delta
    python run_proposal.py --proposal 1 --dataset all --variant basic --baseline

    # try the more selective confidence variant
    python run_proposal.py --proposal 1 --dataset gecco --variant selective

Results are appended to experiments/proposals/results_p{N}.csv (proposal, variant,
dataset, auc_pr, auc_roc, delta vs baseline, and the key hyperparameters), so runs
across days/proposals accumulate in one comparable table.
"""
import argparse
import csv
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from autocegar.proposals import build_model, get_proposal, PROPOSALS
from rw.cnn_rw import CNN_RW

# Load .env (WANDB_ENTITY / WANDB_PROJECT / WANDB_API_KEY / WANDB_MODE ...).
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except Exception:
    pass

DATA_DIR = "/ocean/projects/cis260190p/yhwang2/data/TSB-AD-M/TSB-AD-M/"

# The three datasets selected in the week-8 meeting (topically disconnected,
# spanning the RW-vs-DeepAnT delta). opportunity id_1 is the canonical, fast one.
DATASETS = {
    "opportunity": "129_OPPORTUNITY_id_1_HumanActivity_tr_1801_1st_1901.csv",
    "gecco":       "173_GECCO_id_1_Sensor_tr_16165_1st_16265.csv",
    "creditcard":  "137_CreditCard_id_1_Finance_tr_500_1st_541.csv",
}
SELECTED = ["opportunity", "gecco", "creditcard"]  # 'all' runs these in order

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "experiments", "proposals")


def load_dataset(key):
    df = pd.read_csv(os.path.join(DATA_DIR, DATASETS[key])).dropna()
    data = df.iloc[:, :-1].values.astype(float)
    label = df["Label"].astype(int).to_numpy()
    return data, label


def evaluate(scores, label):
    # align length (score is per-timestep, one per row)
    n = min(len(scores), len(label))
    s, y = np.asarray(scores[:n], float), np.asarray(label[:n], int)
    return average_precision_score(y, s), roc_auc_score(y, s)


def wandb_enabled(args):
    """wandb on unless --no-wandb, WANDB_ENABLED=0, or WANDB_MODE=disabled."""
    if args.no_wandb:
        return False
    if os.environ.get("WANDB_ENABLED", "1") == "0":
        return False
    if os.environ.get("WANDB_MODE", "online") == "disabled":
        return False
    return True


def fit_with_wandb(model, data, run_name, group, tags, config, enabled):
    """Fit ``model`` while streaming its per-epoch metrics to a wandb run.

    Opens ONE run per (model, dataset), logs every epoch via the model's
    ``on_epoch_end`` hook, and returns the run handle (or None) so the caller can
    write final metrics to its summary before finishing it.
    """
    if not enabled:
        model.on_epoch_end = None
        return None, model.fit(data)

    import wandb
    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY") or None,
        project=os.environ.get("WANDB_PROJECT", "rwml-autocegar"),
        mode=os.environ.get("WANDB_MODE", "online"),
        name=run_name, group=group, tags=tags, config=config, reinit=True,
    )
    # Log per-epoch metrics under a "train/" namespace so they group together
    # and stay separate from config columns (lam/tau/...) and eval columns
    # (auc_pr/...). Consistent across all proposals -> reusable column layout.
    model.on_epoch_end = lambda m: wandb.log(
        {f"train/{k}": v for k, v in m.items() if k != "epoch"}, step=int(m["epoch"]))
    scores = model.fit(data)
    return run, scores


def run_one(args, dataset_key):
    print(f"\n{'='*70}\n[proposal {args.proposal} / {args.variant}] dataset={dataset_key}\n{'='*70}", flush=True)
    data, label = load_dataset(dataset_key)
    print(f"data {data.shape} | anomalies {int(label.sum())} ({label.mean()*100:.2f}%)", flush=True)

    common = dict(window_size=args.window, feats=data.shape[1], epochs=args.epochs,
                  batch_size=args.batch, l1_weight=args.l1_weight)
    wb = wandb_enabled(args)
    shared_cfg = dict(dataset=dataset_key, dataset_file=DATASETS[dataset_key],
                      n_anomalies=int(label.sum()), anomaly_frac=float(label.mean()),
                      **common)

    # ── baseline RW-1 first (own wandb run) so we have its AUC for the delta ──
    base_pr = base_roc = None
    if args.baseline:
        print(f"\n-- baseline RW-1 (plain) on {dataset_key} --", flush=True)
        base = CNN_RW(**common)
        brun, bscores = fit_with_wandb(
            base, data, run_name=f"RW1-baseline-{dataset_key}-ep{args.epochs}",
            group=f"proposal{args.proposal}",
            tags=["baseline", "RW-1", dataset_key, f"ep{args.epochs}"] + list(args.tag),
            config={**shared_cfg, "model": "RW-1", "score": "mean|correction|",
                    "tags": list(args.tag)}, enabled=wb)
        base_pr, base_roc = evaluate(bscores, label)
        print(f">> RW-1 baseline {dataset_key}: AUC-PR={base_pr:.4f} AUC-ROC={base_roc:.4f}", flush=True)
        if brun is not None:
            brun.summary.update({"auc_pr": base_pr, "auc_roc": base_roc})
            brun.finish()

    # ── the proposal model (own wandb run; delta goes in its summary) ──
    model = build_model(
        args.proposal, variant=args.variant,
        lam=args.lam, tau=args.tau, k=args.k,
        warmup_epochs=args.warmup, **common)
    entry = get_proposal(args.proposal)
    prop_cfg = {**shared_cfg, "model": entry["name"], "proposal": args.proposal,
                "variant": args.variant, "lam": args.lam, "tau": args.tau, "k": args.k,
                "conf_mode": getattr(model, "conf_mode", None),
                "conf_q": getattr(model, "conf_q", None),
                "warmup_epochs": args.warmup, "scale_normalize": model.scale_normalize,
                "correction_init": model.correction_init, "score": "mean|correction|"}
    prun, scores = fit_with_wandb(
        model, data,
        run_name=f"P{args.proposal}-{args.variant}-{dataset_key}-ep{args.epochs}-t{args.tau}-l{args.lam}",
        group=f"proposal{args.proposal}",
        tags=[f"P{args.proposal}", args.variant, dataset_key,
              f"tau{args.tau}", f"lam{args.lam}", f"ep{args.epochs}"] + list(args.tag),
        config={**prop_cfg, "tags": list(args.tag)}, enabled=wb)
    pr, roc = evaluate(scores, label)
    print(f"\n>> P{args.proposal}-{args.variant} {dataset_key}: AUC-PR={pr:.4f} AUC-ROC={roc:.4f}", flush=True)

    if prun is not None:
        summary = {"auc_pr": pr, "auc_roc": roc}
        if base_pr is not None:
            summary.update({"rw1_auc_pr": base_pr, "rw1_auc_roc": base_roc,
                            "delta_pr": pr - base_pr, "delta_roc": roc - base_roc})
        prun.summary.update(summary)
        prun.finish()
    if base_pr is not None:
        print(f">> DELTA AUC-PR (proposal - RW1) = {pr - base_pr:+.4f}", flush=True)

    log_result(args, dataset_key, pr, roc, base_pr, base_roc)
    return pr, roc


def log_result(args, dataset_key, pr, roc, base_pr, base_roc):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"results_p{args.proposal}.csv")
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "proposal": args.proposal,
        "variant": args.variant,
        "dataset": dataset_key,
        "auc_pr": round(float(pr), 4),
        "auc_roc": round(float(roc), 4),
        "rw1_auc_pr": None if base_pr is None else round(float(base_pr), 4),
        "delta_pr": None if base_pr is None else round(float(pr - base_pr), 4),
        "epochs": args.epochs, "warmup": args.warmup,
        "lam": args.lam, "tau": args.tau, "k": args.k,
        "window": args.window, "batch": args.batch, "l1_weight": args.l1_weight,
    }
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"[logged -> {path}]", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--proposal", type=int, default=1, choices=sorted(PROPOSALS))
    p.add_argument("--dataset", default="all",
                   choices=list(DATASETS) + ["all"],
                   help="one of the selected datasets, or 'all' (opportunity,gecco,creditcard)")
    p.add_argument("--variant", default=None,
                   help="proposal variant (P1: basic|selective). Default = proposal's default.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10, help="forecaster-only warm-up epochs (week-8: 10-15)")
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=2.0, help="robust-z wrongness threshold")
    p.add_argument("--k", type=float, default=1.0, help="gate sigmoid sharpness")
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--l1_weight", type=float, default=0.001)
    p.add_argument("--baseline", action="store_true", help="also run plain RW-1 for the delta")
    p.add_argument("--no-wandb", action="store_true",
                   help="disable wandb logging (default: on if WANDB_ENABLED!=0)")
    p.add_argument("--tag", action="append", default=[],
                   help="extra wandb tag (repeatable), e.g. --tag stage1")
    args = p.parse_args()

    entry = get_proposal(args.proposal)
    if entry["cls"] is None:
        sys.exit(f"Proposal {args.proposal} ({entry['name']}) not implemented yet.")
    if args.variant is None:
        args.variant = entry.get("default_variant")

    keys = SELECTED if args.dataset == "all" else [args.dataset]
    print(f"Proposal {args.proposal}: {entry['name']} | variant={args.variant} | datasets={keys}", flush=True)

    summary = []
    for key in keys:
        pr, roc = run_one(args, key)
        summary.append((key, pr, roc))

    print(f"\n{'='*70}\nSUMMARY  Proposal {args.proposal} ({args.variant})\n{'='*70}")
    for key, pr, roc in summary:
        print(f"  {key:14s} AUC-PR={pr:.4f}  AUC-ROC={roc:.4f}")


if __name__ == "__main__":
    main()
