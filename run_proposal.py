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
    # fail-fast verdict set (week-8 pick: feature-count + length axes, RW-vs-DeepAnT gap)
    "opportunity": "129_OPPORTUNITY_id_1_HumanActivity_tr_1801_1st_1901.csv",
    "gecco":       "173_GECCO_id_1_Sensor_tr_16165_1st_16265.csv",
    "creditcard":  "137_CreditCard_id_1_Finance_tr_500_1st_541.csv",
    # characterization set (domain / anomaly-type diversity — NOT part of the
    # pass/fail verdict; used to map when/why a proposal helps). Added for P2,
    # whose success hinges on the "confident-error vs high-uncertainty" axis.
    "smap":  "144_SMAP_id_1_Sensor_tr_2052_1st_5300.csv",       # satellite telemetry (point anomalies)
    "smd":   "057_SMD_id_1_Facility_tr_4529_1st_4629.csv",      # industrial server (neutral)
    "mitdb": "019_MITDB_id_1_Medical_tr_37500_1st_103211.csv",  # ECG arrhythmia (periodic -> gecco-like control)
}
SELECTED = ["opportunity", "gecco", "creditcard"]  # 'all' runs the verdict set only

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "experiments", "proposals")


def load_dataset(key):
    # key may be a short registry name (DATASETS) or a raw TSB-AD-M filename,
    # so a collection's individual series can be run directly for a full-collection
    # (rather than single-representative) evaluation.
    fname = DATASETS.get(key, key)
    path = fname if os.path.isabs(fname) else os.path.join(DATA_DIR, fname)
    df = pd.read_csv(path).dropna()
    data = df.iloc[:, :-1].values.astype(float)
    label = df["Label"].astype(int).to_numpy()
    return data, label


def collection_of(key):
    """Collection name for a registry key or a raw 'NNN_COLLECTION_id_...' filename."""
    fname = DATASETS.get(key, key)
    parts = fname.split("_")
    if len(parts) >= 2 and parts[0].isdigit():
        return parts[1]            # e.g. 144_SMAP_id_1_... -> SMAP
    return key


def series_label(key):
    """Short label for run names/CSV: registry key, or 'collection_idN' for a raw file."""
    if key in DATASETS:
        return key
    parts = DATASETS.get(key, key).split("_")
    if len(parts) >= 4 and parts[2] == "id":
        return f"{parts[1].lower()}_id{parts[3]}"   # 144_SMAP_id_1_... -> smap_id1
    return parts[1].lower() if len(parts) >= 2 else key


def evaluate(scores, label):
    # align length (score is per-timestep, one per row)
    n = min(len(scores), len(label))
    s, y = np.asarray(scores[:n], float), np.asarray(label[:n], int)
    return average_precision_score(y, s), roc_auc_score(y, s)


def parse_extra(pairs):
    """Parse repeatable ``--extra key=val`` into a kwargs dict for build_model.

    Lets each proposal take its own hyperparameters without new CLI flags
    (e.g. P2 `--extra tau_u=1.0 --extra mc_samples=8`). Values are coerced to
    int, then float, else kept as string.
    """
    out = {}
    for p in pairs or []:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                pass
        out[k.strip()] = v
    return out


def gate_interpretability(scores, gate_per_t, label, thr=0.5):
    """Was the gate triggered AT anomalies, and did correction concentrate there?

    Compares the per-timestep gate activation and correction magnitude on
    anomaly vs normal timesteps (ratio > 1 = fires more on anomalies), plus how
    well the gate alone localizes anomalies, and trigger counts.
    """
    n = min(len(scores), len(gate_per_t), len(label))
    s = np.asarray(scores[:n], float)
    g = np.asarray(gate_per_t[:n], float)
    y = np.asarray(label[:n], int).astype(bool)
    out = {}
    if y.any() and (~y).any():
        out["gate/anom_mean"] = float(g[y].mean())
        out["gate/norm_mean"] = float(g[~y].mean())
        out["gate/anom_over_norm"] = float(g[y].mean() / max(g[~y].mean(), 1e-9))
        out["corr/anom_mean"] = float(s[y].mean())
        out["corr/norm_mean"] = float(s[~y].mean())
        out["corr/anom_over_norm"] = float(s[y].mean() / max(s[~y].mean(), 1e-9))
        # thesis §8.4 correction diagnostics: high-correction = |correction| above
        # its 95th percentile (tau_C). Overlap = precision (of high-corr points,
        # frac that are anomalies); AnomalyCoverage = recall (of anomalies, frac
        # that are high-corr). Same idea as trigger_precision/recall but on
        # |correction| instead of the gate — lets us compare P1/P2 to Baldo directly.
        tau_c = float(np.quantile(s, 0.95))
        hi = s > tau_c
        out["corr/tau_c_q95"] = tau_c
        out["corr/overlap"] = float(y[hi].mean()) if hi.any() else 0.0
        out["corr/anomaly_coverage"] = float(hi[y].mean())
        out["gate/auc_roc_vs_label"] = float(roc_auc_score(y, g))
        out["gate/auc_pr_vs_label"] = float(average_precision_score(y, g))
    trig = g > thr
    out["gate/trigger_frac"] = float(trig.mean())          # fraction of timeline gated
    out["gate/trigger_count"] = int(trig.sum())            # how many timesteps triggered
    if trig.any():
        out["gate/trigger_precision"] = float(y[trig].mean())  # of triggered, frac at anomalies
    if y.any():
        out["gate/trigger_recall"] = float(trig[y].mean())     # of anomalies, frac triggered
    return out


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
    ds = series_label(dataset_key)          # short label (registry key or collection_idN)
    coll = collection_of(dataset_key)
    print(f"\n{'='*70}\n[proposal {args.proposal} / {args.variant}] dataset={ds} (collection={coll})\n{'='*70}", flush=True)
    data, label = load_dataset(dataset_key)
    print(f"data {data.shape} | anomalies {int(label.sum())} ({label.mean()*100:.2f}%)", flush=True)

    common = dict(window_size=args.window, feats=data.shape[1], epochs=args.epochs,
                  batch_size=args.batch, l1_weight=args.l1_weight)
    wb = wandb_enabled(args)
    shared_cfg = dict(dataset=ds, collection=coll,
                      dataset_file=DATASETS.get(dataset_key, dataset_key),
                      n_anomalies=int(label.sum()), anomaly_frac=float(label.mean()),
                      **common)

    # ── matched RW-1-only mode: de-confound deltas from the reproduction baseline ──
    if getattr(args, "baseline_only", False):
        base = CNN_RW(**common)
        brun, bscores = fit_with_wandb(
            base, data, run_name=f"RW1m-{ds}-ep{args.epochs}",
            group=f"proposal{args.proposal}",
            tags=["rw1-matched", "RW-1", coll, f"ep{args.epochs}"] + list(args.tag),
            config={**shared_cfg, "model": "RW-1-matched"}, enabled=wb)
        pr, roc = evaluate(bscores, label)
        nn = min(len(bscores), len(label))
        s = np.asarray(bscores[:nn], float); y = np.asarray(label[:nn], int).astype(bool)
        corr_ratio = float(s[y].mean() / max(s[~y].mean(), 1e-9)) if y.any() and (~y).any() else float("nan")
        print(f">> RW1-matched {ds}: AUC-PR={pr:.4f} AUC-ROC={roc:.4f} corr@anom/norm={corr_ratio:.2f}", flush=True)
        if brun is not None:
            brun.summary.update({"auc_pr": pr, "auc_roc": roc, "corr/anom_over_norm": corr_ratio})
            brun.finish()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, "results_rw1matched.csv")
        row = {"dataset": ds, "collection": coll, "auc_pr": round(pr, 4), "auc_roc": round(roc, 4),
               "corr_anom_over_norm": round(corr_ratio, 4), "epochs": args.epochs}
        wh = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if wh:
                w.writeheader()
            w.writerow(row)
        print(f"[rw1-matched logged -> {path}]", flush=True)
        return pr, roc

    # ── baseline RW-1 first (own wandb run) so we have its AUC for the delta ──
    base_pr = base_roc = None
    if args.baseline:
        print(f"\n-- baseline RW-1 (plain) on {ds} --", flush=True)
        base = CNN_RW(**common)
        brun, bscores = fit_with_wandb(
            base, data, run_name=f"RW1-baseline-{ds}-ep{args.epochs}",
            group=f"proposal{args.proposal}",
            tags=["baseline", "RW-1", coll, f"ep{args.epochs}"] + list(args.tag),
            config={**shared_cfg, "model": "RW-1", "score": "mean|correction|",
                    "tags": list(args.tag)}, enabled=wb)
        base_pr, base_roc = evaluate(bscores, label)
        print(f">> RW-1 baseline {dataset_key}: AUC-PR={base_pr:.4f} AUC-ROC={base_roc:.4f}", flush=True)
        if brun is not None:
            brun.summary.update({"auc_pr": base_pr, "auc_roc": base_roc})
            brun.finish()

    # ── the proposal model (own wandb run; delta goes in its summary) ──
    extra = parse_extra(args.extra)
    model = build_model(
        args.proposal, variant=args.variant,
        lam=args.lam, tau=args.tau, k=args.k,
        warmup_epochs=args.warmup, **common, **extra)
    entry = get_proposal(args.proposal)
    prop_cfg = {**shared_cfg, "model": entry["name"], "proposal": args.proposal,
                "variant": args.variant, "lam": args.lam, "tau": args.tau, "k": args.k,
                "conf_mode": getattr(model, "conf_mode", None),
                "conf_q": getattr(model, "conf_q", None),
                "mc_samples": getattr(model, "mc_samples", None),
                "tau_u": getattr(model, "tau_u", None),
                "warmup_epochs": args.warmup, "scale_normalize": model.scale_normalize,
                "correction_init": model.correction_init, "score": "mean|correction|"}
    extra_suffix = ("-" + "_".join(f"{k}{v}" for k, v in extra.items())) if extra else ""
    prun, scores = fit_with_wandb(
        model, data,
        run_name=f"P{args.proposal}-{args.variant}-{ds}-ep{args.epochs}-t{args.tau}-l{args.lam}{extra_suffix}",
        group=f"proposal{args.proposal}",
        tags=[f"P{args.proposal}", args.variant, coll,
              f"tau{args.tau}", f"lam{args.lam}", f"ep{args.epochs}"] + list(args.tag),
        config={**prop_cfg, **extra, "tags": list(args.tag)}, enabled=wb)
    pr, roc = evaluate(scores, label)
    print(f"\n>> P{args.proposal}-{args.variant} {ds}: AUC-PR={pr:.4f} AUC-ROC={roc:.4f}", flush=True)

    # ── interpretability: did the gate fire at anomalies / correction concentrate there? ──
    interp = {}
    if getattr(model, "gate_per_t", None) is not None:
        interp = gate_interpretability(scores, model.gate_per_t, label)
        print(f">> INTERPRET | gate@anom/norm={interp.get('gate/anom_over_norm', float('nan')):.2f} "
              f"| corr@anom/norm={interp.get('corr/anom_over_norm', float('nan')):.2f} "
              f"| gate→label AUC-ROC={interp.get('gate/auc_roc_vs_label', float('nan')):.3f} "
              f"| trigger_frac={interp.get('gate/trigger_frac', float('nan')):.3f} "
              f"(prec={interp.get('gate/trigger_precision', float('nan')):.3f}, "
              f"recall={interp.get('gate/trigger_recall', float('nan')):.3f})", flush=True)

    if prun is not None:
        summary = {"auc_pr": pr, "auc_roc": roc, **interp}
        if base_pr is not None:
            summary.update({"rw1_auc_pr": base_pr, "rw1_auc_roc": base_roc,
                            "delta_pr": pr - base_pr, "delta_roc": roc - base_roc})
        prun.summary.update(summary)
        prun.finish()
    if base_pr is not None:
        print(f">> DELTA AUC-PR (proposal - RW1) = {pr - base_pr:+.4f}", flush=True)

    log_result(args, dataset_key, pr, roc, base_pr, base_roc, model=model, extra=extra)
    return pr, roc


def log_result(args, dataset_key, pr, roc, base_pr, base_roc, model=None, extra=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"results_p{args.proposal}.csv")
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "proposal": args.proposal,
        "variant": args.variant,
        "dataset": series_label(dataset_key),
        "collection": collection_of(dataset_key),
        "auc_pr": round(float(pr), 4),
        "auc_roc": round(float(roc), 4),
        "rw1_auc_pr": None if base_pr is None else round(float(base_pr), 4),
        "delta_pr": None if base_pr is None else round(float(pr - base_pr), 4),
        "epochs": args.epochs, "warmup": args.warmup,
        "lam": args.lam, "tau": args.tau, "k": args.k,
        # proposal-specific swept params so sweep rows are distinguishable in the
        # CSV (P1 conf_mode; P2 mc_samples/tau_u). Absent ones log as None.
        "conf_mode": getattr(model, "conf_mode", None),
        "mc_samples": getattr(model, "mc_samples", None),
        "tau_u": getattr(model, "tau_u", None),
        "extra": (";".join(f"{k}={v}" for k, v in extra.items()) if extra else ""),
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
                   help="a registry key (opportunity/gecco/...), 'all' (=verdict set), "
                        "or a raw TSB-AD-M series filename for full-collection runs")
    p.add_argument("--variant", default=None,
                   help="proposal variant (P1: basic|selective; P2: mc5|mc10). Default = proposal's default.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10, help="forecaster-only warm-up epochs (week-8: 10-15)")
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=2.0, help="robust-z wrongness threshold")
    p.add_argument("--k", type=float, default=1.0, help="gate sigmoid sharpness")
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--l1_weight", type=float, default=0.001)
    p.add_argument("--baseline", action="store_true", help="also run plain RW-1 for the delta")
    p.add_argument("--baseline-only", dest="baseline_only", action="store_true",
                   help="run ONLY a matched RW-1 (same epochs/HP) on the series, log AUC + "
                        "corr@anom/norm to results_rw1matched.csv (de-confounds P1/P2 deltas)")
    p.add_argument("--no-wandb", action="store_true",
                   help="disable wandb logging (default: on if WANDB_ENABLED!=0)")
    p.add_argument("--tag", action="append", default=[],
                   help="extra wandb tag (repeatable), e.g. --tag stage1")
    p.add_argument("--extra", action="append", default=[],
                   help="proposal-specific kwarg key=val (repeatable), e.g. --extra tau_u=1.0")
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
