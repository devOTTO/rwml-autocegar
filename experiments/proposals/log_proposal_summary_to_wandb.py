#!/usr/bin/env python3
"""Log a per-proposal SUMMARY run to wandb from results_p{N}.csv.

The grid runs each log their own per-epoch curve + final AUC (see run_proposal.py).
This aggregates them into ONE summary run: a full table of every grid run, plus
the stage-1 delta-vs-RW-1 scalars and a delta bar chart, so the proposal's verdict
is visible at a glance in the wandb workspace.

    source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
    cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
    python experiments/proposals/log_proposal_summary_to_wandb.py --proposal 1
"""
import argparse
import os

import pandas as pd
import wandb

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", type=int, default=1)
    args = ap.parse_args()

    csv = os.path.join(HERE, f"results_p{args.proposal}.csv")
    df = pd.read_csv(csv)
    # keep only the latest row per config (results_pN.csv accumulates across
    # re-runs; keep='last' -> the most recent batch for each unique setting).
    # include 'extra' so fixed-λ and auto-λ (extra=lam_mode=auto_tr) rows for the
    # same dataset are NOT collapsed into one (they share dataset/variant/tau/lam).
    keys = [c for c in ["dataset", "variant", "tau", "lam", "epochs", "extra"] if c in df.columns]
    df = df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    print(f"loaded {len(df)} runs (deduped by {keys}) from {csv}")

    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY") or None,
        project=os.environ.get("WANDB_PROJECT", "rwml-autocegar"),
        mode=os.environ.get("WANDB_MODE", "online"),
        name=f"P{args.proposal}-summary", group=f"proposal{args.proposal}_corrected",
        job_type="summary", tags=["summary", f"P{args.proposal}", "corrected"],
        config={"proposal": args.proposal, "n_runs": len(df),
                "correction_init": "neg_x"}, reinit=True,
    )

    # full table of every grid run
    run.log({"all_runs": wandb.Table(dataframe=df)})

    # per-collection verdict vs the reproduced RW-1 (best-HP). The rw1_auc_pr
    # column is empty for the corrected runs, so compare against the reproduction
    # reference means the same way aggregate_collection.py does.
    VARIANT = {1: "basic", 2: "mc5", 3: "full", 4: "gradnorm", 5: "h5"}
    ROOT = os.path.dirname(os.path.dirname(HERE))
    rw_mean = (pd.read_csv(os.path.join(ROOT, "rw/reproduction/summary_rw1_besthp.csv"))
               .groupby("family")["best_pr"].mean())
    d = df[pd.to_numeric(df["auc_pr"], errors="coerce").notna()].copy()
    d["auc_pr"] = d["auc_pr"].astype(float)
    fixed = d[(pd.to_numeric(d["tau"], errors="coerce") == 2.0)
              & (pd.to_numeric(d["lam"], errors="coerce") == 1.0)
              & (d["variant"] == VARIANT[args.proposal])
              & (d["extra"].fillna("") == "")]
    coll_mean = fixed.groupby("collection")["auc_pr"].mean()
    summary = {"n_runs": len(df), "n_collections": int(len(coll_mean))}
    beat, chart_rows = 0, []
    for coll, p_pr in coll_mean.items():
        rwv = float(rw_mean.get(coll, float("nan")))
        delta = float(p_pr) - rwv
        beat += int(delta > 0)
        summary[f"auc_pr/{coll}"] = float(p_pr)
        summary[f"rw1_auc_pr/{coll}"] = rwv
        summary[f"delta_pr/{coll}"] = delta
        chart_rows.append([coll, delta])
    summary["n_collections_beat_rw1"] = beat
    run.summary.update(summary)

    # delta bar chart (P{N} fixed-λ − RW-1 AUC-PR per collection)
    if chart_rows:
        tbl = wandb.Table(data=chart_rows, columns=["collection", "delta_pr"])
        run.log({"delta_pr_by_collection": wandb.plot.bar(
            tbl, "collection", "delta_pr",
            title=f"P{args.proposal} fixed-λ − RW-1 AUC-PR delta by collection")})

    print("summary:", {k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()})
    run.finish()
    print(f"summary run: {run.url}")


if __name__ == "__main__":
    main()
