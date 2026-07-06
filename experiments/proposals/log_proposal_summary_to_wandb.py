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
    print(f"loaded {len(df)} runs from {csv}")

    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY") or None,
        project=os.environ.get("WANDB_PROJECT", "rwml-autocegar"),
        mode=os.environ.get("WANDB_MODE", "online"),
        name=f"P{args.proposal}-summary", group=f"proposal{args.proposal}",
        job_type="summary", tags=["summary", f"P{args.proposal}"],
        config={"proposal": args.proposal, "n_runs": len(df)}, reinit=True,
    )

    # full table of every grid run
    run.log({"all_runs": wandb.Table(dataframe=df)})

    # stage-1 rows carry the RW-1 baseline + delta
    stage1 = df.dropna(subset=["rw1_auc_pr"]).drop_duplicates("dataset", keep="last")
    summary = {"n_runs": len(df),
               "n_datasets_improved": int((stage1["delta_pr"] > 0).sum()),
               "n_datasets": len(stage1)}
    for _, r in stage1.iterrows():
        summary[f"delta_pr/{r['dataset']}"] = float(r["delta_pr"])
        summary[f"auc_pr/{r['dataset']}"] = float(r["auc_pr"])
        summary[f"rw1_auc_pr/{r['dataset']}"] = float(r["rw1_auc_pr"])
    run.summary.update(summary)

    # delta bar chart (P1 - RW-1 AUC-PR per dataset)
    if len(stage1):
        tbl = wandb.Table(data=[[r["dataset"], float(r["delta_pr"])] for _, r in stage1.iterrows()],
                          columns=["dataset", "delta_pr"])
        run.log({"delta_pr_by_dataset": wandb.plot.bar(tbl, "dataset", "delta_pr",
                                                       title=f"P{args.proposal} - RW-1 AUC-PR delta")})

    print("summary:", {k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()})
    run.finish()
    print(f"summary run: {run.url}")


if __name__ == "__main__":
    main()
