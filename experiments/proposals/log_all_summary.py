#!/usr/bin/env python3
"""Log ONE unified P1-P5 summary run to wandb — fully code-generated, zero UI clicks.

Reads results_p{1..5}.csv, aggregates the fixed-config runs to per-(proposal,
collection) means (over the collection's series), attaches the reproduction RW-1
baseline + delta, AND the thesis Sec. 8.4 interpretability metrics
(gate->label AUC, corr@anom/norm, Overlap, AnomalyCoverage). Emits:
  - `all_proposals` : the full combined table
  - `delta_pr`      : bar chart of (P - RW-1) AUC-PR per proposal x collection
  - `gate_auc`      : bar chart of gate->label AUC per proposal x collection
  - summary scalars : per-proposal #collections beaten
Includes the auto-lambda AUC-PR as an extra column when those rows exist.

    python experiments/proposals/log_all_summary.py                 # verdict collections
    python experiments/proposals/log_all_summary.py --collections all
"""
import argparse
import os

import numpy as np
import pandas as pd
import wandb

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
NAMES = {1: "P1-residual", 2: "P2-uncertainty", 3: "P3-consistency",
         4: "P4-dualgrad", 5: "P5-persistence"}
VERDICT = ["OPPORTUNITY", "GECCO", "CreditCard"]


def rw_means():
    rw = pd.read_csv(os.path.join(ROOT, "rw/reproduction/summary_rw1_besthp.csv"))
    return rw.groupby("family")["best_pr"].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collections", default="verdict", choices=["verdict", "all"])
    args = ap.parse_args()
    rw = rw_means()

    rows = []                                        # combined table
    for n in range(1, 6):
        csv = os.path.join(HERE, f"results_p{n}.csv")
        if not os.path.exists(csv):
            continue
        d = pd.read_csv(csv)
        d = d[pd.to_numeric(d["auc_pr"], errors="coerce").notna()].copy()
        for c in ["auc_pr", "auc_roc", "gate_auc_vs_label", "corr_anom_over_norm",
                  "corr_overlap", "corr_anomaly_coverage"]:
            d[c] = pd.to_numeric(d.get(c), errors="coerce")
        d["extra"] = d["extra"].fillna("")
        fx = d[d["extra"] == ""]
        au = d[d["extra"].str.contains("auto_tr")]
        colls = sorted(fx["collection"].dropna().unique())
        if args.collections == "verdict":
            colls = [c for c in colls if c in VERDICT]
        for c in colls:
            s = fx[fx.collection == c]
            a = au[au.collection == c]
            rwp = float(rw[c]) if c in rw.index else float("nan")
            pr = float(s["auc_pr"].mean())
            rows.append({
                "proposal": NAMES[n], "collection": c, "n": int(len(s)),
                "auc_pr": round(pr, 4), "auc_roc": round(float(s["auc_roc"].mean()), 4),
                "rw1_auc_pr": round(rwp, 4), "delta_pr": round(pr - rwp, 4),
                "beats_rw1": bool(pr > rwp),
                "auto_auc_pr": round(float(a["auc_pr"].mean()), 4) if len(a) else None,
                "gate_auc": round(float(s["gate_auc_vs_label"].mean()), 4),
                "corr_anom_over_norm": round(float(s["corr_anom_over_norm"].mean()), 4),
                "overlap": round(float(s["corr_overlap"].mean()), 4),
                "anomaly_coverage": round(float(s["corr_anomaly_coverage"].mean()), 4),
            })
    tbl = pd.DataFrame(rows)
    print(tbl.to_string())

    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY") or None,
        project=os.environ.get("WANDB_PROJECT", "rwml-autocegar"),
        mode=os.environ.get("WANDB_MODE", "online"),
        name="ALL-summary", group="summary", job_type="summary",
        tags=["summary", "P1-P5", "corrected-config"],
        config={"collections": args.collections, "n_rows": len(tbl)}, reinit=True,
    )
    run.log({"all_proposals": wandb.Table(dataframe=tbl)})

    lab = [f"{r.proposal}/{r.collection}" for r in tbl.itertuples()]
    dbar = wandb.Table(data=[[l, float(r.delta_pr)] for l, r in zip(lab, tbl.itertuples())],
                       columns=["proposal_collection", "delta_pr"])
    run.log({"delta_pr": wandb.plot.bar(dbar, "proposal_collection", "delta_pr",
                                        title="AUC-PR delta (proposal - RW-1)")})
    gbar = wandb.Table(data=[[l, float(r.gate_auc)] for l, r in zip(lab, tbl.itertuples())],
                       columns=["proposal_collection", "gate_auc"])
    run.log({"gate_auc": wandb.plot.bar(gbar, "proposal_collection", "gate_auc",
                                        title="gate->label AUC")})

    summ = {"n_rows": len(tbl)}
    for n in range(1, 6):
        sub = tbl[tbl.proposal == NAMES[n]]
        if len(sub):
            summ[f"{NAMES[n]}/beats_rw1"] = int(sub["beats_rw1"].sum())
            summ[f"{NAMES[n]}/n_collections"] = int(len(sub))
    run.summary.update(summ)
    run.finish()
    print("summary:", summ)
    print(f"run: {run.url}")


if __name__ == "__main__":
    main()
