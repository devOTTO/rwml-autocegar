"""Upload TSB-AD benchmark results to W&B.

Reads metric CSVs from eval/metrics/multi/{AD_Name}/ and uploads to wandb.

Usage:
    python upload_wandb_tsb.py --ad-name CNN
    python upload_wandb_tsb.py --ad-name CNN --dry-run
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv

load_dotenv(Path("/ocean/projects/cis260190p/yhwang2/rwml-autocegar/.env"))

METRICS_DIR = Path("eval/metrics/multi")


def parse_filename(filename: str) -> dict:
    m = re.match(
        r"(\d+)_(.+?)_id_(\d+)_(.+?)_tr_(\d+)_1st_(\d+)", filename
    )
    if not m:
        return {"source": "unknown", "series_id": 0}
    return {
        "idx": int(m.group(1)),
        "source": m.group(2),
        "series_id": int(m.group(3)),
        "domain": m.group(4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ad-name", type=str, default="CNN")
    parser.add_argument("--project", type=str, default="rwml-autocegar")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tag", type=str, default="tsb-ad-v4")
    args = parser.parse_args()

    metrics_dir = METRICS_DIR / args.ad_name
    if not metrics_dir.exists():
        print(f"Metrics dir not found: {metrics_dir}")
        return

    metric_files = sorted(metrics_dir.glob("*.csv"))
    print(f"Found {len(metric_files)} metric files for {args.ad_name}")

    results = []
    for mf in metric_files:
        df = pd.read_csv(mf)
        row = df.iloc[0]
        name = mf.stem
        meta = parse_filename(name)

        result = {
            **meta,
            "filename": name,
            "auc_roc": float(row.get("AUC-ROC", float("nan"))),
            "auc_pr": float(row.get("AUC-PR", float("nan"))),
            "vus_roc": float(row.get("VUS-ROC", float("nan"))),
            "vus_pr": float(row.get("VUS-PR", float("nan"))),
            "time": float(row.get("Time", 0)),
        }
        results.append(result)

    # summary
    rdf = pd.DataFrame(results)
    agg = rdf.groupby("source").agg(
        n=("filename", "size"),
        ROC=("auc_roc", "mean"),
        PR=("auc_pr", "mean"),
    ).round(4)
    print(f"\n{agg.to_string()}")

    if args.dry_run:
        print(f"\nDRY RUN — would upload {len(results)} runs")
        return

    # upload
    for i, r in enumerate(results):
        run = wandb.init(
            project=args.project,
            entity=args.entity,
            name=f"{args.ad_name}_{r.get('source','unknown')}_id_{r.get('series_id',0)}",
            group=r.get("source", "unknown"),
            config={
                "method": f"tsb-ad-{args.ad_name}",
                "ad_name": args.ad_name,
                "dataset_source": r.get("source"),
                "dataset_series_id": r.get("series_id"),
                "dataset_domain": r.get("domain"),
                "window_size": 50,
            },
            tags=[args.tag, args.ad_name, r.get("source", "unknown")],
            reinit=True,
        )
        run.log({
            "eval/auc_roc": r["auc_roc"],
            "eval/auc_pr": r["auc_pr"],
            "eval/vus_roc": r["vus_roc"],
            "eval/vus_pr": r["vus_pr"],
            "eval/time": r["time"],
        })
        run.finish()
        print(f"[{i+1}/{len(results)}] {r['filename']}: ROC={r['auc_roc']:.4f} PR={r['auc_pr']:.4f}")

    print(f"\nDone. {len(results)} runs uploaded.")


if __name__ == "__main__":
    main()
