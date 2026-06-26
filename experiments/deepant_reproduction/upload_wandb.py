"""Upload TimeEval results to W&B.

Reads the TimeEval results CSV and creates one W&B run per time series,
grouped by dataset collection.

Usage:
    python upload_wandb.py results/2026_xx_xx/results.csv
    python upload_wandb.py results/2026_xx_xx/results.csv --dry-run
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import wandb
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / "rwml-autocegar" / ".env")


def main():
    parser = argparse.ArgumentParser(description="Upload TimeEval results to W&B")
    parser.add_argument("results_csv", type=str, help="Path to TimeEval results.csv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--project", type=str, default="rwml-autocegar")
    parser.add_argument("--entity", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)
    print(f"Loaded {len(df)} results from {args.results_csv}")
    print(f"Columns: {list(df.columns)}")
    print(df.head())

    if args.dry_run:
        print("\nDRY RUN — would create the following W&B runs:")
        for _, row in df.iterrows():
            collection = row.get("collection_name", row.get("dataset_name", "unknown"))
            dataset = row.get("dataset_name", "unknown")
            roc = row.get("ROC_AUC", float("nan"))
            pr = row.get("PR_AUC", row.get("AVERAGE_PRECISION", float("nan")))
            print(f"  group={collection}  name=deepant_{dataset}  AUC-ROC={roc:.4f}  AUC-PR={pr:.4f}")
        return

    # detect metric columns
    metric_cols = [c for c in df.columns if any(
        m in c for m in ["ROC_AUC", "PR_AUC", "AVERAGE_PRECISION", "FIXED_RANGE",
                         "train_main_time", "execute_main_time"]
    )]
    meta_cols = [c for c in df.columns if c not in metric_cols and c != "algo_name"]

    print(f"\nMetric columns: {metric_cols}")
    print(f"Meta columns: {meta_cols}")
    print(f"\nUploading {len(df)} runs to W&B project={args.project}...")

    for i, row in df.iterrows():
        collection = row.get("collection_name", "unknown")
        dataset = row.get("dataset_name", "unknown")
        run_name = f"deepant_{dataset}"

        config = {col: row[col] for col in meta_cols if pd.notna(row.get(col))}
        config["method"] = "timeeval_deepant"

        metrics = {}
        for col in metric_cols:
            val = row.get(col)
            if pd.notna(val):
                clean_name = col.replace(" ", "_")
                metrics[f"eval/{clean_name}"] = float(val)

        run = wandb.init(
            project=args.project,
            entity=args.entity,
            name=run_name,
            group=collection,
            config=config,
            tags=["timeeval", "deepant", "vanilla", collection],
            reinit=True,
        )
        run.log(metrics)
        run.finish()
        print(f"[{i+1}/{len(df)}] {run_name}: {metrics}")

    print(f"\nDone. {len(df)} runs uploaded to W&B.")


if __name__ == "__main__":
    main()
