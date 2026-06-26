#!/usr/bin/env python3
"""Run vanilla DeepAnT on all TSB-AD-M datasets using TimeEval framework.

Runs datasets in chunks to prevent OOM — each chunk gets a fresh TimeEval
instance so memory is fully released between chunks.

Params match Afonso's paper Table 5.3: batch_size=128, lr=1e-3, epochs=500,
prediction_window_size=50, window_size=PeriodSizeHeuristic(0.5, fb=50).

Usage:
    python exp_deepant_all.py                  # full run (199 datasets)
    python exp_deepant_all.py --collection MSL # one collection only
    python exp_deepant_all.py --chunk-size 20  # 20 datasets per chunk
"""
import argparse
import gc
import sys
from pathlib import Path

import torch
from timeeval import TimeEval, MultiDatasetManager, DefaultMetrics, InputDimensionality
from deepant.run import AlgorithmHelper as deepant
from experiments import params as experiments_params

RESULTS_DIR = Path("results")


def configure_algorithm(algorithm):
    params = experiments_params.default_params()
    params.update(algorithm.default_params())
    params["batch_size"] = 128  # Paper Table 5.3
    return algorithm.config(params)


def run_chunk(dm, datasets, chunk_idx, total_chunks):
    print(f"\n{'='*70}")
    print(f"CHUNK {chunk_idx+1}/{total_chunks} — {len(datasets)} datasets")
    print(f"{'='*70}")
    for d in datasets:
        print(f"  {d}")

    algorithms = [configure_algorithm(deepant)]

    timeeval = TimeEval(
        dm, datasets, algorithms,
        repetitions=1,
        metrics=[
            DefaultMetrics.ROC_AUC,
            DefaultMetrics.PR_AUC,
            DefaultMetrics.AVERAGE_PRECISION,
            DefaultMetrics.FIXED_RANGE_PR_AUC,
        ],
    )

    timeeval.run()

    results = timeeval.get_results(aggregated=False)
    print(f"Chunk {chunk_idx+1} done: {len(results)} results")

    del timeeval, algorithms
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="DeepAnT on TSB-AD-M (TimeEval)")
    parser.add_argument("--collection", type=str, default=None,
                        help="Filter by collection (e.g. MSL, SMD)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of datasets")
    parser.add_argument("--chunk-size", type=int, default=30,
                        help="Datasets per chunk (memory management)")
    args = parser.parse_args()

    dm = MultiDatasetManager([Path("experiments/data")])

    if args.collection:
        datasets = dm.select(
            collection=args.collection,
            input_dimensionality=InputDimensionality.MULTIVARIATE,
        )
    else:
        datasets = dm.select(
            input_dimensionality=InputDimensionality.MULTIVARIATE,
        )

    if args.limit:
        datasets = datasets[:args.limit]

    print(f"Total datasets: {len(datasets)}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Params: batch_size=128, lr=1e-3, epochs=500, pred_window=50, window=heuristic")

    chunks = [datasets[i:i+args.chunk_size] for i in range(0, len(datasets), args.chunk_size)]
    print(f"Chunks: {len(chunks)}")

    all_results = []
    for i, chunk in enumerate(chunks):
        try:
            results = run_chunk(dm, chunk, i, len(chunks))
            all_results.append(results)
        except Exception as e:
            print(f"Chunk {i+1} FAILED: {e}")

    if all_results:
        import pandas as pd
        combined = pd.concat(all_results, ignore_index=True)
        out_path = "results_deepant_tsb.csv"
        combined.to_csv(out_path, index=False)
        print(f"\n{'='*70}")
        print(f"ALL DONE — {len(combined)} results saved to {out_path}")
        print(f"{'='*70}")

        if "ROC_AUC" in combined.columns:
            agg = combined.groupby("collection_name").agg(
                n=("dataset_name", "size"),
                ROC_AUC=("ROC_AUC", "mean"),
                AVERAGE_PRECISION=("AVERAGE_PRECISION", "mean"),
            ).round(4)
            print(f"\n{agg.to_string()}")


if __name__ == "__main__":
    main()
