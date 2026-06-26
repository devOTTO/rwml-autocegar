"""Run vanilla DeepAnT on all 200 TSB-AD-M time series and log to W&B.

Each CSV is a single file with the train/test boundary encoded in the filename
as ``_tr_NNN_``.  The first NNN rows are anomaly-free (training); the remainder
is the test portion (may contain anomalies).

Usage:
    # Dry-run: validate data loading, shapes, splits — no training
    python run_all_tsb.py --dry-run

    # Full run (all 200 series, 500 epochs each)
    python run_all_tsb.py

    # Quick smoke test (first 3 series, 5 epochs)
    python run_all_tsb.py --limit 3 --epochs 5

    # Specific dataset source only
    python run_all_tsb.py --source MSL
"""
import argparse
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from scipy.signal import find_peaks

from config import RWMLConfig, init_wandb
from deepant.dataset import TimeSeries
from deepant.detector import Detector
from train_rwml import RWMLPredictor
from eval_msl import reverse_windowing

DATA_DIR = Path("/ocean/projects/cis260190p/yhwang2/data/TSB-AD-M/TSB-AD-M")
RESULTS_DIR = Path("results/tsb_ad_m")

# Afonso paper Table 5.3 + run.py default_params()
PAPER_DEFAULTS = {
    "epochs": 500,
    "lr": 1e-3,
    "batch_size": 128,         # Table 5.3
    "pred_window": 50,         # run.py default_params()
    "split": 0.8,
    "early_stopping_delta": 0.05,
    "early_stopping_patience": 10,
    "random_state": 42,
    # window_size: determined per-series by period_size_heuristic()
}

# ──────────────────────────────────────────────────────────────── helpers


def period_size_heuristic(values: np.ndarray, factor: float = 0.5,
                          fb_value: int = 50) -> int:
    """Replicate TimeEval's PeriodSizeHeuristic(factor=0.5, fb_value=50).

    Detects the dominant period via autocorrelation on the first channel,
    then returns int(period * factor).  Falls back to fb_value if no clear
    period is found.
    """
    x = values[:, 0].astype(np.float64)
    x = x - x.mean()
    n = len(x)
    if n < 10:
        return fb_value
    acf = np.correlate(x, x, mode="full")[n - 1:]  # one-sided
    acf = acf / (acf[0] + 1e-12)                    # normalize
    # find first peak after lag 0 (skip lag 0 and initial decay)
    min_lag = max(2, n // 200)
    max_lag = n // 2
    peaks, properties = find_peaks(acf[min_lag:max_lag], height=0.1)
    if len(peaks) == 0:
        return fb_value
    period = int(peaks[0] + min_lag)
    window = int(period * factor)
    if window < fb_value:
        return fb_value
    return window


def parse_meta(filename: str) -> Dict:
    """Extract metadata from TSB-AD-M filename convention.

    Example: ``002_MSL_id_1_Sensor_tr_500_1st_900.csv``
      -> {idx: 2, source: "MSL", series_id: 1, domain: "Sensor",
          train_size: 500, first_anomaly: 900}
    """
    m = re.match(
        r"(\d+)_(.+?)_id_(\d+)_(.+?)_tr_(\d+)_1st_(\d+)\.csv", filename
    )
    if not m:
        raise ValueError(f"Cannot parse filename: {filename}")
    return {
        "idx": int(m.group(1)),
        "source": m.group(2),
        "series_id": int(m.group(3)),
        "domain": m.group(4),
        "train_size": int(m.group(5)),
        "first_anomaly": int(m.group(6)),
    }


def load_single_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a TSB-AD-M CSV using the Afonso convention (index_col=0).

    Returns (values [N, C], labels [N]) where C = number of feature columns.
    """
    df = pd.read_csv(path, index_col=0)
    labels = df.iloc[:, -1].to_numpy().astype(int)
    values = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    return values, labels


def compute_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Compute AUC-ROC and AUC-PR, handling edge cases."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    m = min(len(labels), len(scores))
    labels, scores = labels[:m], scores[:m]
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    result = {"auc_roc": float("nan"), "auc_pr": float("nan")}
    if labels.max() == labels.min():
        return result
    try:
        result["auc_roc"] = float(roc_auc_score(labels, scores))
        result["auc_pr"] = float(average_precision_score(labels, scores))
    except ValueError:
        pass
    return result


# ──────────────────────────────────────────────────────────── dry-run


def dry_run_one(csv_path: Path, cfg: RWMLConfig) -> Dict:
    """Validate one CSV without training: load, split, check shapes."""
    meta = parse_meta(csv_path.name)
    values, labels = load_single_csv(csv_path)
    tr = meta["train_size"]
    n_rows, n_features = values.shape
    dcfg = cfg.deepant

    train_vals = values[:tr]

    # compute per-series window size via period heuristic
    window = period_size_heuristic(train_vals)
    min_window = window + dcfg.pred_window

    test_vals = values[tr:]
    test_labels = labels[tr:]

    # train/val split within training portion
    val_split = int(dcfg.split * len(train_vals))
    n_train = val_split
    n_val = len(train_vals) - val_split

    # window counts
    train_windows = max(0, n_train - min_window + 1)
    val_windows = max(0, n_val - min_window + 1)
    test_windows = max(0, len(test_vals) - min_window + 1)

    anomaly_count = int(test_labels.sum())
    anomaly_ratio = anomaly_count / max(len(test_labels), 1)

    issues = []
    if tr >= n_rows:
        issues.append(f"train_size ({tr}) >= total rows ({n_rows}), no test data")
    if train_windows == 0:
        issues.append(f"train portion too small for windowing ({n_train} < {min_window})")
    if val_windows == 0:
        issues.append(f"val portion too small for windowing ({n_val} < {min_window})")
    if test_windows == 0:
        issues.append(f"test portion too small for windowing ({len(test_vals)} < {min_window})")
    if anomaly_count == 0:
        issues.append("no anomalies in test portion (AUC will be NaN)")
    if anomaly_count == len(test_labels):
        issues.append("all test points are anomalies (AUC will be NaN)")

    status = "WARN" if issues else "OK"

    return {
        **meta,
        "n_rows": n_rows,
        "n_features": n_features,
        "window_size": window,
        "train_rows": tr,
        "val_rows": len(train_vals) - val_split,
        "test_rows": len(test_vals),
        "train_windows": train_windows,
        "val_windows": val_windows,
        "test_windows": test_windows,
        "anomaly_count": anomaly_count,
        "anomaly_ratio": round(anomaly_ratio, 4),
        "status": status,
        "issues": "; ".join(issues) if issues else "",
    }


def dry_run_all(csvs: List[Path], cfg: RWMLConfig):
    """Dry-run over all CSVs: validate and print summary table."""
    print(f"\n{'='*80}")
    print(f"DRY RUN — validating {len(csvs)} time series")
    print(f"Data dir: {DATA_DIR}")
    print(f"Hyperparams (Afonso paper): epochs={cfg.deepant.epochs}, lr={cfg.deepant.lr}, "
          f"batch_size={cfg.deepant.batch_size}, "
          f"pred_window={cfg.deepant.pred_window}, split={cfg.deepant.split}, "
          f"window=PeriodSizeHeuristic(factor=0.5, fb=50)")
    print(f"{'='*80}\n")

    results = []
    errors = []
    for i, csv_path in enumerate(csvs):
        try:
            r = dry_run_one(csv_path, cfg)
            results.append(r)
            mark = "!!" if r["status"] == "WARN" else "ok"
            print(
                f"[{i+1:3d}/{len(csvs)}] {mark}  {csv_path.name:<55s}  "
                f"({r['n_rows']:>6d},{r['n_features']:>3d})  "
                f"w={r['window_size']:<4d} "
                f"tr/va/te={r['train_rows']}/{r['val_rows']}/{r['test_rows']}  "
                f"anom={r['anomaly_count']}({r['anomaly_ratio']:.1%})"
            )
            if r["issues"]:
                print(f"         -> {r['issues']}")
        except Exception as e:
            errors.append((csv_path.name, str(e)))
            print(f"[{i+1:3d}/{len(csvs)}] ERR {csv_path.name}: {e}")

    # summary
    df = pd.DataFrame(results)
    ok_count = (df["status"] == "OK").sum()
    warn_count = (df["status"] == "WARN").sum()

    print(f"\n{'='*80}")
    print(f"DRY RUN SUMMARY")
    print(f"{'='*80}")
    print(f"Total files:       {len(csvs)}")
    print(f"  OK:              {ok_count}")
    print(f"  WARN:            {warn_count}")
    print(f"  ERROR:           {len(errors)}")

    if len(results) > 0:
        print(f"\nPer-source breakdown:")
        source_summary = df.groupby("source").agg(
            count=("idx", "size"),
            total_rows=("n_rows", "sum"),
            mean_features=("n_features", "mean"),
            warns=("status", lambda x: (x == "WARN").sum()),
        )
        print(source_summary.to_string())

        print(f"\nEstimated W&B runs:  {ok_count}")
        print(f"W&B project:         {cfg.wandb.project}")
        print(f"W&B entity:          {cfg.wandb.entity}")
        print(f"W&B groups:          {df['source'].nunique()} (one per dataset source)")

    if errors:
        print(f"\nErrors:")
        for name, err in errors:
            print(f"  {name}: {err}")

    # save dry-run report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "dry_run_report.csv"
    df.to_csv(report_path, index=False)
    print(f"\nFull report saved to: {report_path}")


# ──────────────────────────────────────────────────────────── full run


def run_one(csv_path: Path, cfg: RWMLConfig) -> Dict:
    """Train + evaluate vanilla DeepAnT on one TSB-AD-M time series."""
    meta = parse_meta(csv_path.name)
    values, labels = load_single_csv(csv_path)
    tr = meta["train_size"]
    n_channels = values.shape[1]

    train_vals = values[:tr]
    test_vals = values[tr:]
    test_labels = labels[tr:]

    if len(test_vals) == 0:
        print(f"  SKIP: no test data (train_size={tr} >= total={len(values)})")
        return {**meta, "auc_roc": float("nan"), "auc_pr": float("nan"),
                "status": "SKIP", "elapsed_s": 0}

    # per-series window via heuristic (Afonso: PeriodSizeHeuristic(0.5, fb=50))
    window = period_size_heuristic(train_vals)

    dcfg = cfg.deepant
    dcfg.window = window  # override per series
    ccfg = cfg.cegar
    ccfg.enabled = False

    min_window = window + dcfg.pred_window

    # wandb: one run per time series, grouped by source
    cfg.wandb.group = meta["source"]
    cfg.wandb.name = f"deepant_{meta['source']}_id_{meta['series_id']}"
    wandb_run = init_wandb(cfg)

    if wandb_run:
        wandb_run.config.update({
            "dataset_source": meta["source"],
            "dataset_series_id": meta["series_id"],
            "dataset_domain": meta["domain"],
            "dataset_idx": meta["idx"],
            "dataset_filename": csv_path.name,
            "n_rows": len(values),
            "n_features": n_channels,
            "train_rows": tr,
            "test_rows": len(test_vals),
            "anomaly_count": int(test_labels.sum()),
            "window_size": window,
        })

    # train / validation split within training portion
    split_pt = int(dcfg.split * len(train_vals))

    if split_pt < min_window or (len(train_vals) - split_pt) < min_window:
        print(f"  SKIP: train portion too small for windowing (w={window})")
        if wandb_run:
            wandb_run.finish()
        return {**meta, "auc_roc": float("nan"), "auc_pr": float("nan"),
                "status": "SKIP_SMALL", "elapsed_s": 0}

    train_ds = TimeSeries(train_vals[:split_pt], window, dcfg.pred_window, overlap=1)
    valid_ds = TimeSeries(train_vals[split_pt:], window, dcfg.pred_window)

    if len(test_vals) < min_window:
        print(f"  SKIP: test portion too small for windowing (w={window})")
        if wandb_run:
            wandb_run.finish()
        return {**meta, "auc_roc": float("nan"), "auc_pr": float("nan"),
                "status": "SKIP_SMALL", "elapsed_s": 0}

    test_ds = TimeSeries(test_vals, window, dcfg.pred_window)
    print(f"  window={window}  pred_window={dcfg.pred_window}  "
          f"train={len(train_ds)}  val={len(valid_ds)}  test={len(test_ds)}")

    # model save path (per series)
    model_dir = RESULTS_DIR / meta["source"] / f"id_{meta['series_id']}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(model_dir / "model.pt")

    t0 = time.time()

    # train
    predictor = RWMLPredictor(dcfg, ccfg, n_channels, wandb_run=wandb_run)
    predictor.train(train_ds, valid_ds, save_path=model_path)

    # execute
    predictor.load(model_path)
    predicted = predictor.predict(test_ds)
    window_scores = Detector().detect(predicted, test_ds)
    point_scores = reverse_windowing(window_scores, min_window)

    elapsed = time.time() - t0

    # metrics
    metrics = compute_metrics(test_labels, point_scores)

    # log final metrics to wandb
    if wandb_run:
        wandb_run.summary.update({
            "eval/auc_roc": metrics["auc_roc"],
            "eval/auc_pr": metrics["auc_pr"],
            "eval/elapsed_s": elapsed,
        })
        wandb_run.finish()

    # save scores locally
    np.save(model_dir / "scores.npy", point_scores)

    return {
        **meta,
        **metrics,
        "window_size": window,
        "status": "OK",
        "elapsed_s": round(elapsed, 1),
    }


# ──────────────────────────────────────────────────────────── main


def collect_csvs(source: Optional[str] = None) -> List[Path]:
    """Gather and sort CSV files, optionally filtered by source."""
    csvs = sorted(DATA_DIR.glob("*.csv"))
    if source:
        csvs = [c for c in csvs if f"_{source}_" in c.name]
    return csvs


def main():
    parser = argparse.ArgumentParser(
        description="Run vanilla DeepAnT on TSB-AD-M and log to W&B"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate data only, no training")
    parser.add_argument("--source", type=str, default=None,
                        help="Filter by dataset source (e.g. MSL, SMD, GHL)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N files")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)    # Table 5.3
    parser.add_argument("--pred-window", type=int, default=50)    # run.py default_params
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = RWMLConfig()
    cfg.deepant.epochs = args.epochs
    cfg.deepant.lr = args.lr
    cfg.deepant.batch_size = args.batch_size
    cfg.deepant.pred_window = args.pred_window
    cfg.deepant.split = args.split
    cfg.cegar.enabled = False
    if args.no_wandb:
        cfg.wandb.enabled = False

    csvs = collect_csvs(args.source)
    if args.limit:
        csvs = csvs[: args.limit]

    if not csvs:
        print(f"No CSV files found in {DATA_DIR}")
        sys.exit(1)

    if args.dry_run:
        dry_run_all(csvs, cfg)
        return

    # full run
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for i, csv_path in enumerate(csvs):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(csvs)}] {csv_path.name}")
        print(f"{'='*70}")
        try:
            result = run_one(csv_path, cfg)
            results.append(result)
            print(f"  -> AUC-ROC={result['auc_roc']:.4f}  "
                  f"AUC-PR={result['auc_pr']:.4f}  "
                  f"time={result['elapsed_s']}s  "
                  f"status={result['status']}")
        except Exception as e:
            print(f"  -> FAILED: {e}")
            results.append({"idx": i, "source": csv_path.name,
                            "status": "ERROR", "error": str(e)})

    # save summary
    summary_path = RESULTS_DIR / "summary.csv"
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"\n{'='*70}")
    print(f"DONE. {len(results)} series processed. Summary: {summary_path}")

    # print aggregate table
    df = pd.DataFrame(results)
    if "auc_roc" in df.columns:
        agg = df.groupby("source").agg(
            n=("idx", "size"),
            auc_roc_mean=("auc_roc", "mean"),
            auc_pr_mean=("auc_pr", "mean"),
        ).round(4)
        print(f"\nPer-source results:")
        print(agg.to_string())


if __name__ == "__main__":
    main()
