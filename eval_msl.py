"""MSL evaluation harness for the residual-CEGAR DeepAnT pipeline.

Establishes the vanilla DeepAnT MSL control (AUC-ROC ~= 0.679) and, once Luis's
notebook lands and the placeholder formulas are finalized, the residual-CEGAR
comparison (``--mode cegar``).

Data layout (TimeEval format, as on Bridges-2 under ``experiments/data``):
    each dataset is a CSV with columns: ``timestamp, value_0, ..., is_anomaly``.
    A ``*.train.csv`` (training, usually anomaly-free) and a ``*.test.csv``
    (scored) pair per MSL channel.

This harness is intentionally dependency-light: it reimplements TimeEval's
reverse-windowing locally so it runs without the full ``timeeval`` package, but
will use the real ``ReverseWindowing`` if importable (set ``--use-timeeval``).
"""
import argparse
from typing import Tuple

import numpy as np
import pandas as pd

from config import RWMLConfig, init_wandb
from deepant.dataset import TimeSeries
from deepant.detector import Detector
from train_rwml import RWMLPredictor

BASELINE_AUC = 0.679  # vanilla DeepAnT MSL control to reproduce


def load_timeeval_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a TimeEval-format CSV -> (values [T, C], labels [T])."""
    df = pd.read_csv(path)
    # column 0 is the index/timestamp; last column is the label (is_anomaly).
    label_col = df.columns[-1]
    labels = df[label_col].to_numpy().astype(int)
    values = df.iloc[:, 1:-1].to_numpy().astype(np.float32)
    return values, labels


def reverse_windowing(scores: np.ndarray, window_size: int, use_timeeval: bool = False) -> np.ndarray:
    """Map per-window scores back to per-timestep scores (rolling mean).

    Mirrors ``timeeval.utils.window.ReverseWindowing``: output length is
    ``len(scores) + window_size - 1`` and each timestep gets the mean of every
    window score covering it.
    """
    if use_timeeval:
        from timeeval.utils.window import ReverseWindowing
        return ReverseWindowing(window_size=window_size).fit_transform(scores)

    n = len(scores)
    out_len = n + window_size - 1
    acc = np.zeros(out_len, dtype=np.float64)
    cnt = np.zeros(out_len, dtype=np.float64)
    for i, s in enumerate(scores):
        acc[i:i + window_size] += s
        cnt[i:i + window_size] += 1
    cnt[cnt == 0] = 1
    return acc / cnt


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    m = min(len(labels), len(scores))
    labels, scores = labels[:m], scores[:m]
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    if labels.max() == labels.min():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def run_eval(train_csv: str, test_csv: str, cfg: RWMLConfig, use_cegar: bool,
             use_timeeval: bool = False, wandb_run=None) -> float:
    train_vals, _ = load_timeeval_csv(train_csv)
    test_vals, test_labels = load_timeeval_csv(test_csv)
    n_channels = train_vals.shape[1]

    dcfg, ccfg = cfg.deepant, cfg.cegar
    ccfg.enabled = use_cegar

    # train / validation split on the training series
    split = int(dcfg.split * len(train_vals))
    train_ds = TimeSeries(train_vals[:split], window_length=dcfg.window,
                          prediction_length=dcfg.pred_window, overlap=1)
    valid_ds = TimeSeries(train_vals[split:], window_length=dcfg.window,
                          prediction_length=dcfg.pred_window)
    test_ds = TimeSeries(test_vals, window_length=dcfg.window,
                         prediction_length=dcfg.pred_window)

    predictor = RWMLPredictor(dcfg, ccfg, in_channels=n_channels, wandb_run=wandb_run)
    predictor.train(train_ds, valid_ds, save_path=cfg.model_output)

    predicted = predictor.predict(test_ds)
    window_scores = Detector().detect(predicted, test_ds)
    point_scores = reverse_windowing(window_scores, dcfg.window + dcfg.pred_window, use_timeeval)

    auc = _roc_auc(test_labels, point_scores)
    print(f"\n=== MSL eval ({'CEGAR' if use_cegar else 'baseline'}) ===")
    print(f"AUC-ROC: {auc:.4f}   (baseline reference: {BASELINE_AUC})")
    if not use_cegar and not np.isnan(auc):
        print(f"delta vs reference baseline: {auc - BASELINE_AUC:+.4f}")
    if wandb_run is not None:
        try:
            wandb_run.log({"eval/auc_roc": auc, "eval/baseline_ref": BASELINE_AUC})
        except Exception:
            pass
    return auc


def parse_args():
    p = argparse.ArgumentParser(description="MSL evaluation for residual-CEGAR DeepAnT")
    p.add_argument("--train-csv", required=True, help="TimeEval-format training CSV")
    p.add_argument("--test-csv", required=True, help="TimeEval-format test CSV")
    p.add_argument("--mode", choices=["baseline", "cegar"], default="baseline")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--pred-window", type=int, default=None)
    p.add_argument("--use-timeeval", action="store_true", help="use real timeeval ReverseWindowing")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = RWMLConfig()
    if args.epochs is not None:
        cfg.deepant.epochs = args.epochs
    if args.window is not None:
        cfg.deepant.window = args.window
    if args.pred_window is not None:
        cfg.deepant.pred_window = args.pred_window
    if args.no_wandb:
        cfg.wandb.enabled = False
    cfg.wandb.name = cfg.wandb.name or f"deepant_msl_{args.mode}"

    wandb_run = init_wandb(cfg)
    try:
        run_eval(args.train_csv, args.test_csv, cfg, use_cegar=(args.mode == "cegar"),
                 use_timeeval=args.use_timeeval, wandb_run=wandb_run)
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass


if __name__ == "__main__":
    main()
