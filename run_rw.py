"""Run RW-1 / RW-2 / RW-AutoCEGAR experiments on a single dataset.

Usage
-----
# RW-1
python run_rw.py --method rw1 --data data/MSL_1.csv

# RW-2
python run_rw.py --method rw2 --data data/MSL_1.csv

# RW-AutoCEGAR
python run_rw.py --method rw_cegar --data data/MSL_1.csv

# Compare all three + DeepAnT baseline
python run_rw.py --method all --data data/MSL_1.csv

CSV format (TimeEval): timestamp, feature_0, ..., feature_N, is_anomaly
The entire series is used for training (RW-ML trains on train+test).
Test labels are used only for AUC evaluation at the end.
"""
import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import DeepAnTConfig, CegarConfig, RWMLConfig
from rw import RW1Trainer, RW2Trainer, RWCegarTrainer


def load_csv(path: str):
    df = pd.read_csv(path)
    labels = df.iloc[:, -1].to_numpy().astype(int)
    values = df.iloc[:, 1:-1].to_numpy().astype(np.float32)
    return values, labels


def auc_roc(labels, scores):
    scores = np.nan_to_num(scores, nan=0.0)
    if labels.max() == labels.min():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def run_method(method: str, X: np.ndarray, labels: np.ndarray,
               dcfg: DeepAnTConfig, ccfg: CegarConfig,
               model_dir: str, wandb_run=None) -> float:
    save_path = os.path.join(model_dir, f"{method}_model.pt")

    if method == "rw1":
        trainer = RW1Trainer(dcfg, in_channels=X.shape[1],
                             alpha=dcfg_alpha(dcfg),
                             correction_lr=1e-2,
                             wandb_run=wandb_run)
    elif method == "rw2":
        trainer = RW2Trainer(dcfg, in_channels=X.shape[1],
                             alpha=dcfg_alpha(dcfg),
                             correction_lr=1e-2,
                             wandb_run=wandb_run)
    elif method == "rw_cegar":
        trainer = RWCegarTrainer(dcfg, in_channels=X.shape[1],
                                 alpha=dcfg_alpha(dcfg),
                                 correction_lr=1e-2,
                                 lam=ccfg.lam,
                                 k=ccfg.k,
                                 tau=ccfg.tau,
                                 wandb_run=wandb_run)
    else:
        raise ValueError(f"Unknown method: {method}")

    trainer.train(X, save_path)
    scores = trainer.anomaly_score(X)
    auc = auc_roc(labels, scores)
    print(f"\n{'─'*50}")
    print(f"[{method.upper():12}]  AUC-ROC = {auc:.4f}")
    print(f"{'─'*50}\n")
    return auc


def dcfg_alpha(dcfg):
    """Default correction regularisation weight (correction_rate in thesis)."""
    return getattr(dcfg, "rw_alpha", 1e-2)


def parse_args():
    p = argparse.ArgumentParser(description="RW-ML experiment runner")
    p.add_argument("--method", choices=["rw1", "rw2", "rw_cegar", "all"],
                   default="all")
    p.add_argument("--data", required=True, help="CSV (TimeEval format)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--window", type=int, default=45)
    p.add_argument("--pred-window", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--correction-lr", type=float, default=1e-2)
    p.add_argument("--alpha", type=float, default=1e-2,
                   help="L1 regularisation weight on correction (correction_rate)")
    p.add_argument("--lam", type=float, default=1.0, help="CEGAR gate strength")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    X, labels = load_csv(args.data)
    print(f"Loaded: {args.data}  shape={X.shape}  anomalies={labels.sum()}/{len(labels)}")

    dcfg = DeepAnTConfig(
        window=args.window, pred_window=args.pred_window,
        lr=args.lr, batch_size=args.batch_size, epochs=args.epochs,
    )
    # attach rw_alpha so dcfg_alpha() can read it
    dcfg.rw_alpha = args.alpha

    ccfg = CegarConfig(lam=args.lam)

    os.makedirs(args.output_dir, exist_ok=True)

    methods = ["rw1", "rw2", "rw_cegar"] if args.method == "all" else [args.method]
    results = {}
    for m in methods:
        results[m] = run_method(m, X, labels, dcfg, ccfg, args.output_dir)

    if len(results) > 1:
        print("\n══ Summary ══════════════════════════════════")
        for m, auc in results.items():
            print(f"  {m:<14} AUC-ROC = {auc:.4f}")
        print("═════════════════════════════════════════════")


if __name__ == "__main__":
    main()
