"""Standalone DeepAnT runner (no TimeEval required).

Wraps ``deepant.algorithm.main_cli`` and adds helpers for the common
train → execute workflow on a single CSV.

Quick start
-----------
# 1. train on MSL_1.train.csv, save model to results/
python run_deepant.py train --data data/MSL_1.train.csv

# 2. score MSL_1.test.csv with the trained model
python run_deepant.py execute --data data/MSL_1.test.csv

# 3. train + execute in one shot
python run_deepant.py run --train data/MSL_1.train.csv --test data/MSL_1.test.csv

# 4. see all options
python run_deepant.py train --help
"""
import argparse
import sys
from pathlib import Path

from deepant.algorithm import main_cli, ExecutionType, Config, train, execute


def _shared_args(p: argparse.ArgumentParser):
    p.add_argument("--model", default="results/model.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--window", type=int, default=45)
    p.add_argument("--pred-window", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--split", type=float, default=0.8)
    p.add_argument("--es-delta", type=float, default=0.05)
    p.add_argument("--es-patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)


def _make_config(args, mode, data_path, output_path="results/anomalies.csv"):
    exe_type = ExecutionType.TRAIN if mode == "train" else ExecutionType.EXECUTE
    return Config({
        "dataInput": data_path,
        "dataOutput": output_path,
        "modelInput": args.model,
        "modelOutput": args.model,
        "executionType": exe_type,
        "customParameters": {
            "epochs": args.epochs,
            "window_size": args.window,
            "prediction_window_size": args.pred_window,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "split": args.split,
            "early_stopping_delta": args.es_delta,
            "early_stopping_patience": args.es_patience,
            "random_state": args.seed,
        },
    })


def cmd_train(args):
    cfg = _make_config(args, "train", args.data)
    print(cfg)
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    train(cfg)


def cmd_execute(args):
    cfg = _make_config(args, "execute", args.data, args.output)
    print(cfg)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    execute(cfg)


def cmd_run(args):
    """Train then execute in one shot."""
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("─── TRAIN ───────────────────────────────")
    cfg_train = _make_config(args, "train", args.train)
    train(cfg_train)

    print("\n─── EXECUTE ─────────────────────────────")
    cfg_exec = _make_config(args, "execute", args.test, args.output)
    scores = execute(cfg_exec)
    print(f"\nDone.  {len(scores)} window scores written to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        prog="run_deepant",
        description="Standalone DeepAnT runner",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── train ──
    p_train = sub.add_parser("train", help="Train on a CSV")
    p_train.add_argument("--data", required=True, help="Training CSV (TimeEval format)")
    _shared_args(p_train)
    p_train.set_defaults(func=cmd_train)

    # ── execute ──
    p_exec = sub.add_parser("execute", help="Score a CSV with a trained model")
    p_exec.add_argument("--data", required=True, help="Test CSV (TimeEval format)")
    p_exec.add_argument("--output", default="results/anomalies.csv")
    _shared_args(p_exec)
    p_exec.set_defaults(func=cmd_execute)

    # ── run (train + execute) ──
    p_run = sub.add_parser("run", help="Train then execute in one shot")
    p_run.add_argument("--train", required=True, help="Training CSV")
    p_run.add_argument("--test", required=True, help="Test CSV")
    p_run.add_argument("--output", default="results/anomalies.csv")
    _shared_args(p_run)
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
