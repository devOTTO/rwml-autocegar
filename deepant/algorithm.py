"""DeepAnT standalone train/execute entry point.

Adapted from ``TimeEval-algorithms/deepant/algorithm.py``:
  - timeeval dependency removed (ExecutionType replaced with plain strings)
  - imports updated for our flat ``deepant/`` package layout
  - helper path creation made robust (os.makedirs)

Usage (standalone, no TimeEval):
    python -m deepant.algorithm --mode train  --data data/train.csv --model results/model.pt
    python -m deepant.algorithm --mode execute --data data/test.csv  --model results/model.pt --output results/scores.csv
"""
import argparse
import json
import sys
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .dataset import TimeSeries
from .detector import Detector
from .predictor import Predictor
from .helper import retrieve_save_path

# ---- defaults (mirror original algorithm.py) ----
EPOCHS = 50
WINDOW = 45
PRED_WINDOW = 1
LR = 1e-3
WEIGHT_DECAY = 1e-6
TRAIN_SPLIT = 0.8
BATCH_SIZE = 64
EARLY_STOPPING_DELTA = 0.05
EARLY_STOPPING_PATIENCE = 10
RANDOM_STATE = 42


class ExecutionType(str, Enum):
    TRAIN = "train"
    EXECUTE = "execute"


class Config:
    def __init__(self, params: dict):
        self.dataInput = Path(params.get("dataInput", "data/dataset.csv"))
        self.dataOutput = Path(params.get("dataOutput", "results/anomalies.csv"))
        self.modelInput = Path(params.get("modelInput", "results/model.pt"))
        self.modelOutput = Path(params.get("modelOutput", "results/model.pt"))
        self.executionType = params.get("executionType", ExecutionType.TRAIN)

        cp = params.get("customParameters", {})
        self.epochs = cp.get("epochs", EPOCHS)
        self.window = cp.get("window_size", WINDOW)
        self.pred_window = cp.get("prediction_window_size", PRED_WINDOW)
        self.lr = cp.get("learning_rate", LR)
        self.batch_size = cp.get("batch_size", BATCH_SIZE)
        self.split = cp.get("split", TRAIN_SPLIT)
        self.early_stopping_delta = cp.get("early_stopping_delta", EARLY_STOPPING_DELTA)
        self.early_stopping_patience = cp.get("early_stopping_patience", EARLY_STOPPING_PATIENCE)
        self.random_state = cp.get("random_state", RANDOM_STATE)

    def __str__(self):
        return (
            f"Config(executionType={self.executionType}, dataInput={self.dataInput}, "
            f"window={self.window}, pred_window={self.pred_window}, epochs={self.epochs}, "
            f"lr={self.lr}, batch_size={self.batch_size})"
        )


def preprocess_data(config: Config) -> dict:
    """Load CSV → TimeSeries datasets.

    CSV format (TimeEval):
        col 0  = index / timestamp
        col 1…N-1 = feature values
        last col   = is_anomaly label (dropped during training)
    """
    ts_data = pd.read_csv(config.dataInput, index_col=0).iloc[:, :-1]  # drop label col
    print(f"Dataset: {config.dataInput}  shape={ts_data.shape}")
    channels = ts_data.shape[1]

    if config.executionType == ExecutionType.TRAIN:
        n_train = int(config.split * len(ts_data))
        train_ds = TimeSeries(ts_data.iloc[:n_train].values, window_length=config.window,
                              prediction_length=config.pred_window, overlap=1)
        valid_ds = TimeSeries(ts_data.iloc[n_train:].values, window_length=config.window,
                              prediction_length=config.pred_window)
        print(f"Train windows: {len(train_ds)}  Val windows: {len(valid_ds)}")
        return {"train": train_ds, "val": valid_ds, "n_channels": channels}

    else:  # EXECUTE
        test_ds = TimeSeries(ts_data.values, window_length=config.window,
                             prediction_length=config.pred_window)
        print(f"Test windows: {len(test_ds)}")
        return {"test": test_ds, "n_channels": channels}


def set_random_state(config: Config):
    import random
    random.seed(config.random_state)
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)
    torch.cuda.manual_seed_all(config.random_state)


def train(config: Config):
    print("\n=== PREPROCESSING ===")
    data = preprocess_data(config)
    set_random_state(config)

    predictor = Predictor(
        window=config.window, pred_window=config.pred_window,
        lr=config.lr, batch_size=config.batch_size, in_channels=data["n_channels"],
    )
    print(predictor.model)

    print("\n=== TRAINING ===")
    predictor.train(
        data["train"], data["val"],
        n_epochs=config.epochs,
        save_path=config.modelOutput,
        early_stopping_delta=config.early_stopping_delta,
        early_stopping_patience=config.early_stopping_patience,
    )
    print(f"Model saved to {config.modelOutput}")


def execute(config: Config) -> np.ndarray:
    print("\n=== PREPROCESSING ===")
    data = preprocess_data(config)
    set_random_state(config)

    predictor = Predictor(window=config.window, pred_window=config.pred_window,
                          in_channels=data["n_channels"])
    predictor.load(config.modelInput)
    print(predictor.model)

    print("\n=== PREDICTION ===")
    predicted = predictor.predict(data["test"])
    anomalies = Detector().detect(predicted, data["test"])

    out_path = retrieve_save_path(config.dataOutput, "anomalies.csv")
    anomalies.tofile(out_path, sep="\n")
    print(f"Scores saved to {out_path}  ({len(anomalies)} windows)")
    return anomalies


# ------------------------------------------------------------------ CLI
def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="DeepAnT standalone train/execute")
    p.add_argument("--mode", choices=["train", "execute"], required=True)
    p.add_argument("--data", required=True, help="Path to input CSV")
    p.add_argument("--model", default="results/model.pt", help="Model save/load path")
    p.add_argument("--output", default="results/anomalies.csv", help="Score output path (execute)")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--window", type=int, default=WINDOW)
    p.add_argument("--pred-window", type=int, default=PRED_WINDOW)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--split", type=float, default=TRAIN_SPLIT)
    p.add_argument("--es-delta", type=float, default=EARLY_STOPPING_DELTA)
    p.add_argument("--es-patience", type=int, default=EARLY_STOPPING_PATIENCE)
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    return p.parse_args(argv)


def main_cli(argv=None):
    args = _parse_args(argv)
    exe_type = ExecutionType.TRAIN if args.mode == "train" else ExecutionType.EXECUTE
    config = Config({
        "dataInput": args.data,
        "dataOutput": args.output,
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
    print(config)
    if exe_type == ExecutionType.TRAIN:
        train(config)
    else:
        execute(config)


if __name__ == "__main__":
    main_cli()
