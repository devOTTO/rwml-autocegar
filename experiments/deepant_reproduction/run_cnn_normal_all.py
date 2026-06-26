"""Run vanilla DeepAnT on all TSB-AD-M datasets using Afonso's standalone code.

v3: Uses run_cnn_normal.py style (window=50, pred_w=50, lr=1e-4, batch_size=64)
No TimeEval heuristic — matches Afonso's actual experiment scripts.

Usage:
    python run_cnn_normal_all.py                    # full run
    python run_cnn_normal_all.py --collection MSL   # one collection
    python run_cnn_normal_all.py --limit 3          # first 3
    python run_cnn_normal_all.py --dry-run           # validate only
"""
import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

from deepant.deepant.model import DeepAnTCNN
from deepant.deepant.dataset import TimeSeries
from deepant.deepant.predictor import Predictor
from deepant.deepant.detector import Detector
from timeeval.utils.window import ReverseWindowing, Method

DATA_DIR = Path("experiments/data")
RESULTS_DIR = Path("results_v3")

WINDOW = 50
PRED_W = 50
BATCH_SIZE = 64
LR = 1e-4
NUM_EPOCHS = 500
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def create_sliding_window(X, window, pred_w, step=1, shuffle=True):
    total_length = X.shape[2]
    channels = X.shape[1]
    num_samples = (total_length - window - pred_w) // step + 1
    new_stride = step * X.stride(2)

    X_train = X.as_strided(
        size=(num_samples, channels, window),
        stride=(new_stride, total_length, 1),
    )
    y_train = X.as_strided(
        size=(num_samples, channels, pred_w),
        stride=(new_stride, total_length, 1),
        storage_offset=window,
    )

    if shuffle:
        indices = torch.randperm(num_samples, device=X.device)
        return X_train, y_train, indices
    return X_train, y_train, torch.arange(num_samples, device=X.device)


def train_and_eval(collection, dataset_name, device):
    """Train + evaluate one dataset. Returns dict with metrics."""
    train_path = DATA_DIR / collection / f"{dataset_name}.train.csv"
    test_path = DATA_DIR / collection / f"{dataset_name}.test.csv"

    if not train_path.exists() or not test_path.exists():
        return {"status": "SKIP", "error": "file not found"}

    # load train data
    data = pd.read_csv(train_path)
    data = data.iloc[:, 1:-1]  # drop index col and is_anomaly
    in_channels = data.shape[1]

    split = 0.8
    train_samples = int(split * len(data))
    train_data = data.iloc[:train_samples].values
    val_data = data.iloc[train_samples:].values

    if train_data.shape[0] < WINDOW + PRED_W + 1:
        return {"status": "SKIP", "error": f"train too small ({train_data.shape[0]})"}
    if val_data.shape[0] < WINDOW + PRED_W + 1:
        return {"status": "SKIP", "error": f"val too small ({val_data.shape[0]})"}

    train_data = torch.from_numpy(train_data).float().permute(1, 0).unsqueeze(0).to(device)
    val_data = torch.from_numpy(val_data).float().permute(1, 0).unsqueeze(0).to(device)

    set_seed(SEED)

    model = DeepAnTCNN(WINDOW, PRED_W, in_channels, 128, 32, 2, 2, 1).to(device)
    loss_fn = nn.MSELoss()
    optimizer = Adam([{"params": model.parameters(), "lr": LR}])

    X_train, y_train, idx_train = create_sliding_window(train_data, WINDOW, PRED_W, step=1, shuffle=True)
    X_val, y_val, idx_val = create_sliding_window(val_data, WINDOW, PRED_W, step=1, shuffle=False)

    set_seed(SEED)

    best_val_loss = np.inf
    best_model = None
    counter_no_improvement = 0

    t0 = time.time()
    final_epoch = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for i in range(0, X_train.shape[0], BATCH_SIZE):
            optimizer.zero_grad()
            batch_indices = idx_train[i : i + BATCH_SIZE]
            X_batch = X_train[batch_indices, :, :]
            y_batch = y_train[batch_indices, :, :]
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        n_batches = max(X_train.shape[0] // BATCH_SIZE, 1)
        epoch_loss /= n_batches

        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for i in range(0, X_val.shape[0], BATCH_SIZE):
                X_batch = X_val[i : i + BATCH_SIZE, :, :]
                y_batch = y_val[i : i + BATCH_SIZE, :, :]
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                eval_loss += loss.item()
            n_val_batches = max(X_val.shape[0] // BATCH_SIZE, 1)
            eval_loss /= n_val_batches

        if eval_loss < best_val_loss:
            best_val_loss = eval_loss
            best_model = model.state_dict().copy()
            counter_no_improvement = 0
        else:
            counter_no_improvement += 1
        if counter_no_improvement >= 10:
            final_epoch = epoch + 1
            break
        final_epoch = epoch + 1

    train_time = time.time() - t0
    model.load_state_dict(best_model)

    # save model
    model_dir = RESULTS_DIR / collection
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{dataset_name}.pt"
    torch.save(model.state_dict(), model_path)

    # evaluate
    data_test = pd.read_csv(test_path, index_col=0)
    ts_data = data_test.iloc[:, :-1]
    y_true = data_test.iloc[:, -1].values

    if ts_data.shape[0] < WINDOW + PRED_W + 1:
        return {"status": "SKIP", "error": f"test too small ({ts_data.shape[0]})"}

    test_dataset = TimeSeries(ts_data.values, WINDOW, PRED_W)

    predictor = Predictor(WINDOW, PRED_W, LR, 1, in_channels=in_channels,
                          filter1_size=128, filter2_size=32)
    predictor.load(str(model_path))

    t1 = time.time()
    predictedY = predictor.predict(test_dataset)
    detector = Detector()
    anomalies = detector.detect(predictedY, test_dataset)

    def _post_deepant(scores):
        size = WINDOW + PRED_W
        return ReverseWindowing(window_size=size, reduction=Method.MEAN).fit_transform(scores)

    final_scores = _post_deepant(anomalies)
    exec_time = time.time() - t1

    # metrics
    m = min(len(y_true), len(final_scores))
    y_true = y_true[:m]
    final_scores = final_scores[:m]
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        prec, rec, _ = precision_recall_curve(y_true, final_scores)
        auc_pr = float(auc(rec, prec))
    except ValueError:
        auc_pr = float("nan")

    try:
        auc_roc = float(roc_auc_score(y_true, final_scores))
    except ValueError:
        auc_roc = float("nan")

    # cleanup
    del model, optimizer, X_train, y_train, X_val, y_val, train_data, val_data
    del predictedY, anomalies, detector, predictor
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "status": "OK",
        "auc_roc": round(auc_roc, 6),
        "auc_pr": round(auc_pr, 6),
        "epochs": final_epoch,
        "train_time": round(train_time, 1),
        "exec_time": round(exec_time, 1),
    }


def get_all_datasets():
    """Discover all datasets from experiments/data/datasets.csv."""
    ds_csv = DATA_DIR / "datasets.csv"
    df = pd.read_csv(ds_csv)
    pairs = list(zip(df["collection_name"], df["dataset_name"]))
    return sorted(pairs)


def main():
    parser = argparse.ArgumentParser(description="DeepAnT v3 — Afonso standalone style")
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Params: window={WINDOW}, pred_w={PRED_W}, lr={LR}, "
          f"batch_size={BATCH_SIZE}, epochs={NUM_EPOCHS}")

    datasets = get_all_datasets()
    if args.collection:
        datasets = [(c, d) for c, d in datasets if c == args.collection]
    if args.limit:
        datasets = datasets[: args.limit]

    print(f"Datasets: {len(datasets)}")

    if args.dry_run:
        for i, (c, d) in enumerate(datasets):
            train_path = DATA_DIR / c / f"{d}.train.csv"
            exists = "OK" if train_path.exists() else "MISSING"
            print(f"[{i+1:3d}/{len(datasets)}] {c}/{d}  {exists}")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for i, (collection, dataset_name) in enumerate(datasets):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(datasets)}] {collection}/{dataset_name}")
        print(f"{'='*60}")

        try:
            result = train_and_eval(collection, dataset_name, device)
            result["collection"] = collection
            result["dataset"] = dataset_name
            results.append(result)

            if result["status"] == "OK":
                print(f"  AUC-ROC={result['auc_roc']:.4f}  "
                      f"AUC-PR={result['auc_pr']:.4f}  "
                      f"epochs={result['epochs']}  "
                      f"time={result['train_time']}s")
            else:
                print(f"  {result['status']}: {result.get('error', '')}")
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"collection": collection, "dataset": dataset_name,
                            "status": "ERROR", "error": str(e)})

    # save results
    df = pd.DataFrame(results)
    summary_path = RESULTS_DIR / "summary_v3.csv"
    df.to_csv(summary_path, index=False)
    print(f"\n{'='*60}")
    print(f"DONE. {len(results)} datasets. Summary: {summary_path}")

    ok = df[df["status"] == "OK"]
    if len(ok) > 0:
        agg = ok.groupby("collection").agg(
            n=("dataset", "size"),
            AUC_ROC=("auc_roc", "mean"),
            AUC_PR=("auc_pr", "mean"),
        ).round(4)
        print(f"\n{agg.to_string()}")


if __name__ == "__main__":
    main()
