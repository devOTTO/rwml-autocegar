"""Convert TSB-AD-M single-CSV format to TimeEval's expected directory layout.

TimeEval expects:
    experiments/data/
    ├── datasets.csv                          # metadata index
    └── {collection_name}/
        ├── {dataset_name}.train.csv          # anomaly-free training portion
        └── {dataset_name}.test.csv           # test portion (with anomalies)

Each CSV keeps the same format: index_col=0, last column = "Label".

Usage:
    python prepare_tsb_data.py                # default paths
    python prepare_tsb_data.py --dry-run      # preview only
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path("/ocean/projects/cis260190p/yhwang2/data/TSB-AD-M/TSB-AD-M")
DST_DIR = Path("/ocean/projects/cis260190p/yhwang2/TimeEval-algorithms/experiments/data")


def parse_meta(filename: str) -> dict:
    m = re.match(
        r"(\d+)_(.+?)_id_(\d+)_(.+?)_tr_(\d+)_1st_(\d+)\.csv", filename
    )
    if not m:
        raise ValueError(f"Cannot parse: {filename}")
    return {
        "idx": int(m.group(1)),
        "source": m.group(2),
        "series_id": int(m.group(3)),
        "domain": m.group(4),
        "train_size": int(m.group(5)),
        "first_anomaly": int(m.group(6)),
    }


def compute_period(values: np.ndarray) -> int:
    """Simple FFT-based period detection. Returns 0 if none found."""
    from scipy.signal import find_peaks
    x = values[:, 0].astype(np.float64)
    x = x - x.mean()
    n = len(x)
    if n < 20:
        return 0
    fft = np.fft.rfft(x)
    power = np.abs(fft) ** 2
    power[0] = 0  # remove DC
    freqs = np.fft.rfftfreq(n)
    peaks, props = find_peaks(power, height=power.max() * 0.1)
    if len(peaks) == 0:
        return 0
    dominant = peaks[np.argmax(props["peak_heights"])]
    if freqs[dominant] == 0:
        return 0
    period = int(round(1.0 / freqs[dominant]))
    if period <= 1 or period >= n // 2:
        return 0
    return period


def process_one(csv_path: Path, dst_dir: Path, dry_run: bool = False) -> dict:
    meta = parse_meta(csv_path.name)
    collection = meta["source"]
    dataset_name = f"{collection}_id_{meta['series_id']}"
    tr = meta["train_size"]

    df = pd.read_csv(csv_path, index_col=0)
    n_rows = len(df)
    n_dims = df.shape[1] - 1  # exclude Label
    label_col = df.columns[-1]
    labels = df[label_col].values

    train_df = df.iloc[:tr]
    test_df = df.iloc[tr:]

    test_labels = labels[tr:]
    anomaly_positions = np.where(test_labels == 1)[0]
    num_anomalies = len(anomaly_positions)
    contamination = num_anomalies / max(len(test_labels), 1)

    if num_anomalies > 0:
        diffs = np.diff(anomaly_positions)
        breaks = np.where(diffs > 1)[0]
        lengths = np.diff(np.concatenate([[-1], breaks, [len(anomaly_positions) - 1]]))
        min_anom_len = int(lengths.min())
        median_anom_len = int(np.median(lengths))
        max_anom_len = int(lengths.max())
    else:
        min_anom_len = median_anom_len = max_anom_len = 0

    # period detection on training data
    train_values = train_df.iloc[:, :-1].values.astype(np.float32)
    period = compute_period(train_values)

    # stats
    values_all = df.iloc[:, :-1].values.astype(np.float64)
    mean_val = float(np.mean(values_all))
    std_val = float(np.std(values_all))

    # write files
    train_rel = f"{collection}/{dataset_name}.train.csv"
    test_rel = f"{collection}/{dataset_name}.test.csv"

    if not dry_run:
        out_dir = dst_dir / collection
        out_dir.mkdir(parents=True, exist_ok=True)
        train_out = train_df.rename(columns={label_col: "is_anomaly"})
        test_out = test_df.rename(columns={label_col: "is_anomaly"})
        train_out.to_csv(dst_dir / train_rel)
        test_out.to_csv(dst_dir / test_rel)

    return {
        "collection_name": collection,
        "dataset_name": dataset_name,
        "train_path": train_rel,
        "test_path": test_rel,
        "dataset_type": "real",
        "datetime_index": False,
        "split_at": tr,
        "train_type": "semi-supervised",
        "train_is_normal": True,
        "input_type": "multivariate",
        "length": n_rows,
        "dimensions": n_dims,
        "contamination": round(contamination, 6),
        "num_anomalies": num_anomalies,
        "min_anomaly_length": min_anom_len,
        "median_anomaly_length": median_anom_len,
        "max_anomaly_length": max_anom_len,
        "mean": round(mean_val, 6),
        "stddev": round(std_val, 6),
        "trend": "none",
        "stationarity": "difference_stationary",
        "period_size": period if period > 0 else np.nan,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert TSB-AD-M to TimeEval format")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--src", type=str, default=str(SRC_DIR))
    parser.add_argument("--dst", type=str, default=str(DST_DIR))
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    csvs = sorted(src.glob("*.csv"))

    print(f"Source: {src} ({len(csvs)} files)")
    print(f"Dest:   {dst}")
    if args.dry_run:
        print("DRY RUN — no files will be written\n")

    rows = []
    errors = []
    for i, csv_path in enumerate(csvs):
        try:
            row = process_one(csv_path, dst, dry_run=args.dry_run)
            rows.append(row)
            period_str = str(int(row["period_size"])) if not np.isnan(row["period_size"]) else "none"
            print(f"[{i+1:3d}/{len(csvs)}] {row['collection_name']:>12s}/{row['dataset_name']:<20s}  "
                  f"train={row['split_at']:>5d}  test={row['length']-row['split_at']:>6d}  "
                  f"dims={row['dimensions']:>3d}  period={period_str:>5s}  "
                  f"anom={row['num_anomalies']}")
        except Exception as e:
            errors.append((csv_path.name, str(e)))
            print(f"[{i+1:3d}/{len(csvs)}] ERROR {csv_path.name}: {e}")

    # write datasets.csv
    df = pd.DataFrame(rows)
    df.set_index(["collection_name", "dataset_name"], inplace=True)

    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)
        df.to_csv(dst / "datasets.csv")
        print(f"\ndatasets.csv written: {dst / 'datasets.csv'}")

    print(f"\nDone: {len(rows)} datasets, {len(errors)} errors")
    if errors:
        print("Errors:")
        for name, err in errors:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
