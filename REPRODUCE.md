# Reproduction Guide

How to reproduce the DeepAnT / RW / RW-1 results in this repo (and run the
AutoCEGAR extension) from scratch. Two paths:

- **Path A — Standalone quick check** (this repo only): run a method on one or
  more TSB-AD-M CSVs and get AUC-PR / AUC-ROC. No benchmark harness. Best for a
  fast sanity check or to try the AutoCEGAR extension.
- **Path B — Full paper-comparable reproduction** (via the TSB-AD benchmark):
  reproduces the exact numbers in [`reproduction_comparison.md`](reproduction_comparison.md)
  over all 199 datasets. This is how every number in this repo was produced.

Expected results are in [`reproduction_comparison.md`](reproduction_comparison.md).

---

## 0. What this repo is

Reproduction of the DeepAnT / RW / RW-1 detectors from Afonso Baldo's MSc thesis,
plus **AutoCEGAR**, a residual-CEGAR extension on top of the reproduced RW-1.

The three detector source files are lifted from the TSB-AD benchmark and only had
their imports repointed to this repo's `tsb_common/`:

| Method | This repo | = TSB-AD model | Paper |
|--------|-----------|----------------|-------|
| DeepAnT | `deepant/cnn.py` | `TSB_AD/models/CNN.py` | baseline |
| RW | `rw/cnn_uns.py` | `TSB_AD/models/CNN_uns.py` | Algorithm 1 |
| RW-1 | `rw/cnn_rw.py` | `TSB_AD/models/CNN_RW.py` (bug-fixed) | Algorithm 2 |
| AutoCEGAR | `autocegar/rw_cegar.py` | — (novel extension) | — |

RW-1's two fixes vs. the author's original (`activation='linear'`, `l1_weight=0.001`)
are documented in the top-level [`README.md`](README.md).

---

## 1. Prerequisites

- **Python 3.11** (3.10+ should work)
- **A CUDA GPU** for full runs (RW/RW-1 ≈ 17 h for all 199 datasets on one V100;
  DeepAnT ≈ 40 min). Path A on a single small dataset runs on CPU in minutes.
- ~2 GB for the TSB-AD-M dataset.
- (Path B only) the **TSB-AD benchmark** repository and, optionally, a SLURM cluster.

---

## 2. Setup

```bash
# 2.1 clone this repo
git clone https://github.com/devOTTO/rwml-autocegar.git
cd rwml-autocegar

# 2.2 create an environment and install deps
python3 -m venv .venv && source .venv/bin/activate      # or conda create -n rwml python=3.11
pip install -r requirements.txt

# 2.3 get the TSB-AD-M dataset (200 multivariate CSVs)
#     from the TSB-AD benchmark: https://github.com/Afonsob1/TSB-AD
#     Each file: [T, n_features] columns + a final `Label` column (0/1).
#     Put them somewhere and remember the path, e.g.:
export DATA_DIR=/path/to/TSB-AD-M          # dir containing 001_*.csv ... 200_*.csv
```

That is everything Path A needs.

---

## 3. Path A — Standalone quick check (no benchmark harness)

`reproduce_standalone.py` runs RW / RW-1 / AutoCEGAR directly on CSV(s) and prints
AUC-PR / AUC-ROC. RW-family detectors are unsupervised: they fit the full series
and score each timestep by its correction magnitude.

```bash
# RW-1 on one dataset at a fixed l1_weight
python reproduce_standalone.py --method rw1 --data_dir "$DATA_DIR" \
    --file 001_Genesis_id_1_Sensor_tr_4055_1st_15538.csv --l1_weight 0.001

# RW-1 "best-HP": sweep l1 in {1.0,0.1,0.01,0.001}, keep the best AUC-PR (paper protocol)
python reproduce_standalone.py --method rw1 --best_hp --data_dir "$DATA_DIR" \
    --file 001_Genesis_id_1_Sensor_tr_4055_1st_15538.csv

# RW (Algorithm 1) over every dataset -> summary CSV
python reproduce_standalone.py --method rw --all --data_dir "$DATA_DIR" --out rw_scores.csv

# AutoCEGAR extension (with warm-up curriculum)
python reproduce_standalone.py --method cegar --data_dir "$DATA_DIR" \
    --file 001_Genesis_id_1_Sensor_tr_4055_1st_15538.csv --lam 1.0 --warmup_epochs 10
```

Notes:
- `--epochs` defaults to 200 (the paper config). Use a small value (e.g. `--epochs 5`)
  for a quick CPU smoke.
- This path scores the full labeled series. It is a fast approximation; the
  authoritative, paper-comparable evaluation is Path B (which uses TSB-AD's exact
  train/test protocol and its 9-metric evaluator).

---

## 4. Path B — Full paper-comparable reproduction (TSB-AD benchmark)

This is exactly how the numbers in this repo were generated. The detectors run
inside the TSB-AD benchmark via `benchmark_exp.Run_Detector_M`, which handles the
train/test split and computes all 9 metrics (AUC-PR, AUC-ROC, VUS, F1 variants).

```bash
# 4.1 get the TSB-AD benchmark and install it
git clone https://github.com/Afonsob1/TSB-AD.git
cd TSB-AD && pip install -e . && cd ..
export TSB_AD=$(pwd)/TSB-AD

# 4.2 make sure TSB-AD uses THIS repo's (fixed) detector code.
#     The reproduction uses the bug-fixed RW-1; copy the three model files in:
cp deepant/cnn.py    "$TSB_AD/TSB_AD/models/CNN.py"
cp rw/cnn_uns.py     "$TSB_AD/TSB_AD/models/CNN_uns.py"
cp rw/cnn_rw.py      "$TSB_AD/TSB_AD/models/CNN_RW.py"
#     (then re-point their imports back to TSB-AD's `..utils.*` — see each file's
#      top two `from tsb_common...` lines; in TSB-AD they are `from ..utils...`.)
```

### 4.3 Run each detector (loops over all files in $DATA_DIR)

```bash
cd "$TSB_AD"

# DeepAnT (baseline)                       -> eval/metrics/multi/CNN/*.csv
for f in "$DATA_DIR"/*.csv; do
  python -m benchmark_exp.Run_Detector_M --AD_Name CNN \
    --dataset_dir "$DATA_DIR/" --filename "$(basename "$f")" \
    --save True --save_dir eval/metrics/multi/ --score_dir eval/score/multi/
done

# RW (Algorithm 1 = CNN_UNS)               -> eval/metrics/multi_algofaithful/CNN_UNS/*.csv
#   same loop with --AD_Name CNN_UNS and --save_dir eval/metrics/multi_algofaithful/

# RW-1 best-HP sweep (Algorithm 2 = CNN_RW): run the loop 4x, once per l1_weight.
#   RW1_L1_WEIGHT is read by the fixed CNN_RW; results go to l1_<w>/CNN_RW/*.csv
for w in 1.0 0.1 0.01 0.001; do
  export RW1_L1_WEIGHT=$w
  for f in "$DATA_DIR"/*.csv; do
    python -m benchmark_exp.Run_Detector_M --AD_Name CNN_RW \
      --dataset_dir "$DATA_DIR/" --filename "$(basename "$f")" \
      --save True --save_dir "eval/metrics/multi_rw1_l1sweep/l1_${w}/" \
      --score_dir "eval/score/multi_rw1_l1sweep/l1_${w}/"
  done
done
```

The exact SLURM scripts used are in the repo (adjust the hardcoded paths, see §6):
`deepant/reproduction/submit_cnn.sh`, `rw/reproduction/submit_rw.sh`,
`experiments/exp_e_algofaithful/submit_l1sweep_rw1.sh`.

### 4.4 Aggregate

```bash
# back in the rwml-autocegar repo, with the sweep dirs available:
python experiments/exp_e_algofaithful/combine_rw1_besthp.py
#   -> prints per-family + overall best-HP numbers, writes rw1_besthp_summary.csv
```

DeepAnT / RW summaries are the mean over their per-dataset CSVs (see
`deepant/reproduction/` and `rw/reproduction/`).

---

## 5. Expected results

See [`reproduction_comparison.md`](reproduction_comparison.md) for the full
overall + per-collection + per-dataset tables. Headline (mean over n=199):

| Method | ours AUC-PR | paper AUC-PR (T6.1) | ours AUC-ROC |
|--------|:---:|:---:|:---:|
| DeepAnT | 0.350 | 0.33 | 0.770 |
| RW | 0.321 | 0.29 | 0.703 |
| RW-1 (best-HP) | 0.289 | 0.28 | 0.725 |

Small run-to-run variance is expected (random init / shuffling; the sweep did not
fix a global seed).

---

## 6. Portability note (important)

The committed SLURM `submit_*.sh` scripts and a few experiment scripts contain
**absolute paths from the original author's account**, e.g.:

```
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd     /ocean/projects/cis260190p/yhwang2/TSB-AD
dataset_dir="/ocean/projects/cis260190p/yhwang2/data/TSB-AD-M/TSB-AD-M/"
```

Before running them, replace these with your own env-activation, `$TSB_AD`, and
`$DATA_DIR`. Path A (`reproduce_standalone.py`) takes `--data_dir` as an argument
and has no hardcoded paths.

---

## 7. AutoCEGAR extension (not a paper reproduction)

`CNN_RW_CEGAR` (`autocegar/rw_cegar.py`) adds a CEGAR gate on top of the reproduced
RW-1. It runs end-to-end (`--method cegar` above, or `sbatch submit_autocegar_smoke.sh`).
Its `E_t` / `C_t` signal formulas are still placeholders pending the
confidence/wrongness definition; auto-λ works today, auto-τ activates once `C_t`
becomes per-window. See `autocegar/reproduction/README.md`.

## 8. W&B (optional)

Result runs are published at `wandb.ai/yoonmeeh-cmu/rwml-autocegar` (per-dataset +
summary runs for DeepAnT / RW / RW-1). Logging is optional and off by the default
benchmark path; the aggregation-to-W&B script is
`experiments/exp_e_algofaithful/log_perdataset_to_wandb.py`.
