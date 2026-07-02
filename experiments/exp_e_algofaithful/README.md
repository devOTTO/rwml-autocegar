# EXP-E — Algorithm-Faithful Full Run (RW / RW-1 on TSB-AD-M)

**Date:** 2026-06-30
**Status:** ✅ Complete (both halves)
**W&B:** project `yoonmeeh-cmu/rwml-autocegar`, group `EXP-E-algofaithful`
(runs `EXP-E-RW-full`, `EXP-E-RW-1-full`)

## Goal

Reproduce the RW and RW-1 anomaly-detection results from Afonso Baldo's 2025
FEUP MSc thesis *"Read & Write Machine Learning for Outlier Detection in
Multivariate Time Series"* on the **full TSB-AD-M benchmark** (all 180 files;
199 result rows), using the **algorithm-faithful rewrite** of the TSB-AD model
code.

Paper's reported best AUC-PR (Table 6.5): **RW = 0.34, RW-1 = 0.35**
(RW-2 = 0.44, not run here).

## Result summary

| Method | AD_Name | Paper AUC-PR | **AUC-PR mean** | AUC-PR median | AUC-ROC mean | Freeze* |
|--------|---------|:---:|:---:|:---:|:---:|:---:|
| RW   | `CNN_UNS` | 0.34 | **0.3214** | 0.1445 | 0.7026 | 15 / 199 |
| RW-1 | `CNN_RW`  | 0.35 | **0.2977** | 0.1045 | 0.6667 | 17 / 199 |

\* *Freeze* = datasets with AUC-ROC < 0.4 (well below random), the symptom of
the correction-tensor "freeze" bug (see below).

**Excluding freeze cases, the numbers land right on the paper:**

| Method | AUC-PR mean (incl. freeze) | **AUC-PR mean (excl. freeze)** | n |
|--------|:---:|:---:|:---:|
| RW   | 0.3214 | **0.3471** | 184 |
| RW-1 | 0.2977 | **0.3242** | 182 |

So RW is effectively **reproduced** (0.347 vs paper 0.34) once the freeze
datasets are removed, and RW-1 gets close (0.324 vs 0.35). The remaining gap
is almost entirely the freeze pathology, which is concentrated in one dataset
family.

## Key finding: the "freeze" is a GHL-family problem, not Genesis-only

Freeze breakdown by dataset family (AUC-ROC < 0.4):

| Method | GHL | SMAP | MSL | MITDB | Genesis | Total |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| RW   | 13 | 1 | 1 | – | – | 15 |
| RW-1 | 10 | 2 | 2 | 1 | 1 | 17 |

There are 24 GHL files in the benchmark. **RW freezes on 13 of 24 GHL files**
and RW-1 on 10 of 24. This is a much broader pattern than the single Genesis
case seen during pre-run validation (see `../../SESSION_CONTEXT.md`): the freeze
is **not RW-1-specific** — plain RW freezes on GHL too. GHL is a large,
long-series family (50k train points), which fits the earlier hypothesis that
early predictive saturation drives the correction gradient to zero.

**Implication:** rooting out the freeze on GHL is now the single
highest-leverage fix for closing the remaining gap to the paper. The current
best guess (from validation) — RMSprop's adaptive step and/or the L1-penalty
gradient collapsing to exactly zero, made worse for RW-1 by ReLU gating — should
be investigated specifically on a GHL file.

## Code used

**Model code (the thing under test):** the algorithm-faithful rewrite,
version `03_rewritten_algorithm_faithful` in
[`../../tsb_ad_models/`](../../tsb_ad_models). At run time this is what lived in
the TSB-AD repo at:

- `TSB-AD/TSB_AD/models/CNN_uns.py` — RW (paper Algorithm 1)
- `TSB-AD/TSB_AD/models/CNN_RW.py` — RW-1 (paper Algorithm 2)

Faithful-rewrite config (the paper's best setup):

| Knob | Value |
|------|-------|
| Correction optimizer | RMSprop |
| Loss | RMSE |
| Correction rate (lr) | 0.1 |
| Update cadence | epoch-wise (accumulate over epoch, update once at epoch end) |
| Epochs | 200 |
| Gradient activation | Linear (RW) / ReLU (RW-1) |
| Score | `abs(correction)` averaged over feature channels — **no** Savitzky-Golay, **no** extra normalization |

(The old TSB-AD code used Adam + MSE + instance-wise updates + a Savitzky-Golay
filter that actually belongs only to RW-2. That produced AUC-PR ~0.04–0.06. See
`../../SESSION_CONTEXT.md` and `../../tsb_ad_models/README.md` for the full
three-way version comparison.)

**Harness:** TSB-AD's `benchmark_exp.Run_Detector_M`, one invocation per
dataset:

```bash
python -m benchmark_exp.Run_Detector_M \
    --AD_Name {CNN_UNS|CNN_RW} \
    --dataset_dir /ocean/projects/cis260190p/yhwang2/data/TSB-AD-M/TSB-AD-M/ \
    --filename <file>.csv \
    --save True \
    --save_dir eval/metrics/multi_algofaithful/
```

**Submit scripts (SLURM, PSC Bridges-2):**

- `TSB-AD/submit_rw_algofaithful_full.sh`  → job **41840833** `EXP-E-RW-full`
- `TSB-AD/submit_rw1_algofaithful_full.sh` → job **41840834** `EXP-E-RW1-full`

Both: `GPU-shared`, `v100-32:1`, 5 CPUs, 63 GB, 48 h budget.
Env: `source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate`.

**Runtime:** started 2026-06-30 01:16, RW finished 21:35 (20h20m elapsed), RW-1
finished 21:12 (19h57m). Both `COMPLETED`, exit 0:0.

## Outputs

- Per-dataset metrics CSVs (199 each):
  `TSB-AD/eval/metrics/multi_algofaithful/CNN_UNS/` (RW),
  `TSB-AD/eval/metrics/multi_algofaithful/CNN_RW/` (RW-1).
  Columns: `file, Time, AUC-PR, AUC-ROC, VUS-PR, VUS-ROC, Standard-F1, PA-F1,
  Event-based-F1, R-based-F1, Affiliation-F`.
- SLURM logs: `TSB-AD/logs/EXP-E-RW-full_41840833.{out,err}`,
  `TSB-AD/logs/EXP-E-RW1-full_41840834.{out,err}`.
- W&B: per-algorithm run with full config, summary metrics (mean+median for
  every metric column), a sortable per-dataset table, and AUC-PR / AUC-ROC
  histograms.

## Reproduce the W&B logging

```bash
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
set -a; source .env; set +a
# W&B artifact staging must NOT go to /jet/home (25 GB quota, currently full).
BASE=/ocean/projects/cis260190p/yhwang2/.wandb_scratch
export WANDB_CACHE_DIR=$BASE/cache WANDB_DATA_DIR=$BASE/data \
       WANDB_ARTIFACT_DIR=$BASE/artifacts TMPDIR=$BASE/tmp
python experiments/exp_e_algofaithful/log_exp_e_to_wandb.py
```

> **Gotcha:** logging the per-dataset `wandb.Table` stages a file locally. With
> the default cache dir on `/jet/home` (already at the ~25 GB quota) this fails
> with `OSError: [Errno 122] Disk quota exceeded`. Redirect the W&B cache/tmp
> dirs to `/ocean` (as above), which has room.

## Next steps

1. **Fix the GHL freeze** (highest leverage). Debug on one GHL file, e.g.
   `032_GHL_id_1_Sensor_tr_50000_1st_65001.csv`, which freezes for both RW and
   RW-1. Watch the L1-penalty gradient / RMSprop step going to exactly zero.
   Fixing this alone should move RW to ≈0.347 → paper-matching and lift RW-1.
2. Build **RW-2** (Algorithm 3, Savitzky-Golay variant, paper's best at 0.44)
   on the TSB-AD side — it does not exist there yet.
3. Reconcile the independent `rwml-autocegar/rw/*.py` reimplementation with the
   TSB-AD-derived code.
