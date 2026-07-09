# Proposal 1 — Residual-Gated RW-CEGAR: Results

**Verdict: P1 loses to the RW-1 baseline on all evaluated collections → fail-fast to Proposal 2.**

## What Proposal 1 is
Simplest CEGAR × RW-ML hybrid, built on the reproduced RW-1 (Algorithm 2). It
overrides only the wrongness / confidence signals:
- **Wrongness** `E_t = sigmoid(k·(robust_z − τ))`, `robust_z = (r − median)/MAD`
- **Confidence** `C_t = 1` (basic) or `sigmoid(k·(r − Q_q(r))/MAD)` (selective) — the
  variant only switches C_t; τ and λ are variant-independent.
- **Gate** `g = E_t·C_t`, **scale** `s = (1+λg)/mean(1+λg)`, applied to the
  per-window forecasting-loss **gradient** via `ScaleGrad` (loss value unchanged).
- Anomaly score = `mean|correction|` (same as RW-1, for a fair delta).

## Methodology (collection-level)
Fixed config: `epochs=100`, `warmup=10`, `window=50`, `batch=256`, `l1_weight=0.001`,
`activation=linear`, fixed HP (`lam=1, tau=2, k=1`), auto-tuning off.

**Unit = whole collection.** Each series in a collection is run and averaged to a
per-collection mean AUC-PR/ROC. **Baseline** RW-1 (and DeepAnT reference) are the
**reproduction per-collection means** (`rw/reproduction/summary_rw1_besthp.csv`,
`deepant/reproduction/summary_per_dataset.csv`; best-HP / 200ep / TSB-AD eval — so
the delta is indicative, config not identical to P1's 100ep). `all` verdict set =
opportunity + gecco + creditcard (gecco & creditcard are n=1, i.e. already whole
collections; opportunity = mean over its 8 series).

## Collection-level results (primary verdict)

`*` DeepAnT / RW-1 = reproduction per-collection means (reference). Δ = P1 − RW-1.

| collection | n | DeepAnT AUC-PR* | RW-1 AUC-PR* | P1 AUC-PR | **Δ (P1−RW-1)** | P1 AUC-ROC |
|---|:-:|:--:|:--:|:--:|:--:|:--:|
| OPPORTUNITY | 8 | 0.272 | 0.138 | 0.063 | **−0.075** | 0.419 |
| GECCO | 1 | 0.454 | 0.639 | 0.472 | **−0.167** | 0.920 |
| CreditCard | 1 | 0.147 | 0.111 | 0.003 | **−0.108** | 0.610 |

**P1 beats RW-1 on 0/3 collections.** The opportunity collection mean (8 series)
confirms the earlier single-series screen — P1 does not recover at collection level.
DeepAnT is strongest on opportunity/creditcard, RW-1 on gecco; P1 is never best.

## Interpretability (why it fails)
Matches the risk the proposal flagged: the gate correctly localizes anomalies
(gate→label AUC ≈ 0.90 on gecco) and pours correction into them (gecco
corr@anom/norm ≈ 5.5×), which "erases" the anomalies from the `mean|correction|`
score, so AUC-PR drops. Per-series `gate/*` and `corr/*` are on the `interp`/
`collection`-tagged wandb runs. A Baldo-Fig-6.1-style original-vs-corrected plot is
in `experiments/proposals/plot_correction_example.py` (gecco figure in wandb).

## Single-series exploratory sweeps (not re-run at collection level)
Earlier τ∈{1.5,2.5,3} and λ∈{0.5,2} sweeps and the `selective` variant were run on
opportunity id_1 only (fail-fast screen). All stayed ≈0.02 AUC-PR / <0.5 ROC and did
not recover P1; sweeps/auto-tuning were held for the collection re-run since the
failure is conceptual, not a HP-tuning issue.

## Decision
**Fail-fast → Proposal 2.** P1 (RW-CEGAR form) does not beat RW-1 on any collection.

## Reproduce
```bash
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
sbatch experiments/proposals/submit_p1_coll.sh       # collection-level, per-series array
python experiments/proposals/aggregate_collection.py --proposal 1   # -> collection table
wandb sync --include-offline ./wandb/offline-run-<date>_*
```
