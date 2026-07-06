# Proposal 1 — Residual-Gated RW-CEGAR: Fail-Fast Results

**Date:** 2026-07-05 · **Slurm array:** `41945419` (11 tasks, GPU-shared, v100-32) ·
**wandb:** project `rwml-autocegar`, group `proposal1`

## What Proposal 1 is

The simplest CEGAR × RW-ML hybrid from the research proposal. Built on the
reproduced RW-1 (Algorithm 2) substrate; it overrides only the wrongness /
confidence signals:

- **Wrongness** `E_t = sigmoid(k·(robust_z − τ))`, `robust_z = (r − median)/MAD`
- **Confidence** `C_t = 1` (basic) or `sigmoid(k·(r − Q_q(r))/MAD)` (selective)
- **Gate** `g = E_t·C_t`, **scale** `s = (1 + λg)/mean(1 + λg)` (batch-normalized)
- `s` scales the per-window forecasting-loss **gradient** via `ScaleGrad`
  (identity forward → reported loss stays true RMSE).
- Anomaly score = `mean|correction|` (same as RW-1, for a fair delta).

## Fixed configuration (kept at RW-1 reproduction values)

`epochs=100`, `warmup_epochs=10` (forecaster-only, then correction+gate turn on),
`window=50`, `batch=256`, `l1_weight=0.001`, `activation=linear`, `lr=0.0008`,
`correction_rate=0.1`, `correction_init=zero`, `lam_mode/tau_mode=fixed`,
`scale_normalize=True`. Warm-up was verified active in the logs (epochs 1–10 show
`L1=0, gate=0`; correction/gate engage from epoch 11).

## Datasets (week-8 selection: topically disconnected, span the RW-vs-DeepAnT range)

| key | dataset | domain |
|---|---|---|
| opportunity | 129_OPPORTUNITY_id_1_HumanActivity | activity recognition (DeepAnT-favored) |
| gecco | 173_GECCO_id_1_Sensor | water quality (RW-strong) |
| creditcard | 137_CreditCard_id_1_Finance | finance (middle, headroom) |

## What was tested

| stage | tag | vary | datasets | baseline |
|---|---|---|---|---|
| 1 | `stage1` | — (basic, τ=2, λ=1) | all 3 | RW-1 |
| 2 | `stage2-tau` | τ ∈ {1.5, 2.5, 3.0} | opportunity | — |
| 3 | `stage3-lam` | λ ∈ {0.5, 2.0} | opportunity | — |
| 4 | `stage4-sel` | variant=selective | all 3 | — |

## Results

### Stage 1 — P1 (basic, default HP) vs RW-1 baseline

Primary metric AUC-PR. **Δ = P1 − RW-1.**

| dataset | P1 AUC-PR | RW-1 AUC-PR | **Δ AUC-PR** | P1 AUC-ROC |
|---|:--:|:--:|:--:|:--:|
| opportunity | 0.0209 | 0.0284 | **−0.0075** | 0.367 |
| gecco | 0.4565 | 0.6671 | **−0.2105** | 0.922 |
| creditcard | 0.0032 | 0.1227 | **−0.1195** | 0.595 |

**P1 is worse than RW-1 on all three datasets**, and heavily so on gecco (the set
where RW-1 was strongest) and creditcard.

### Stage 2 — τ sweep (opportunity, basic, λ=1)

| τ | AUC-PR | AUC-ROC |
|:--:|:--:|:--:|
| 1.5 | 0.0207 | 0.365 |
| 2.0 | 0.0209 | 0.367 |
| 2.5 | 0.0208 | 0.336 |
| 3.0 | 0.0200 | 0.340 |

τ has essentially no effect; AUC-ROC stays < 0.5.

### Stage 3 — λ sweep (opportunity, basic, τ=2)

| λ | AUC-PR | AUC-ROC |
|:--:|:--:|:--:|
| 0.5 | 0.0202 | 0.332 |
| 1.0 | 0.0209 | 0.367 |
| 2.0 | 0.0235 | 0.426 |

Stronger gating (λ↑) helps slightly on opportunity but still stays below RW-1
(0.0284) and below AUC-ROC 0.5.

### Stage 4 — selective confidence (τ=2, λ=1)

| dataset | basic AUC-PR | selective AUC-PR | basic AUC-ROC | selective AUC-ROC |
|---|:--:|:--:|:--:|:--:|
| opportunity | 0.0209 | 0.0221 | 0.367 | 0.355 |
| gecco | 0.4565 | 0.4473 | 0.922 | 0.914 |
| creditcard | 0.0032 | 0.0041 | 0.595 | 0.631 |

Selective ≈ basic (marginally worse on gecco, marginally better on creditcard).
No meaningful improvement.

## Interpretation

The negative result matches the risk the proposal explicitly flagged for
Proposal 1: **high residual is also anomaly evidence.** The CEGAR gate amplifies
the correction gradient on high-residual (anomaly-candidate) windows, so the RW
correction learns to "absorb" the anomalies. That shrinks `|correction|` exactly
where anomalies are, degrading the correction-based anomaly score. gecco — where
RW-1 had the most signal to lose — drops the most, consistent with this
mechanism. The gate itself worked as intended (`gate ≈ 0.22` in the main phase);
this is a conceptual outcome, not a bug.

**Scope caveat.** This tests the *RW-correction* reading of Proposal 1 (gate →
correction, score = `|correction|`), which is what the RW×CEGAR project targets.
The proposal's headline §1.2 phrasing (amplify model gradient, score by residual)
was not tested, but with RW fully active the residual score is largely entangled
with the correction (the model fits corrected data), so it is unlikely to change
the verdict without weakening RW — which would leave the RW×CEGAR premise.

## Decision

**Fail-fast: move on to Proposal 2 (Uncertainty-Aware Residual CEGAR).** Proposal 1
in its RW-CEGAR form does not beat the RW-1 baseline on any selected dataset.

## Reproduce

```bash
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
sbatch experiments/proposals/submit_p1_grid.sh      # 11-task array
# raw per-run rows: experiments/proposals/results_p1.csv
# sync offline wandb runs to the online project:
wandb sync --include-offline ./wandb/offline-run-*
```
