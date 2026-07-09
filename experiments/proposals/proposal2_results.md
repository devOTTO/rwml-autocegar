# Proposal 2 — Uncertainty-Aware Residual CEGAR: Results

**Verdict: P2 loses to the RW-1 baseline on all 6 collections → fail-fast to Proposal 3.**

## What Proposal 2 is (docx-faithful)
A confident forecasting error = high residual with LOW predictive uncertainty.
Uncertainty from **MC-dropout** (M stochastic forward passes). Only
`_compute_signals` is overridden (uncertainty enters BOTH wrongness and confidence,
per the doc):
```
u_t   = (1/M) Σ_m ‖Ŷ_t^m − μ_t‖²                  # MC-dropout variance
e_t   = ‖Y_t − μ_t‖ / √(u_t + ε)                   # uncertainty-standardized residual
g_err = σ(k_e·(e_t − τ_e)),  g_conf = σ(k_c·(τ_u − u_t))   # low uncertainty ⇒ confident
gate  = g_err · g_conf
```
Known deviation: the doc's conservative `g_safe` write-back is not implemented
(same single-graph coupling as P1); RW correction uses the inherited ScaleGrad path.
`e_t`/`u_t` are un-normalized (per the doc), so τ_e/τ_u are scale-dependent.

## Methodology (collection-level)
Same as P1: `epochs=100`, fixed HP (`mc5`, τ=2, λ=1, τ_u=0), auto-tuning/sweeps off.
Per-collection mean over all series; RW-1 / DeepAnT = reproduction per-collection
means (reference, best-HP/200ep). Verdict set (opportunity/gecco/creditcard) +
characterization set (SMAP/SMD/MITDB — domain / anomaly-type diversity).

## Collection-level results

`*` DeepAnT / RW-1 = reproduction per-collection means (reference). Δ = P2 − RW-1.

| collection | n | DeepAnT AUC-PR* | RW-1 AUC-PR* | P2 AUC-PR | **Δ (P2−RW-1)** | P2 AUC-ROC |
|---|:-:|:--:|:--:|:--:|:--:|:--:|
| GECCO | 1 | 0.454 | 0.639 | 0.205 | **−0.434** | 0.778 |
| SMD | 22 | 0.396 | 0.233 | 0.038 | **−0.195** | 0.475 |
| CreditCard | 1 | 0.147 | 0.111 | 0.002 | **−0.110** | 0.469 |
| OPPORTUNITY | 8 | 0.272 | 0.138 | 0.034 | **−0.104** | 0.338 |
| SMAP | 27 | 0.146 | 0.119 | 0.069 | **−0.050** | 0.583 |
| MITDB | 13 | 0.166 | 0.089 | 0.087 | **−0.002** | 0.603 |

**P2 beats RW-1 on 0/6 collections.**

## Interpretability — P2 fails differently from P1
P1's gate *localizes* anomalies and then erases them (GECCO: gate→label AUC ≈ 0.90,
correction ≈ 5.5× on anomaly windows). **P2's uncertainty-based gate does neither** —
per-series (id_1) diagnostics:

| collection (id_1) | gate→label AUC | corr @anom/norm | trigger frac |
|---|:-:|:-:|:-:|
| SMAP | 0.45 | 1.29 | ≈0.00 |
| MITDB | 0.44 | 1.19 | ≈0.00 |
| SMD | 0.23 | 1.00 | ≈0.00 |

Across series the gate→label AUC scatters 0.17–0.78 (≈0.5 / below on average),
correction concentration is ≈1× (not targeted), and the gate rarely exceeds its
threshold (`trigger_frac ≈ 0`). So the MC-dropout confidence is near-flat / poorly
calibrated — the gate is essentially **uninformative**, and P2 degenerates to a
(slightly worse) perturbed RW-1. This directly confirms the doc's "MC-dropout
uncertainty may be poorly calibrated" risk. **Different failure from P1**:
P1 over-targets (erases anomalies); P2's gate is under-informative.

## Characterization hypothesis — NOT supported
Hypothesis (a-priori): P2 helps where anomalies are genuinely uncertain (SMAP
satellite telemetry), is neutral on industrial (SMD), and hurts on periodic signals
(MITDB ECG, like gecco).

Observed: P2 loses everywhere, and the pattern **contradicts** the hypothesis:
- **MITDB (ECG, periodic)** — predicted to hurt most, is instead **nearly tied
  (Δ−0.002)**, the *closest* to RW-1 of all collections.
- **SMAP (uncertain telemetry)** — predicted to help, did not win (Δ−0.050), though
  its gap is among the smaller ones.
- **GECCO / SMD** — the largest losses (Δ−0.43 / −0.20).

So the "uncertainty axis" does not explain when P2 helps (it never helps). What
varies is the *degree* of harm — near-harmless on MITDB, severe on gecco/SMD. The
uncertainty-standardized residual + inverse-uncertainty confidence does not rescue
the core RW-CEGAR failure mode (amplifying correction on anomaly windows).

## Decision
**Fail-fast → Proposal 3 (RW-Correction-Consistency CEGAR).** P2, the cleanest
theoretical CEGAR analogue, does not beat RW-1 on any collection, and the
characterization hypothesis is not supported.

## Reproduce
```bash
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
sbatch experiments/proposals/submit_p2_coll.sh       # 72-series collection array
python experiments/proposals/aggregate_collection.py --proposal 2
wandb sync --include-offline ./wandb/offline-run-<date>_*
```
