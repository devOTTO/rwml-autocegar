# Proposal 2 — Uncertainty-Aware Residual CEGAR: Results

**Verdict: P2 does not beat the best-HP/200ep RW-1 on the verdict set (0/3). Under the
corrected config it near-ties on OPPORTUNITY but stays well below on GECCO.**

## What Proposal 2 is (docx-faithful)
A confident forecasting error = high residual with LOW predictive uncertainty. Uncertainty
from MC-dropout (M passes): `u_t = (1/M)Σ‖Ŷ^m−μ‖²`, `e_t = ‖Y−μ‖/√(u_t+ε)`,
`g = σ(k_e(e_t−τ_e))·σ(k_c(τ_u−u_t))`. Only `_compute_signals` overridden. Score = mean|correction|.

## Methodology (corrected config)
`epochs=100`, `warmup=10`, `correction_init='neg_x'`, variant `mc5`, fixed `lam=1, tau=2`.
Warm-up = plain RW-1 then gate on. Unit = whole collection. Baseline = reproduction
best-HP/200ep. **Caveat**: Δ config-confounded (epoch/HP) → indicative.
**Cost** (gecco, 100ep): ~6 min (M=5 MC-dropout passes; grows with #features).

## Verdict set (AUC-PR; fixed / auto-λ)
| collection | n | DeepAnT* | RW-1* | P2 fixed | auto-λ | **Δ (fixed−RW-1)** | P2 AUC-ROC |
|---|:-:|:--:|:--:|:--:|:--:|:--:|:--:|
| OPPORTUNITY | 8 | 0.272 | 0.138 | 0.125 | 0.125 | **−0.013** | 0.708 |
| GECCO | 1 | 0.454 | 0.639 | 0.380 | 0.383 | **−0.259** | 0.838 |
| CreditCard | 1 | 0.147 | 0.111 | 0.025 | 0.025 | **−0.086** | 0.605 |

Beats RW-1 on **0/3**. auto-λ flat (gate weak → nothing to amplify).

## Shape spectrum (AUC-PR; fixed / auto-λ; W = beats RW-1)
| TAO (point, RW.995) | PSM (mixed, RW.137) | MSL (block, RW.131) | SWaT (block, RW.444) |
|:--:|:--:|:--:|:--:|
| 0.995 / 0.995 W | 0.116 / 0.115 | **0.135 W / 0.134 W** | 0.133 / 0.136 |

Ties TAO, beats MSL (weak baseline, margin ~noise), loses SWaT/PSM.

## Correction diagnostics (thesis §8.4, fixed)
| collection | gate→label AUC | corr@anom/norm | Overlap | AnomalyCoverage |
|---|:--:|:--:|:--:|:--:|
| GECCO | 0.510 | 6.43 | 0.154 | 0.617 |
| CreditCard | 0.409 | 1.63 | 0.007 | 0.201 |
| OPPORTUNITY | 0.182 | 1.08 | 0.121 | 0.140 |

## Interpretability
The MC-dropout uncertainty gate is the weakest localizer of the five (GECCO gate→label
AUC 0.51 ≈ random), so on GECCO P2 lags the residual-based proposals despite the corrected
config. It near-ties RW-1 on opportunity (like the others). Confirms the docx
MC-dropout-miscalibration risk.

## Decision
Does not beat tuned RW-1; weakest gate localization → fail-fast to Proposal 3.

## Reproduce
```bash
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
sbatch experiments/proposals/proposal2/submit_p2_coll.sh
python experiments/proposals/aggregate_collection.py --proposal 2
```
