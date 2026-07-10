# Proposal 4 — Dual-Gate Residual-and-Gradient RW-CEGAR: Results

**Verdict: P4 does not beat the best-HP/200ep RW-1 on the verdict set (0/3), but it is the
closest to RW-1 on GECCO after P5 (0.599, Δ−0.040; auto-λ 0.628).**

## What Proposal 4 is (docx spec, amplification-only Stage-1)
A point is a confident error if it has high residual AND is gradient-correctable.
`g_res = σ(k_r(robust_z(resid)−τ_r))`, `h_t = ‖∂loss/∂input‖` (per window, extra fwd+bwd),
`g_grad = σ(k_h(robust_z(h_t)−τ_h))`, `g = g_res·g_grad`. `benefit` variant uses the
loss-reduction estimate instead of ‖grad‖. Amplification only (docx write-back not
implemented). Only `_compute_signals` overridden. Score = mean|correction|.

## Methodology (corrected config)
`epochs=100`, `warmup=10`, `correction_init='neg_x'`, variant `gradnorm`, fixed `lam=1`.
Warm-up = plain RW-1 then gate on. Unit = whole collection. Baseline = reproduction
best-HP/200ep. **Caveat**: Δ config-confounded (epoch/HP) → indicative.
**Cost** (gecco, 100ep): medium — an extra input-gradient fwd+bwd per batch (~1.5–2× P1).

## Verdict set (AUC-PR; fixed / auto-λ)
| collection | n | DeepAnT* | RW-1* | P4 fixed | auto-λ | **Δ (fixed−RW-1)** | P4 AUC-ROC |
|---|:-:|:--:|:--:|:--:|:--:|:--:|:--:|
| OPPORTUNITY | 8 | 0.272 | 0.138 | 0.107 | 0.110 | **−0.031** | 0.671 |
| GECCO | 1 | 0.454 | 0.639 | 0.599 | 0.628 | **−0.040** | 0.935 |
| CreditCard | 1 | 0.147 | 0.111 | 0.026 | 0.025 | **−0.085** | 0.637 |

Beats RW-1 on **0/3**; GECCO closest of the non-P5 proposals (auto-λ 0.628).

## Shape spectrum (AUC-PR; fixed / auto-λ; W = beats RW-1)
| TAO (point, RW.995) | PSM (mixed, RW.137) | MSL (block, RW.131) | SWaT (block, RW.444) |
|:--:|:--:|:--:|:--:|
| 0.995 / 0.995 | 0.125 / 0.128 | 0.122 / 0.121 | 0.143 / 0.149 |

No wins — does not edge MSL; loses SWaT/PSM; TAO marginally below RW-1.

## Correction diagnostics (thesis §8.4, fixed)
| collection | gate→label AUC | corr@anom/norm | Overlap | AnomalyCoverage |
|---|:--:|:--:|:--:|:--:|
| GECCO | 0.809 | 10.40 | 0.209 | 0.840 |
| CreditCard | 0.839 | 1.70 | 0.009 | 0.246 |
| OPPORTUNITY | 0.495 | 1.08 | 0.105 | 0.120 |

## Interpretability
The dual gate localizes well on GECCO (0.81) with strong correction concentration
(10.4×, 84% coverage) → its high GECCO score (0.599). But the input-gradient signal is
noisy (docx risk: gradients also spike at noise/discontinuities), and P4 does not
generalize across shapes. Near the RW-1 ceiling on GECCO but never over it.

## Decision
Does not beat tuned RW-1 → fail-fast to Proposal 5.

## Reproduce
```bash
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
sbatch experiments/proposals/proposal4/submit_p4_coll.sh
python experiments/proposals/aggregate_collection.py --proposal 4
```
