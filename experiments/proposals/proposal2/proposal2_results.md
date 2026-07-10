# Proposal 2 — Uncertainty-Aware Residual CEGAR: Results

**Verdict: P2 does not beat the best-HP/200ep RW-1 reproduction on any verdict
collection (0/3). Near-ties on OPPORTUNITY (Δ−0.013) but stays well below RW-1 on
GECCO (0.380 vs 0.639) — the weakest of the five on GECCO.**

## Config note
Corrected config: warm-up = **plain RW-1** (gate OFF) then gate ON; `correction_init
='neg_x'`. Old-config docs are in `_backup_oldconfig/`. Delta is config-confounded on
the epoch/HP axis (proposals 100ep/default-HP vs RW-1 best-HP/200ep).

## What Proposal 2 is (docx-faithful)
A confident forecasting error = high residual with LOW predictive uncertainty
(MC-dropout, M passes). Only `_compute_signals` overridden:
```
u_t = (1/M) Σ‖Ŷ^m − μ‖²                    # MC-dropout variance
e_t = ‖Y − μ‖ / √(u_t + ε)                  # uncertainty-standardized residual
g_err = σ(k_e·(e_t − τ_e)),  g_conf = σ(k_c·(τ_u − u_t))
gate = g_err · g_conf
```

## Methodology (collection-level)
Verdict set (opportunity 8 + gecco 1 + creditcard 1 = 10 series), `epochs=100`,
`warmup=10`, variant `mc5` (M=5), fixed HP. RW-1/DeepAnT = reproduction means.

## Collection-level results (fixed)

`*` reference reproduction means. Δ = P2 − RW-1.

| collection | n | DeepAnT AUC-PR* | RW-1 AUC-PR* | P2 AUC-PR | **Δ (P2−RW-1)** | P2 AUC-ROC |
|---|:-:|:--:|:--:|:--:|:--:|:--:|
| OPPORTUNITY | 8 | 0.272 | 0.138 | 0.125 | **−0.013** | 0.708 |
| GECCO | 1 | 0.454 | 0.639 | 0.380 | **−0.259** | 0.838 |
| CreditCard | 1 | 0.147 | 0.111 | 0.025 | **−0.086** | 0.605 |

**P2 beats RW-1 on 0/3.**

### Correction diagnostics (thesis §8.4, fixed)
| collection | gate→label AUC | corr@anom/norm | Overlap | AnomalyCoverage |
|---|:--:|:--:|:--:|:--:|
| GECCO | 0.510 | 6.43 | 0.154 | 0.616 |
| CreditCard | 0.409 | 1.63 | 0.007 | 0.201 |
| OPPORTUNITY | 0.182 | 1.08 | 0.121 | 0.140 |

## Auto-tuning ablation (auto-λ)
| collection | fixed | auto-λ | RW-1 | beats? |
|---|:--:|:--:|:--:|:--:|
| OPPORTUNITY | 0.125 | 0.125 | 0.138 | no |
| GECCO | 0.380 | 0.383 | 0.639 | no |
| CreditCard | 0.025 | 0.025 | 0.111 | no |

auto-λ essentially unchanged — the MC-dropout gate is weak (gate→label AUC ≤ 0.51), so
there is little for the controller to amplify.

## Interpretability
The MC-dropout confidence localizes poorly (gate→label AUC 0.51 on GECCO, 0.18 on
OPPORTUNITY) — much weaker than P1's residual gate (0.90). On GECCO correction still
concentrates (6.4×), but that is largely RW-1-native given the near-random gate. P2 is
the weakest on GECCO of all five: the uncertainty signal does not help and drags the
gate. Confirms the docx's MC-dropout-miscalibration risk.

## Cost
Per-series wall-clock ~1.2× P1 (M=5 MC-dropout forward passes for the uncertainty
signal; grows with #features). Indicative.

## Decision
0/3; the uncertainty axis does not help. Weakest GECCO of the five.

## Reproduce
```bash
sbatch experiments/proposals/submit_rerun_all.sh
python experiments/proposals/aggregate_collection.py --proposal 2
```
