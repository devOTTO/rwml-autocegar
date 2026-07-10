# Proposal 5 — Temporal-Persistence Confident-Error CEGAR: Results

**Verdict: P5 is the only proposal that beats the best-HP/200ep RW-1 on a verdict
collection — GECCO, robustly with auto-λ (6/6 runs, min 0.661 > 0.639). Fixed-λ ties.
But the win does NOT generalize across the shape spectrum.**

## What Proposal 5 is (docx-faithful)
Gate only errors that PERSIST over neighbouring time windows, not isolated spikes.
`m_t = 1(e_t>τ_e)` (τ_e = residual q_e quantile), `p_t = mean(m_{t-h..t+h})` (temporal
persistence, epoch-end), `g = σ(k(e_t−τ_e)/mad)·σ(k_p(p_t−τ_p))`. Previous-epoch gate
drives amplification (ScaleGrad). Variants h5 / h25. Score = mean|correction|.

## Methodology (corrected config)
`epochs=100`, `warmup=10`, `correction_init='neg_x'`, variant `h5`, fixed `lam=1`.
Warm-up = plain RW-1 then gate on. Unit = whole collection. Baseline = reproduction
best-HP/200ep. **Caveat**: Δ config-confounded (epoch/HP) → indicative.
**Cost** (gecco, 100ep): low — a moving-average of the residual indicator, no extra
forward (cheapest of the gated proposals).

## Verdict set (AUC-PR; fixed / auto-λ)
| collection | n | DeepAnT* | RW-1* | P5 fixed | auto-λ | **Δ (fixed−RW-1)** | P5 AUC-ROC |
|---|:-:|:--:|:--:|:--:|:--:|:--:|:--:|
| OPPORTUNITY | 8 | 0.272 | 0.138 | 0.114 | 0.111 | **−0.025** | 0.683 |
| GECCO | 1 | 0.454 | 0.639 | **0.643** | **0.677** | **+0.004 / +0.038** | 0.953 |
| CreditCard | 1 | 0.147 | 0.111 | 0.029 | 0.032 | **−0.082** | 0.629 |

**Beats RW-1 on GECCO (1/3)** — the only proposal to win any verdict collection.

## GECCO robustness (6 runs each, no fixed seed)
| | runs | mean | > RW-1 0.639 |
|---|---|:--:|:--:|
| fixed-λ | 0.633, 0.635, 0.639, 0.648, 0.648, 0.652 | 0.643 | 4/6 (min 0.633 — **tie**) |
| **auto-λ** | 0.661, 0.672, 0.672, 0.682, 0.683, 0.690 | **0.677** | **6/6 (min 0.661 — robust win)** |

Fixed P5 straddles RW-1 (statistical tie); **auto-λ P5 beats it on every run** — a real,
reproducible win, not no-seed noise.

## Shape spectrum (AUC-PR; fixed / auto-λ; W = beats RW-1)
| TAO (point, RW.995) | PSM (mixed, RW.137) | MSL (block, RW.131) | SWaT (block, RW.444) |
|:--:|:--:|:--:|:--:|
| 0.996 / 0.995 W | 0.124 / 0.130 | **0.137 W** / 0.130 | 0.141 / 0.154 |

**The GECCO win does not generalize.** TAO is a trivial tie (both saturate); MSL edges by
~0.005 (noise level); **SWaT (block) is a heavy loss (0.14 vs 0.444)**; PSM below. P5 is
NOT a general "block-anomaly" method — it loses on the extreme-block SWaT and near-ties
the block OPPORTUNITY.

## Correction diagnostics (thesis §8.4, fixed)
| collection | gate→label AUC | corr@anom/norm | Overlap | AnomalyCoverage |
|---|:--:|:--:|:--:|:--:|
| GECCO | 0.945 | 12.55 | 0.219 | 0.877 |
| CreditCard | 0.899 | 1.75 | 0.008 | 0.222 |
| OPPORTUNITY | 0.684 | 1.08 | 0.118 | 0.134 |

P5 has the best gate localization (GECCO 0.945) and correction concentration/coverage of
the five — consistent with it being the one that wins on GECCO.

## Decision
**End of the P1–P5 arc.** The corrected config overturns the old "gating never helps"
verdict; P5 + auto-λ is a genuine, reproducible win over the tuned RW-1 on GECCO. However
it does not generalize across the shape spectrum (SWaT block lost, MSL within noise,
point/mixed ≈/below RW-1), and all deltas remain config-confounded on the epoch/HP axis.
Net: a scoped positive (GECCO) on a corrected-config negative-results arc.

## Reproduce
```bash
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
sbatch experiments/proposals/runs/submit_p5_coll.sh
python experiments/proposals/aggregate_collection.py --proposal 5
```
