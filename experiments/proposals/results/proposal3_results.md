# Proposal 3 — RW-Correction-Consistency CEGAR: Results

**Verdict: P3 does not beat the best-HP/200ep RW-1 on the verdict set (0/3). Under the
corrected config it near-ties on OPPORTUNITY; GECCO stays well below.**

## What Proposal 3 is (docx-faithful, full)
Signal = the correction's own behaviour. `d_t = mean_feat|C_t|` (magnitude),
`v_t = cos(ΔC^e, ΔC^{e-1})` (direction stability), `g = σ(k_d(d−τ_d)/sd)·σ(k_v(v−τ_v))`,
EMA-smoothed across epochs. Drives gradient amplification (previous-epoch gate → ScaleGrad)
AND a preserve write-back `grad_C·(1−γg)`. Variants full / preserve_only / soft.

## Methodology (corrected config)
`epochs=100`, `warmup=10`, `correction_init='neg_x'`, variant `full`, fixed `lam=1`.
Warm-up = plain RW-1 then gate on. Unit = whole collection. Baseline = reproduction
best-HP/200ep. **Caveat**: Δ config-confounded (epoch/HP) → indicative.
**Cost** (gecco, 100ep): ~7 min — the epoch-end direction-stability + amplification +
preserve write-back stack on top of the RW-1 step.

## Verdict set (AUC-PR; fixed / auto-λ)
| collection | n | DeepAnT* | RW-1* | P3 fixed | auto-λ | **Δ (fixed−RW-1)** | P3 AUC-ROC |
|---|:-:|:--:|:--:|:--:|:--:|:--:|:--:|
| OPPORTUNITY | 8 | 0.272 | 0.138 | 0.135 | 0.131 | **−0.004** | 0.717 |
| GECCO | 1 | 0.454 | 0.639 | 0.379 | 0.386 | **−0.260** | 0.841 |
| CreditCard | 1 | 0.147 | 0.111 | 0.027 | 0.026 | **−0.084** | 0.614 |

Beats RW-1 on **0/3** (OPPORTUNITY closest near-tie, Δ−0.004).

## Shape spectrum (AUC-PR; fixed / auto-λ; W = beats RW-1)
| TAO (point, RW.995) | PSM (mixed, RW.137) | MSL (block, RW.131) | SWaT (block, RW.444) |
|:--:|:--:|:--:|:--:|
| 0.996 / 0.996 W | 0.118 / 0.118 | 0.128 / 0.130 | 0.141 / 0.143 |

Only the trivial TAO tie; loses MSL/SWaT/PSM.

## Correction diagnostics (thesis §8.4, fixed)
| collection | gate→label AUC | corr@anom/norm | Overlap | AnomalyCoverage |
|---|:--:|:--:|:--:|:--:|
| GECCO | 0.829 | 7.51 | 0.152 | 0.610 |
| CreditCard | 0.576 | 1.70 | 0.007 | 0.213 |
| OPPORTUNITY | 0.359 | 1.10 | 0.137 | 0.167 |

## Interpretability
The consistency gate localizes reasonably on GECCO (0.83) and preserves correction there
(7.5×), but P3 still trails the residual/dual-gradient proposals on GECCO. The
amplify-vs-preserve pair (docx) nets out roughly neutral. Near-ties RW-1 on opportunity.

## Decision
Does not beat tuned RW-1 → fail-fast to Proposal 4.

## Reproduce
```bash
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
sbatch experiments/proposals/runs/submit_p3_coll.sh
python experiments/proposals/aggregate_collection.py --proposal 3
```
