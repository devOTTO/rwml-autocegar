# Proposal 1 — Residual-Gated RW-CEGAR: Results

**Verdict: P1 does not beat the best-HP/200ep RW-1 on the verdict set (0/3), but under
the corrected config it is competitive (OPPORTUNITY near-tie, GECCO close). The old
"clear loss" was largely a config artifact.**

## What Proposal 1 is
Simplest CEGAR × RW-ML hybrid on the reproduced RW-1 (Algorithm 2); overrides only the
signals: wrongness `E_t = σ(k·(robust_z − τ))`, confidence `C_t = 1` (basic); gate
`g = E_t·C_t`, scale `s = 1+λg` applied to the per-window forecasting-loss gradient via
ScaleGrad. Score = `mean|correction|`.

## Methodology (corrected config)
`epochs=100`, `warmup=10`, `correction_init='neg_x'`, fixed `lam=1, tau=2`. **Warm-up =
plain RW-1 (correction trained, gate OFF), then the gate switches on** — model starts as
RW-1. (Replaces the earlier forecaster-only warm-up + zero-init confound, archived in
`_backup_oldconfig/`.) Unit = whole collection (opportunity mean over 8 series). Baseline
RW-1 / DeepAnT = reproduction per-collection means (best-HP/200ep).
**Caveat**: Δ is config-confounded on the epoch/HP axis (100ep/default-HP vs best-HP/200ep)
→ indicative. **Cost** (gecco, 100ep): ~5 min, the cheapest (1 forward/window, like RW-1).

## Verdict set (AUC-PR; fixed / auto-λ)
| collection | n | DeepAnT* | RW-1* | P1 fixed | auto-λ | **Δ (fixed−RW-1)** | P1 AUC-ROC |
|---|:-:|:--:|:--:|:--:|:--:|:--:|:--:|
| OPPORTUNITY | 8 | 0.272 | 0.138 | 0.123 | 0.113 | **−0.015** | 0.703 |
| GECCO | 1 | 0.454 | 0.639 | 0.565 | 0.618 | **−0.073** | 0.918 |
| CreditCard | 1 | 0.147 | 0.111 | 0.032 | 0.035 | **−0.108** | 0.716 |

Beats RW-1 on **0/3** (OPPORTUNITY near-tie). auto-λ lifts GECCO (0.565→0.618).

## Shape spectrum (AUC-PR; fixed / auto-λ; W = beats RW-1)
| TAO (point, RW.995) | PSM (mixed, RW.137) | MSL (block, RW.131) | SWaT (block, RW.444) |
|:--:|:--:|:--:|:--:|
| 0.995 / 0.995 W | 0.125 / 0.126 | **0.136 W** / 0.118 | 0.139 / 0.131 |

Ties on TAO (both saturate), edges MSL (weak baseline, margin ~noise), loses SWaT/PSM.

## Correction diagnostics (thesis §8.4, fixed)
| collection | gate→label AUC | corr@anom/norm | Overlap | AnomalyCoverage |
|---|:--:|:--:|:--:|:--:|
| GECCO | 0.896 | 11.60 | 0.194 | 0.776 |
| CreditCard | 0.951 | 1.84 | 0.011 | 0.315 |
| OPPORTUNITY | 0.480 | 1.09 | 0.126 | 0.145 |

## Interpretability
On GECCO the residual gate localizes anomalies (AUC 0.90) and correction concentrates
there (11.6×, 78% coverage); with the corrected warm-up this now coincides with a high
AUC-PR (0.565) rather than the depressed old value. On opportunity the gate is
uninformative (0.48) yet P1 near-ties RW-1. CreditCard (point) stays weak.

## Decision
Competitive but does not beat tuned RW-1 → fail-fast to Proposal 2. The "gating clearly
loses" reading of the old config does not hold.

## Reproduce
```bash
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
sbatch experiments/proposals/runs/submit_p1_coll.sh
python experiments/proposals/aggregate_collection.py --proposal 1
```
