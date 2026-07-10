# Proposal 1 — Residual-Gated RW-CEGAR: Results

**Verdict: P1 does not beat the best-HP/200ep RW-1 reproduction on any verdict
collection (0/3), but under the corrected config it comes very close on OPPORTUNITY
(Δ−0.015) and GECCO (Δ−0.074) — far better than the old-config run.**

## Config note (important)
These results use the **corrected training config**: warm-up runs **plain RW-1**
(correction trained, CEGAR gate OFF) for `warmup=10`, then the gate switches on;
`correction_init='neg_x'` (Algorithm-2 faithful, matching the RW-1 baseline). The
earlier docs (now in `_backup_oldconfig/`) used forecaster-only warm-up + zero init,
which depressed every proposal — a config artifact, not a property of the gate. The
delta is still **config-confounded on one axis**: proposals are 100ep/default-HP while
the RW-1 baseline is best-HP/200ep (indicative, not a clean isolation of the gate).

## What Proposal 1 is
Simplest CEGAR × RW-ML hybrid on the reproduced RW-1 (Algorithm 2). Overrides only
the signals:
- **Wrongness** `E_t = sigmoid(k·(robust_z − τ))`, `robust_z = (r − median)/MAD`
- **Confidence** `C_t = 1` (basic) or tail-quantile (selective)
- **Gate** `g = E_t·C_t`, **scale** `s = 1+λg` on the per-window forecasting-loss
  gradient via `ScaleGrad`. Score = `mean|correction|`.

## Methodology (collection-level)
Verdict set: opportunity (8 series) + gecco (1) + creditcard (1) = 10 series, averaged
per collection. `epochs=100`, `warmup=10`, fixed HP (`lam=1, tau=2, k=1`), variant
`basic`. RW-1 / DeepAnT = reproduction per-collection means (best-HP/200ep, reference).

## Collection-level results (fixed λ)

`*` DeepAnT / RW-1 = reproduction per-collection means (reference). Δ = P1 − RW-1.

| collection | n | DeepAnT AUC-PR* | RW-1 AUC-PR* | P1 AUC-PR | **Δ (P1−RW-1)** | P1 AUC-ROC |
|---|:-:|:--:|:--:|:--:|:--:|:--:|
| OPPORTUNITY | 8 | 0.272 | 0.138 | 0.123 | **−0.015** | 0.703 |
| GECCO | 1 | 0.454 | 0.639 | 0.565 | **−0.074** | 0.918 |
| CreditCard | 1 | 0.147 | 0.111 | 0.032 | **−0.079** | 0.716 |

**P1 beats RW-1 on 0/3**, but near-ties on OPPORTUNITY and closes most of the GECCO gap.

### Correction diagnostics (thesis §8.4, fixed λ)
`Overlap` = of top-5% |correction| steps, fraction anomalous; `AnomalyCoverage` = of
anomalies, fraction in the top-5%; `corr@anom/norm` = anomaly/normal correction ratio;
`gate→label AUC` = gate localization.

| collection | gate→label AUC | corr@anom/norm | Overlap | AnomalyCoverage |
|---|:--:|:--:|:--:|:--:|
| GECCO | 0.896 | 11.60 | 0.194 | 0.776 |
| CreditCard | 0.951 | 1.85 | 0.011 | 0.315 |
| OPPORTUNITY | 0.480 | 1.09 | 0.126 | 0.145 |

## Auto-tuning ablation (auto-λ, `lam_mode=auto_tr`)
| collection | fixed λ | auto-λ | RW-1 | beats? |
|---|:--:|:--:|:--:|:--:|
| OPPORTUNITY | 0.123 | 0.113 | 0.138 | no |
| GECCO | 0.565 | 0.618 | 0.639 | no (closes to −0.021) |
| CreditCard | 0.032 | 0.035 | 0.111 | no |

auto-λ helps on GECCO (0.565→0.618) but still 0/3.

## Interpretability
On GECCO the gate localizes anomalies (gate→label AUC 0.90) and correction concentrates
there (11.6×, covering 78%). Under the corrected config P1 is close to RW-1 on
GECCO/OPPORTUNITY; CreditCard (isolated point anomalies) is where it fails — a
`mean|correction|` score struggles to rank single-point anomalies.

## Cost
Per-series wall-clock ~1–14 min (100ep, ∝ series length); P1 is the cheapest (1 forward
pass/window, same as RW-1). Indicative, single no-seed run.

## Decision
Under the corrected config P1 is competitive but does not surpass the tuned RW-1
ceiling (0/3). See P5 for the one proposal that beats RW-1 (GECCO).

## Reproduce
```bash
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
sbatch experiments/proposals/submit_rerun_all.sh          # P1-P5 verdict, corrected config
python experiments/proposals/aggregate_collection.py --proposal 1
```
