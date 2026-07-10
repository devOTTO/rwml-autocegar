# Auto-CEGAR proposals P1–P5 — cross-proposal summary

**Headline: under the corrected config, P5 (temporal-persistence) with auto-λ robustly
beats the best-HP/200ep RW-1 on GECCO (6/6 runs, min 0.661 > 0.639). Fixed-λ P5 ties.
The other proposals are competitive but do not beat RW-1 on the verdict set.**

## Config (corrected)
- Warm-up = **plain RW-1** (correction trained, gate OFF) for `warmup=10`, then the CEGAR
  gate switches on. `correction_init='neg_x'` (Algorithm-2 faithful). `epochs=100`.
- This replaces the earlier **forecaster-only warm-up + zero init**, which was a confound
  that depressed all proposals (old "0/N loss" — archived in `_backup_oldconfig/`).
- **Caveat**: the Δ vs RW-1 is still config-confounded on one axis — proposals run
  100ep/default-HP, the RW-1 baseline is best-HP/200ep — so magnitudes are indicative.
- Evaluation unit = whole collection (opportunity = mean of 8 series; gecco/creditcard n=1).

## Verdict set (AUC-PR; fixed / auto-λ)
RW-1 / DeepAnT = reproduction per-collection means (best-HP/200ep). **bold** = beats RW-1.

| proposal | OPPORTUNITY (RW-1 0.138) | GECCO (RW-1 0.639) | CreditCard (RW-1 0.111) |
|---|:--:|:--:|:--:|
| DeepAnT* | 0.272 | 0.454 | 0.147 |
| P1 residual | 0.123 / 0.113 | 0.565 / 0.618 | 0.032 / 0.035 |
| P2 uncertainty | 0.125 / 0.125 | 0.380 / 0.383 | 0.025 / 0.025 |
| P3 consistency | 0.135 / 0.131 | 0.379 / 0.386 | 0.027 / 0.026 |
| P4 dual-grad | 0.107 / 0.110 | 0.599 / 0.628 | 0.026 / 0.025 |
| **P5 persistence** | 0.114 / 0.111 | **0.643 / 0.677** | 0.029 / 0.032 |

- **OPPORTUNITY**: all proposals near-tie RW-1 (Δ −0.004…−0.031) — big lift from the old config.
- **GECCO**: P5 wins; P4 close (−0.040); P1 close (−0.073). auto-λ lifts GECCO for P1/P4/P5.
- **CreditCard** (isolated point anomalies): everyone loses clearly — correction-magnitude
  scoring is weak for point anomalies (RW-1 itself is only 0.111).

## P5 GECCO robustness (6 runs each, no fixed seed)
| | runs | mean | > RW-1 0.639 |
|---|---|:--:|:--:|
| fixed-λ | 0.633, 0.635, 0.639, 0.648, 0.648, 0.652 | 0.643 | 4/6 (min 0.633 — **tie**) |
| **auto-λ** | 0.661, 0.672, 0.672, 0.682, 0.683, 0.690 | **0.677** | **6/6 (min 0.661 — robust win)** |

→ Fixed P5 straddles RW-1 (statistical tie). **auto-λ P5 beats RW-1 on every run** — a real,
consistent win, not no-seed noise.

## Correction diagnostics (thesis §8.4, fixed, GECCO)
| proposal | gate→label AUC | corr@anom/norm | Overlap (prec) | Coverage (recall) |
|---|:--:|:--:|:--:|:--:|
| P5 | 0.945 | 12.55 | 0.219 | 0.877 |
| P1 | 0.896 | 11.60 | 0.194 | 0.776 |
| P4 | 0.809 | 10.40 | 0.209 | 0.840 |
| P3 | 0.829 | 7.51 | 0.152 | 0.610 |
| P2 | 0.510 | 6.43 | 0.154 | 0.617 |

P5 has the best gate localization (0.945) and correction concentration/coverage on GECCO —
consistent with it being the one that wins.

## Shape-spectrum extension (TAO/PSM/MSL/SWaT) — IN PROGRESS
Testing whether the pattern is shape-dependent (P5 wins on block, loses on point/mixed).
Partial (P1 done):
- **MSL** (block, RW-1 weak 0.131): **P1 = 0.136 → beats.** ← block + beatable baseline
- **TAO** (point, RW-1 0.995): P1 = 0.995 tie (both ~perfect).
- **PSM** (mixed): P1 0.125 vs 0.137 (close).
- P2–P5 shape runs still queued; table to be completed.

## Decision (interim)
The corrected config overturns the old "gating never helps" reading: proposals are
competitive, and **P5+auto-λ genuinely beats the tuned RW-1 on GECCO (block)**. The open
question — does P5 win on block anomalies generally (GECCO/MSL/SWaT) but not point/mixed
(TAO/PSM)? — is being answered by the shape extension.
