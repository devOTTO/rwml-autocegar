# Auto-CEGAR proposals P1–P5 — cross-proposal summary

**Headline: under the corrected config, P5 (temporal-persistence) with auto-λ robustly
beats the best-HP/200ep RW-1 on GECCO (6/6 runs, min 0.661 > 0.639). Fixed-λ P5 ties.
The other proposals are competitive but do not beat RW-1 on the screening set.**

## Config (corrected)
- Warm-up = **plain RW-1** (correction trained, gate OFF) for `warmup=10`, then the CEGAR
  gate switches on. `correction_init='neg_x'` (Algorithm-2 faithful). `epochs=100`.
- This replaces the earlier **forecaster-only warm-up + zero init**, which was a confound
  that depressed all proposals (old "0/N loss" — archived in `_backup_oldconfig/`).
- **Caveat**: the Δ vs RW-1 is still config-confounded on one axis — proposals run
  100ep/default-HP, the RW-1 baseline is best-HP/200ep — so magnitudes are indicative.
- Evaluation unit = whole collection (opportunity = mean of 8 series; gecco/creditcard n=1).

## Screening set (AUC-PR; fixed / auto-λ)
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

How to read (all computed per timestep in the FINAL training epoch, against the
ground-truth labels; labels are used for analysis only, never during training):

- **gate->label AUC**: ROC-AUC when the per-timestep gate activation is used as if it
  were an anomaly score. 0.5 = the gate fires randomly w.r.t. the true anomalies,
  1.0 = it fires exactly at them. Measures how well the gate LOCALIZES anomalies.
- **corr@anom/norm**: mean |correction| on anomaly timesteps / mean |correction| on
  normal timesteps. Since the anomaly score IS mean |correction|, this is the score
  contrast: e.g. 12.55 means anomalous points end up with 12.55x more correction than
  normal points (higher = better separation = higher AUC-PR, all else equal).
- **Overlap (prec)**: thesis Sec. 8.4 definition. A point is "high-correction" when its
  |correction| exceeds the series' own 95th percentile (tau_C). Overlap = fraction of
  high-correction points that are true anomalies (precision of the correction).
- **Coverage (recall)**: fraction of true anomaly points that are high-correction
  (recall of the correction; thesis Sec. 8.4 calls it AnomalyCoverage).

| proposal | gate→label AUC | corr@anom/norm | Overlap (prec) | Coverage (recall) |
|---|:--:|:--:|:--:|:--:|
| P5 | 0.945 | 12.55 | 0.219 | 0.877 |
| P1 | 0.896 | 11.60 | 0.194 | 0.776 |
| P4 | 0.809 | 10.40 | 0.209 | 0.840 |
| P3 | 0.829 | 7.51 | 0.152 | 0.610 |
| P2 | 0.510 | 6.43 | 0.154 | 0.617 |

P5 has the best gate localization (0.945) and correction concentration/coverage on GECCO —
consistent with it being the one that wins.

## Shape-spectrum extension (AUC-PR; fixed / auto-λ; **W** = beats RW-1)
P1-P5 × 4 collections chosen to span the anomaly-shape axes.

| proposal | TAO (point, RW-1 0.995) | PSM (mixed, RW-1 0.137) | MSL (block, RW-1 0.131) | SWaT (block, RW-1 0.444) |
|---|:--:|:--:|:--:|:--:|
| DeepAnT* | 0.996 | 0.407 | 0.116 | 0.516 |
| P1 | 0.995 / 0.995 **W** | 0.125 / 0.126 | **0.136 W** / 0.118 | 0.139 / 0.131 |
| P2 | 0.995 / 0.995 **W** | 0.116 / 0.115 | **0.135 W / 0.134 W** | 0.133 / 0.136 |
| P3 | 0.996 / 0.996 **W** | 0.118 / 0.118 | 0.128 / 0.130 | 0.141 / 0.143 |
| P4 | 0.995 / 0.995 | 0.125 / 0.128 | 0.122 / 0.121 | 0.143 / 0.149 |
| P5 | 0.996 / 0.995 **W** | 0.124 / 0.130 | **0.137 W** / 0.130 | 0.141 / 0.154 |

**The "P5 = block-anomaly method" hypothesis is NOT supported.** There is no clean
shape→win rule:
- **TAO** (point): everyone ≈ 0.995 = RW-1 — a trivial tie (both saturate on easy point
  anomalies), not a meaningful win.
- **MSL** (block, weak RW-1 0.131): P1/P2/P5 edge above by ~0.005 — margins at the
  no-seed noise level, not a confirmed win.
- **SWaT** (block, strong RW-1 0.444): **everyone collapses to ~0.13–0.15** — a heavy
  loss on a block collection, directly contradicting "P5 wins on block".
- **PSM** (mixed): all slightly below RW-1.

So P5 does **not** win on block in general (it loses SWaT badly and near-ties OPPORTUNITY);
its GECCO win does not generalize to a shape rule.

## Decision (final)
1. **The corrected config overturns the old "gating never helps" conclusion** — the earlier
   0/N was largely a forecaster-only-warm-up + zero-init artifact; with warm-up = plain
   RW-1 then gate (neg_x), all proposals are competitive with the tuned RW-1.
2. **P5 + auto-λ robustly beats the best-HP/200ep RW-1 on GECCO** (6/6 runs, min 0.661 >
   0.639) — a genuine, reproducible win.
3. **But it does not generalize**: across the shape spectrum there is no consistent
   pattern (SWaT block lost, MSL win within noise, point/mixed ≈ or below RW-1). The win
   is GECCO-specific, not a shape-driven method advantage.
4. **Caveat**: all deltas are still config-confounded on the epoch/HP axis (proposals
   100ep/default-HP vs RW-1 best-HP/200ep), so they are indicative.

**Net**: a scoped positive result (P5+auto beats tuned RW-1 on GECCO) on top of a
corrected-config negative-results arc (no general win across shapes/collections).
