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

## Interpretability — why P2 loses (per-series id_1 diagnostics)

| collection (id_1) | gate→label AUC | corr @anom/norm | trigger frac | Δ AUC-PR | regime |
|---|:-:|:-:|:-:|:-:|---|
| GECCO | 0.525 | **2.81** | ≈0.00 | −0.434 | most correction concentration, but near-random gate (0.53) so concentration is likely RW-1-native, not gate-driven |
| SMD | 0.23 | 1.00 | ≈0.00 | −0.195 | gate anti-aligned; correction degraded broadly |
| CreditCard | 0.463 | 0.99 | ≈0.00 | −0.110 | near-neutral gate; fragile RW-1 score collapses |
| OPPORTUNITY | 0.209 | 0.89 | ≈0.00 | −0.104 | gate anti-aligned with anomalies |
| SMAP | 0.45 | 1.29 | ≈0.00 | −0.050 | near-neutral |
| MITDB | 0.44 | 1.19 | ≈0.00 | −0.002 | near-neutral → nearly tied |

**Reading.** The MC-dropout confidence is uniformly weak: trigger_frac ≈ 0 on every
series, and gate→label AUC exceeds 0.5 on only 17/72 series (even GECCO's 0.53 is
barely above random, versus P1's 0.90). So P2's gate does not localize anomalies on
any collection, and the per-collection losses are not attributable to a P1-style,
gate-driven erase.
- GECCO has the most correction concentration (corr@anom/norm 2.8×), but with a
  near-random gate (AUC 0.53) this most likely reflects RW-1's native residual-driven
  correction growth, not gate-induced targeting. Confirming it would need RW-1's own
  corr@anom/norm, which we did not measure.
- SMAP / MITDB: near-neutral gate, smallest losses (−0.05, −0.002).
- OPPORTUNITY / SMD: the gate anti-aligns with anomalies (AUC 0.21 / 0.23) and P2
  degrades the correction broadly; the mechanism is unclear.

Net: the uncertainty confidence never helps. Because the reference RW-1 is
best-HP/200ep while P2 is fixed-HP/100ep, the exact loss magnitudes are indicative
rather than mechanism-attributable; on GECCO the co-trained RW-1@100 (≈0.674) is not
below the reproduction RW-1@200 (0.639), so the epoch gap does not inflate the
baseline. This confirms the doc's MC-dropout-miscalibration risk. (Contrast P1 on
GECCO: gate→label AUC ≈ 0.90, correction ≈ 5.5×, genuinely stronger targeting.)

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
