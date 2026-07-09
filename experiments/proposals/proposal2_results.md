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

**Cost** (gecco, 100 epochs, GPU wall-clock; indicative, single no-seed run):
P1 5:05 (1.0×) · **P2 6:02 (1.19×)** · P3 6:44 (1.32×). P2's overhead is the M=5
MC-dropout forward passes for the uncertainty signal, and grows with the number of
features (gecco has 9; a wide series like opportunity/248-feat costs more).

## Collection-level results

`*` DeepAnT / RW-1 = reproduction per-collection means (reference). Δ = P2 − RW-1.

| collection | n | DeepAnT AUC-PR* | RW-1 AUC-PR* | P2 AUC-PR | **Δ (P2−RW-1)** | P2 AUC-ROC |
|---|:-:|:--:|:--:|:--:|:--:|:--:|
| GECCO | 1 | 0.454 | 0.639 | 0.221 | **−0.418** | 0.799 |
| SMD | 22 | 0.396 | 0.233 | 0.038 | **−0.195** | 0.484 |
| CreditCard | 1 | 0.147 | 0.111 | 0.002 | **−0.110** | 0.440 |
| OPPORTUNITY | 8 | 0.272 | 0.138 | 0.033 | **−0.106** | 0.345 |
| SMAP | 27 | 0.146 | 0.119 | 0.066 | **−0.053** | 0.575 |
| MITDB | 13 | 0.166 | 0.089 | 0.085 | **−0.004** | 0.600 |

**P2 beats RW-1 on 0/6 collections.** All rows are from the single §8.4 re-run
(consistent with the diagnostics below).

### Correction diagnostics (thesis §8.4, all collections)
`Overlap` = of top-5% |correction| timesteps, fraction anomalous (precision);
`AnomalyCoverage` = of anomalies, fraction in the top-5% |correction| (recall);
`corr@anom/norm` = anomaly-vs-normal correction ratio; `gate→label AUC` = gate
localization. Contrast with P1, whose GECCO gate localizes at AUC 0.90.

| collection | gate→label AUC | corr@anom/norm | Overlap | AnomalyCoverage |
|---|:--:|:--:|:--:|:--:|
| GECCO | 0.531 | 2.60 | 0.128 | 0.512 |
| MITDB | — | 1.84 | 0.067 | 0.191 |
| SMAP | — | 1.09 | 0.068 | 0.122 |
| SMD | — | 0.99 | 0.036 | 0.047 |
| OPPORTUNITY | 0.171 | 0.93 | 0.026 | 0.025 |
| CreditCard | 0.433 | 0.98 | 0.001 | 0.035 |

The gate never localizes (verdict-set AUC ≤ 0.53, vs P1's 0.90 on GECCO). Only GECCO
and MITDB show above-1 correction concentration (2.6× / 1.8×), but on GECCO the gate
is near-random (0.53), so that concentration is RW-1-native residual growth, not
gate-driven targeting — the same reading as the per-series diagnostics below.
(`gate→label AUC` shown for the verdict set; characterization rows report the
correction diagnostics only.)

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

## Auto-tuning ablation (verdict set)
Turned on Auto-CEGAR's controllers (kept off elsewhere): `lam_mode=auto_tr` (auto-λ)
+ `tau_mode=auto_q_valley` (auto-τ).

| collection | auto-λ/τ AUC-PR | RW-1 | beats RW-1? |
|---|:--:|:--:|:--:|
| GECCO | 0.194 | 0.639 | no |
| OPPORTUNITY | 0.032 | 0.138 | no |
| CreditCard | 0.002 | 0.111 | no |

Still **0/3** (within run-to-run variance of the fixed verdict: GECCO 0.205,
OPPORTUNITY 0.034, CreditCard 0.002; no fixed seed). Caveats: auto-λ has nothing to
amplify (the gate is uninformative), and **auto-τ is not a clean test for P2** — the
valley controller sets τ in raw-residual-quantile units, but P2's wrongness uses the
dimensionless standardized residual `e_t = ‖Y−μ‖/√(u+ε)`, so the τ it picks is in the
wrong units. Either way, the verdict is unchanged.

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
