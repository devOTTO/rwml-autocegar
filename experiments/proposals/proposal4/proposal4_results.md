# Proposal 4 — Dual-Gate Residual-and-Gradient RW-CEGAR: Results

**Verdict: P4 does not beat the best-HP/200ep RW-1 reproduction (0/3), but it is
very close on GECCO — fixed 0.599 (Δ−0.040), auto-λ 0.628 (Δ−0.011) — the second-best
GECCO after P5.**

## Config note
Corrected config: warm-up = **plain RW-1** (gate OFF) then gate ON; `correction_init
='neg_x'`. Delta config-confounded on the epoch/HP axis (100ep/default-HP vs
best-HP/200ep).

## What Proposal 4 is (docx-faithful gate, amplification-only)
A confident error = high residual AND gradient-correctable (small input changes
strongly reduce the loss).
```
g_res  = σ(k_r·(robust_z(resid) − τ_r))
h_t    = ‖∂loss/∂input‖ (per window)        # input-gradient correctability
g_grad = σ(k_h·(robust_z(h_t) − τ_h))
g      = g_res · g_grad
```
`h_t` is an extra fwd+bwd w.r.t. the input each batch; `benefit` variant uses the
estimated loss reduction instead of ‖∇‖. **Stage-1: amplification only** (docx's
normalized write-back not implemented). Shared hooks base.

## Methodology (collection-level)
Verdict set (10 series), `epochs=100`, `warmup=10`, variant `gradnorm`, fixed HP.
RW-1/DeepAnT = reproduction means.

## Collection-level results (fixed)

| collection | n | DeepAnT AUC-PR* | RW-1 AUC-PR* | P4 AUC-PR | **Δ (P4−RW-1)** | P4 AUC-ROC |
|---|:-:|:--:|:--:|:--:|:--:|:--:|
| OPPORTUNITY | 8 | 0.272 | 0.138 | 0.107 | **−0.031** | 0.671 |
| GECCO | 1 | 0.454 | 0.639 | 0.599 | **−0.040** | 0.935 |
| CreditCard | 1 | 0.147 | 0.111 | 0.026 | **−0.085** | 0.637 |

**P4 beats RW-1 on 0/3**, but GECCO is very close (Δ−0.040 fixed, −0.011 auto).

### Correction diagnostics (thesis §8.4, fixed)
| collection | gate→label AUC | corr@anom/norm | Overlap | AnomalyCoverage |
|---|:--:|:--:|:--:|:--:|
| GECCO | 0.810 | 10.40 | 0.209 | 0.840 |
| CreditCard | 0.839 | 1.70 | 0.008 | 0.246 |
| OPPORTUNITY | 0.495 | 1.08 | 0.105 | 0.120 |

## Auto-tuning ablation (auto-λ)
| collection | fixed | auto-λ | RW-1 | beats? |
|---|:--:|:--:|:--:|:--:|
| OPPORTUNITY | 0.107 | 0.110 | 0.138 | no |
| GECCO | 0.599 | 0.628 | 0.639 | no (Δ−0.011) |
| CreditCard | 0.026 | 0.025 | 0.111 | no |

auto-λ nearly closes the GECCO gap (0.599→0.628, Δ−0.011).

## Interpretability
The dual gate localizes well on GECCO (gate→label AUC 0.81, corr 10.4×, coverage 0.84)
and gets P4 to within ~0.01–0.04 of tuned RW-1 there. The input-gradient signal
(`gradnorm`) is the primary novelty; on the injected-anomaly smoke test the `benefit`
variant localized better than `gradnorm` (which, per the docx risk, also spikes at
noise/discontinuities). Known minor: input-gradient forward runs in train mode (dropout
noise).

## Cost
Medium — one extra input-gradient backward per batch (~1.5–2× P1). Indicative.

## Decision
0/3 but the closest non-winner on GECCO; the correctability gate is a strong trigger.

## Reproduce
```bash
sbatch experiments/proposals/submit_rerun_all.sh
python experiments/proposals/aggregate_collection.py --proposal 4
```
