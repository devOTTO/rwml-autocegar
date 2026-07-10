# Proposal 3 — RW-Correction-Consistency CEGAR: Results

**Verdict: P3 loses to the RW-1 baseline on all 3 verdict collections, in every
variant (full / preserve-only / auto-λ) → fail-fast to Proposal 4.**

## What Proposal 3 is (docx-faithful, full)
The signal is the RW correction's OWN behaviour, not the residual (P1) or predictive
uncertainty (P2). A point corrected large AND in a consistent direction over epochs is
a "confident RW anomaly candidate"; P3 PRESERVES the correction there (as evidence)
instead of reshaping it away — the opposite of P1/P2's "correct the anomaly away".

```
d_t   = mean_feat |C_t|                                   # correction magnitude
v_t   = cos(C_t^e − C_t^{e-1}, C_t^{e-1} − C_t^{e-2})     # epoch-to-epoch direction stability
g     = σ(k_d·(d−τ_d)/sd) · σ(k_v·(v−τ_v)),  τ_d = Q_corr_q(d)   # EMA-smoothed over epochs
```
Both docx mechanisms are on: (1) gradient amplification — the previous-epoch gate drives
the per-window ScaleGrad path (scale = 1+λ·g); (2) preserve write-back — the epoch-wise
RW step is suppressed on those points, `grad_C·(1−γ·g)` (docx "more safely" variant).
Implemented on a dedicated base (`rw_cegar_p3.py`) so the P1/P2 base stays clean.

## Methodology (collection-level)
Verdict set only (opportunity 8 + gecco 1 + creditcard 1 = 10 series), `epochs=100`,
`warmup=10`, fixed HP (`gamma=0.9, corr_q=0.95, k_d=1, k_v=5, tau_v=0`). RW-1 / DeepAnT
= reproduction per-collection means (reference, best-HP/200ep). Three variants:
`full` (amp+preserve, docx default), `preserve_only` (λ=0, write-back isolated),
`full@auto-λ` (`lam_mode=auto_tr`, the auto-tuning ablation). auto-τ is N/A (P3 does
not use the residual τ). Score = mean|correction| (same as RW-1). **Note the delta is
config-confounded**: the proposals run default-HP / 100ep, the RW-1 baseline is
best-HP / 200ep, so it is indicative — not a clean isolation of the gate's effect.

**Cost** (gecco, 100 epochs, GPU wall-clock; indicative, single no-seed run):
P1 5:05 (1.0×) · P2 6:02 (1.19×) · **P3 6:44 (1.32×)**. P3 is the most expensive of
the three — the epoch-end direction-stability (cos over [feats,T]) + gradient
amplification (ScaleGrad, λ>0) + preserve write-back stack on top of the RW-1 step.

## Collection-level results (primary = `full`)

`*` DeepAnT / RW-1 = reproduction per-collection means (reference). Δ = P3(full) − RW-1.

| collection | n | DeepAnT AUC-PR* | RW-1 AUC-PR* | P3 AUC-PR | **Δ (P3−RW-1)** | P3 AUC-ROC |
|---|:-:|:--:|:--:|:--:|:--:|:--:|
| OPPORTUNITY | 8 | 0.272 | 0.138 | 0.046 | **−0.092** | 0.385 |
| GECCO | 1 | 0.454 | 0.639 | 0.227 | **−0.412** | 0.813 |
| CreditCard | 1 | 0.147 | 0.111 | 0.002 | **−0.109** | 0.496 |

**P3 (full) beats RW-1 on 0/3 collections.**

### Variant comparison (AUC-PR) — amplification & auto-λ add nothing
| collection | preserve_only (λ=0) | full (amp+preserve) | full@auto-λ | RW-1 |
|---|:--:|:--:|:--:|:--:|
| OPPORTUNITY | 0.044 | 0.046 | 0.048 | 0.138 |
| GECCO | 0.193 | 0.227 | 0.223 | 0.639 |
| CreditCard | 0.002 | 0.002 | 0.002 | 0.111 |

All three variants lose 0/3. preserve_only ≈ full ≈ auto-λ — the gradient amplification
and the auto-λ controller barely move the score, so the amplify-vs-freeze tension the
docx leaves open resolves to "both negligible here".

### Correction diagnostics (thesis §8.4, `full`)
| collection | gate→label AUC | corr@anom/norm | Overlap | AnomalyCoverage |
|---|:--:|:--:|:--:|:--:|
| GECCO | 0.876 | 2.69 | 0.133 | 0.534 |
| CreditCard | 0.563 | 1.00 | 0.002 | 0.045 |
| OPPORTUNITY | 0.537 | 0.96 | 0.045 | 0.039 |

## Interpretability — the preserve mechanism works, but does not reach the tuned baseline
- **The gate localizes anomalies well** on GECCO (gate→label AUC **0.88**, ~P1's 0.90)
  and — unlike P2 — even on opportunity/creditcard it is above random (0.54–0.56),
  because the direction-stability term filters noise. So the correction-consistency
  signal is real.
- **Preserve behaves as designed**: on GECCO correction concentrates on anomalies
  (corr@anom/norm 2.7×, covering 53%). It is NOT erased.
- **Yet AUC-PR still falls short** (GECCO 0.227 vs the best-HP/200ep RW-1 0.639).
  Preserving the correction does not lift P3 to the RW-1 ceiling. We do NOT claim the
  gate itself degrades RW-1: proposal and baseline differ in config (default-HP/100ep
  vs best-HP/200ep) and no matched-config gate-off control was run, so the gap mixes
  gate effect and config effect. What is supported is only that P3-as-configured does
  not reach the tuned RW-1 baseline.
- **P3 is actually worse than P1 on GECCO** (0.227 vs P1's 0.486; both < RW-1 0.639).
  P1's aggressive gating drove a higher anomaly-vs-normal correction ratio (5.3× vs
  P3's 2.7×). A higher such ratio *coincides* with a higher AUC-PR here, but note
  AUC-PR is a ranking metric, so this is a suggestive association, not a proven
  monotonic law. The "erase is the problem" story is thus incomplete: gating the
  correction (amplify or preserve) does not beat the best-HP RW-1 reproduction (the
  documented baseline, `rw/reproduction/summary_rw1_besthp.csv`). Assessing whether the
  gate helps at matched config would need a same-config gate-off control, which we did
  not run — so we make no claim about the gate's effect, only that P3-as-configured
  does not reach the tuned RW-1 baseline.

## Decision
**Fail-fast → Proposal 4 (Dual-Gate Residual-and-Gradient RW-CEGAR).** P3, the
correction-consistency preserve, does not reach the best-HP/200ep RW-1 baseline on any
verdict collection, and the preserve mechanism — though it works mechanically (it
localizes and keeps correction on anomalies) — does not close that gap. Whether the
gate itself helps or hurts at matched config is out of scope: no gate-off control was
run and the delta vs the tuned RW-1 baseline is config-confounded.

## Reproduce
```bash
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
sbatch experiments/proposals/proposal3/submit_p3_coll.sh        # 30 tasks = 10 series x 3 variants
python experiments/proposals/aggregate_collection.py --proposal 3   # full-variant table
wandb sync --include-offline ./wandb/offline-run-<date>_*
```
