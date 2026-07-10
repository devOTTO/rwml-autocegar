# Proposal 3 — RW-Correction-Consistency CEGAR: Results

**Verdict: P3 does not beat the best-HP/200ep RW-1 reproduction on any verdict
collection (0/3). Near-ties on OPPORTUNITY (Δ−0.003) but stays low on GECCO (0.379).**

## Config note
Corrected config: warm-up = **plain RW-1** (gate OFF) then gate ON; `correction_init
='neg_x'`. Old-config docs in `_backup_oldconfig/`. Delta config-confounded on the
epoch/HP axis (100ep/default-HP vs best-HP/200ep).

## What Proposal 3 is (docx-faithful, full)
Signal = the RW correction's own behaviour. A point corrected large AND in a consistent
direction over epochs is a confident anomaly candidate; P3 preserves the correction
there instead of reshaping it away.
```
d_t = mean_feat|C_t|;  v_t = cos(ΔC^e, ΔC^{e-1})
g   = σ(k_d(d−τ_d)/sd) · σ(k_v(v−τ_v)),  EMA-smoothed
```
Both docx mechanisms: gradient amplification (ScaleGrad, prev-epoch gate) + preserve
write-back `grad_C·(1−γg)`. Variants: `full` (default), `preserve_only` (λ=0), `soft`.
On the shared hooks base (`rw_cegar_hooks.py`).

## Methodology (collection-level)
Verdict set (10 series), `epochs=100`, `warmup=10`, variant `full`, fixed HP
(`gamma=0.9, corr_q=0.95, k_d=1, k_v=5`). RW-1/DeepAnT = reproduction means.

## Collection-level results (fixed)

| collection | n | DeepAnT AUC-PR* | RW-1 AUC-PR* | P3 AUC-PR | **Δ (P3−RW-1)** | P3 AUC-ROC |
|---|:-:|:--:|:--:|:--:|:--:|:--:|
| OPPORTUNITY | 8 | 0.272 | 0.138 | 0.135 | **−0.003** | 0.717 |
| GECCO | 1 | 0.454 | 0.639 | 0.379 | **−0.260** | 0.841 |
| CreditCard | 1 | 0.147 | 0.111 | 0.027 | **−0.084** | 0.614 |

**P3 beats RW-1 on 0/3** (closest of any proposal on OPPORTUNITY, Δ−0.003).

### Correction diagnostics (thesis §8.4, fixed)
| collection | gate→label AUC | corr@anom/norm | Overlap | AnomalyCoverage |
|---|:--:|:--:|:--:|:--:|
| GECCO | 0.829 | 7.51 | 0.152 | 0.610 |
| CreditCard | 0.576 | 1.70 | 0.007 | 0.213 |
| OPPORTUNITY | 0.359 | 1.10 | 0.137 | 0.167 |

## Auto-tuning ablation (auto-λ)
| collection | fixed | auto-λ | RW-1 | beats? |
|---|:--:|:--:|:--:|:--:|
| OPPORTUNITY | 0.135 | 0.131 | 0.138 | no |
| GECCO | 0.379 | 0.386 | 0.639 | no |
| CreditCard | 0.027 | 0.026 | 0.111 | no |

Near-unchanged with auto-λ.

## Interpretability
The consistency gate localizes reasonably on GECCO (0.83) and preserve keeps correction
on anomalies (7.5×). It nearly ties RW-1 on OPPORTUNITY but the preserve mechanism does
not lift GECCO to RW-1's level. `preserve_only` / `soft` variants available for ablation.

## Cost
Most expensive of the correction-hook proposals (epoch-end direction-stability + preserve
write-back on top of RW-1). Indicative.

## Decision
0/3; preserve is competitive on OPPORTUNITY but does not surpass tuned RW-1.

## Reproduce
```bash
sbatch experiments/proposals/submit_rerun_all.sh
python experiments/proposals/aggregate_collection.py --proposal 3
```
