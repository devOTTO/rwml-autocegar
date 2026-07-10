# Proposal 3 — RW-Correction-Consistency CEGAR: Results

**Verdict: P3 does not beat the best-HP/200ep RW-1 (0/3). Closest near-tie on OPPORTUNITY
(Δ−0.003); GECCO well below.**

## What Proposal 3 is (docx-faithful, full)
Signal = the correction's own behaviour. `d_t=mean_feat|C_t|`, `v_t=cos(ΔC^e,ΔC^{e-1})`,
`g=σ(k_d(d−τ_d)/sd)·σ(k_v(v−τ_v))`, EMA-smoothed over epochs. Drives gradient amplification
(previous-epoch gate → ScaleGrad) AND a preserve write-back `grad_C·(1−γg)`.

## Experiment settings
| group | values |
|---|---|
| training | `epochs=100`, `warmup=10` (**plain RW-1, gate OFF; gate on after**), `correction_init='neg_x'` |
| RW-1 base | `window=50`, `batch=256`, `l1_weight=0.001`, `activation=linear`, `correction_rate=0.1` |
| gate | `λ=1` (fixed) **or** `lam_mode='auto_tr'`; `gamma=0.9`, `corr_q=0.95`, `k_d=1`, `k_v=5`, `tau_v=0`, `persist_alpha=0.9` |
| variant | `full` (amplify + preserve) |
| eval | whole collection; **no fixed seed** (1 run/cell) |
| baseline | reproduction best-HP/200ep → Δ config-confounded (indicative) |

## Results — all collections (AUC-PR; fixed / auto-λ)
`set`: V = verdict, E = extension. **W** = fixed beats RW-1.

| collection | shape | set | n | DeepAnT* | RW-1* | P3 fixed | auto-λ | Δ (fixed−RW-1) |
|---|:-:|:-:|:-:|:--:|:--:|:--:|:--:|:--:|
| GECCO | block | V | 1 | 0.454 | 0.639 | 0.379 | 0.386 | −0.260 |
| OPPORTUNITY | block | V | 8 | 0.272 | 0.138 | 0.135 | 0.131 | −0.003 |
| CreditCard | point | V | 1 | 0.147 | 0.111 | 0.027 | 0.026 | −0.084 |
| TAO | point | E | 13 | 0.996 | 0.995 | 0.996 | 0.996 | ≈0 (tie) |
| PSM | mixed | E | 1 | 0.407 | 0.137 | 0.118 | 0.118 | −0.019 |
| MSL | block | E | 16 | 0.116 | 0.131 | 0.128 | 0.130 | −0.003 |
| SWaT | block | E | 2 | 0.516 | 0.444 | 0.141 | 0.143 | −0.303 |

Beats RW-1 on **0/3** verdict; only the trivial TAO tie on the extension (loses MSL/SWaT/PSM).
AUC-ROC (fixed): OPP 0.717, GECCO 0.841, CC 0.614.

## Correction diagnostics (thesis §8.4, fixed)
| collection | gate→label AUC | corr@anom/norm | Overlap | Coverage |
|---|:--:|:--:|:--:|:--:|
| GECCO | 0.829 | 7.51 | 0.152 | 0.610 |
| CreditCard | 0.576 | 1.70 | 0.007 | 0.213 |
| OPPORTUNITY | 0.359 | 1.10 | 0.137 | 0.167 |

## Interpretability
The consistency gate localizes reasonably on GECCO (0.83) and preserves correction there
(7.5×), but P3 still trails the residual/dual-gradient proposals; the amplify-vs-preserve
pair nets out roughly neutral. Near-ties RW-1 on opportunity.

## Decision
Does not beat tuned RW-1 → move to Proposal 4.

## Reproduce
```bash
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
sbatch experiments/proposals/runs/submit_p3_coll.sh
python experiments/proposals/aggregate_collection.py --proposal 3
```
