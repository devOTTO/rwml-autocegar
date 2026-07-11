# Proposal 1 — Residual-Gated RW-CEGAR: Results

**Verdict: P1 does not beat the best-HP/200ep RW-1 (0/3 on the verdict set; near-tie on
OPPORTUNITY, close on GECCO). The old "clear loss" was largely a config artifact.**

## What Proposal 1 is
Simplest CEGAR × RW-ML hybrid; overrides only the signals: wrongness
`E_t = σ(k·(robust_z − τ))`, `robust_z = (r − median)/MAD`; confidence `C_t = 1` (basic);
gate `g = E_t·C_t`, scale `s = 1+λg` on the per-window forecasting-loss gradient (ScaleGrad).
Score = `mean|correction|`.

## Experiment settings
| group | values |
|---|---|
| training | `epochs=100`, `warmup=10` (**warm-up = plain RW-1, gate OFF; gate on after**), `correction_init='neg_x'` |
| RW-1 base | `window=50`, `batch=256`, `l1_weight=0.001`, `activation=linear`, `correction_rate=0.1`, score = `mean|correction|` |
| gate | `k=1`, `τ=2`, `λ=1` (fixed) **or** `lam_mode='auto_tr'` (auto-λ column) |
| variant | `basic` (C_t = 1) |
| eval | unit = whole collection (mean over its series); **no fixed seed** (1 run/cell) |
| baseline | RW-1 / DeepAnT = reproduction per-collection means (**best-HP, 200ep**) → Δ is config-confounded on the epoch/HP axis, indicative |

## Results — all collections (AUC-PR; fixed / auto-λ)
`set`: V = verdict, E = shape-spectrum extension. **W** = fixed beats RW-1.

| collection | shape | set | n | DeepAnT* | RW-1* | P1 fixed | auto-λ | Δ (fixed−RW-1) |
|---|:-:|:-:|:-:|:--:|:--:|:--:|:--:|:--:|
| GECCO | block | V | 1 | 0.454 | 0.639 | 0.565 | 0.618 | −0.074 |
| OPPORTUNITY | block | V | 8 | 0.272 | 0.138 | 0.123 | 0.113 | −0.015 |
| CreditCard | point | V | 1 | 0.147 | 0.111 | 0.032 | 0.035 | −0.079 |
| TAO | point | E | 13 | 0.996 | 0.995 | 0.995 | 0.995 | ≈0 (tie) |
| PSM | mixed | E | 1 | 0.407 | 0.137 | 0.125 | 0.126 | −0.012 |
| MSL | block | E | 16 | 0.116 | 0.131 | 0.136 **W** | 0.118 | +0.005 |
| SWaT | block | E | 2 | 0.516 | 0.444 | 0.139 | 0.131 | −0.305 |

Beats RW-1 on **0/3** verdict; on the extension only MSL edges it (+0.005, ~noise) and TAO
ties; SWaT/PSM lose. AUC-ROC (fixed, verdict): OPPORTUNITY 0.703, GECCO 0.918, CreditCard 0.716.

## Correction diagnostics (thesis §8.4, fixed)
| collection | gate→label AUC | corr@anom/norm | Overlap (prec) | Coverage (recall) |
|---|:--:|:--:|:--:|:--:|
| GECCO | 0.896 | 11.60 | 0.194 | 0.776 |
| CreditCard | 0.951 | 1.84 | 0.011 | 0.315 |
| OPPORTUNITY | 0.480 | 1.09 | 0.126 | 0.145 |

## Interpretability
On GECCO the residual gate localizes anomalies (0.90) and correction concentrates there
(11.6×, 78% coverage) → high AUC-PR (0.565). Under the corrected config this concentration
is now *good* (score-contrast up), not the old "erase" risk. Uninformative gate on
opportunity (0.48) yet near-ties RW-1. Point anomalies (creditcard) stay weak.

## Decision
Competitive but does not beat tuned RW-1 → move to Proposal 2.

## Correction examples

Original signal vs. the trained correction (`neg_x` init, gate on after warm-up). Each row of the corpus is one series; the correction concentrates where the model flags anomalies. Rendered from `../figures/` (also logged to each `-example` wandb run).

### Verdict collections

**GECCO (block) — the win**

![P1 GECCO (block) — the win correction example](../figures/P1_basic_gecco_correction_example.png)

**OPPORTUNITY (block)**

![P1 OPPORTUNITY (block) correction example](../figures/P1_basic_opportunity_correction_example.png)

**CreditCard (point)**

![P1 CreditCard (point) correction example](../figures/P1_basic_creditcard_correction_example.png)

### Shape extension

**TAO (point)**

![P1 TAO (point) correction example](../figures/P1_basic_116_TAO_id_1_Environment_tr_500_1st_3.csv_correction_example.png)

**PSM (mixed)**

![P1 PSM (mixed) correction example](../figures/P1_basic_115_PSM_id_1_Facility_tr_50000_1st_129872.csv_correction_example.png)

**MSL (block)**

![P1 MSL (block) correction example](../figures/P1_basic_002_MSL_id_1_Sensor_tr_500_1st_900.csv_correction_example.png)

**SWaT (block)**

![P1 SWaT (block) correction example](../figures/P1_basic_171_SWaT_id_1_Sensor_tr_3749_1st_9522.csv_correction_example.png)

### Characterization set

**SMAP (point)**

![P1 SMAP (point) correction example](../figures/P1_basic_smap_correction_example.png)

**SMD (neutral)**

![P1 SMD (neutral) correction example](../figures/P1_basic_smd_correction_example.png)

**MITDB (periodic)**

![P1 MITDB (periodic) correction example](../figures/P1_basic_mitdb_correction_example.png)

## Reproduce
```bash
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
sbatch experiments/proposals/runs/submit_p1_coll.sh
python experiments/proposals/aggregate_collection.py --proposal 1
```
