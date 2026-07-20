# Proposal 5 — Temporal-Persistence Confident-Error CEGAR: Results

**Result: P5 is the only proposal that beats the best-HP/200ep RW-1 on a screening
collection — GECCO, robustly with auto-λ (6/6 runs, min 0.661 > 0.639). Fixed-λ ties.
The win does NOT generalize across the shape spectrum.**

## What Proposal 5 is (docx-faithful)
Gate only errors that PERSIST over neighbouring time windows. `m_t=1(e_t>τ_e)`
(τ_e = residual q_e quantile), `p_t=mean(m_{t-h..t+h})` (temporal persistence, epoch-end),
`g=σ(k(e_t−τ_e)/mad)·σ(k_p(p_t−τ_p))`. Previous-epoch gate drives amplification (ScaleGrad).
Score = `mean|correction|`.

## Experiment settings
| group | values |
|---|---|
| training | `epochs=100`, `warmup=10` (**plain RW-1, gate OFF; gate on after**), `correction_init='neg_x'` |
| RW-1 base | `window=50`, `batch=256`, `l1_weight=0.001`, `activation=linear`, `correction_rate=0.1` |
| gate | `λ=1` (fixed) **or** `lam_mode='auto_tr'`; `q_e=0.8`, `persist_h=5`, `k_p=5`, `tau_p=0.5`, `persist_alpha=0.9` |
| variant | `h5` (persistence window ±5) |
| eval | whole collection; **no fixed seed** — GECCO run ×6 (robustness), others 1/cell |
| baseline | reproduction best-HP/200ep → Δ config-confounded (indicative) |

## Results — all collections (AUC-PR; fixed / auto-λ)
Top three rows = the 3 screening collections picked at the start (GECCO / OPPORTUNITY / CreditCard); bottom four = the later shape-spectrum extension. **W** = fixed beats RW-1.

| collection | shape | n | DeepAnT* | RW-1* | P5 fixed | auto-λ | Δ (fixed−RW-1) |
|---|:-:|:-:|:--:|:--:|:--:|:--:|:--:|
| GECCO | block | 1 | 0.454 | 0.639 | 0.643 **W** | 0.677 **W** | +0.004 |
| OPPORTUNITY | block | 8 | 0.272 | 0.138 | 0.114 | 0.111 | −0.024 |
| CreditCard | point | 1 | 0.147 | 0.111 | 0.029 | 0.032 | −0.082 |
| TAO | point | 13 | 0.996 | 0.995 | 0.996 | 0.995 | ≈0 (tie) |
| PSM | mixed | 1 | 0.407 | 0.137 | 0.124 | 0.130 | −0.013 |
| MSL | block | 16 | 0.116 | 0.131 | 0.137 **W** | 0.130 | +0.006 |
| SWaT | block | 2 | 0.516 | 0.444 | 0.141 | 0.154 | −0.303 |

**Beats RW-1 on GECCO (the only screening-set win of any proposal).** On the extension only MSL
edges it (+0.006, ~noise); SWaT (block) is a heavy loss → NOT a general block method.
The SWaT gap is likely the config axis rather than gating: it is uniform across all
five gates, and SWaT's best-HP baseline uses `l1_weight` 0.1/1.0 vs our fixed 0.001
(see SUMMARY, shape-spectrum section, for the evidence and the cheap confirm run).
AUC-ROC (fixed): OPP 0.683, GECCO 0.953, CC 0.629.

## GECCO robustness (6 runs each, no fixed seed)
| | runs | mean | > RW-1 0.639 |
|---|---|:--:|:--:|
| fixed-λ | 0.633, 0.635, 0.639, 0.648, 0.648, 0.652 | 0.643 | 4/6 (min 0.633 — **tie**) |
| **auto-λ** | 0.661, 0.672, 0.672, 0.682, 0.683, 0.690 | **0.677** | **6/6 (min 0.661 — robust win)** |

Fixed straddles RW-1 (tie); **auto-λ beats it on every run** — a real, reproducible win.
Note (observation, not a controller judgment): in all 6 auto-λ runs λ reached the
`lam_max = 1.5` cap by the final epoch (τ stayed fixed at 2.0), so in these runs the
auto-vs-fixed comparison is effectively λ ≈ 1.5 vs λ = 1. Whether the lift reflects
adaptivity or simply a higher λ is a question for the Auto-CEGAR controller side; see
SUMMARY "What auto-λ actually did" for the ablation plan that would disentangle it.

## Correction diagnostics (thesis §8.4, fixed)

How to read (all computed per timestep in the FINAL training epoch, against the
ground-truth labels; labels are used for analysis only, never during training):

- **gate->label AUC**: ROC-AUC when the per-timestep gate activation is used as if it
  were an anomaly score. 0.5 = the gate fires randomly w.r.t. the true anomalies,
  1.0 = it fires exactly at them. Measures how well the gate LOCALIZES anomalies.
  (Gate activation of a timestep = mean gate value of the training windows whose
  prediction target is that timestep.)
- **corr@anom/norm**: mean |correction| on anomaly timesteps / mean |correction| on
  normal timesteps. Since the anomaly score IS mean |correction|, this is the score
  contrast: e.g. a value of 10 means anomalous points end up with 10x more correction
  than normal points (higher = better separation = higher AUC-PR, all else equal).
- **Overlap (prec)**: thesis Sec. 8.4 definition. A point is "high-correction" when its
  |correction| exceeds the series' own 95th percentile (tau_C). Overlap = fraction of
  high-correction points that are true anomalies (precision of the correction).
- **Coverage (recall)**: fraction of true anomaly points that are high-correction
  (recall of the correction; thesis Sec. 8.4 calls it AnomalyCoverage).

| collection | gate→label AUC | corr@anom/norm | Overlap | Coverage |
|---|:--:|:--:|:--:|:--:|
| GECCO | 0.945 | 12.55 | 0.219 | 0.877 |
| CreditCard | 0.899 | 1.75 | 0.008 | 0.222 |
| OPPORTUNITY | 0.684 | 1.08 | 0.118 | 0.134 |

P5 has the best gate localization (GECCO 0.945) and correction concentration/coverage of
the five — consistent with it being the winner on GECCO.

## Decision
**End of the P1–P5 screening arc.** Under the corrected schedule (RW-1 from epoch 1, only
the gate waits through warm-up) P5+auto-λ is a reproducible win over tuned RW-1 on GECCO.
It did not generalize under the fixed config (MSL within noise, point/mixed ≈/below RW-1,
SWaT gap likely the config axis); deltas config-confounded (epoch/HP). Next: per-dataset
P5 tuning (λ in the grid) vs per-file best-HP RW-1, then the full benchmark if competitive.


## Performance (AUC-PR by collection)

![P5 AUC-PR comparison by collection](../figures/P5_comparison_bars.png)

P5 is the only proposal with a bar above RW-1 on a real screening collection: GECCO auto-λ (0.677 > 0.639), flagged above. Everywhere else it ties (TAO) or sits below (SWaT block, CreditCard point) — the win is GECCO-specific.

## Correction examples

**How to read these.** *Middle panel*: `original x` (blue) vs `corrected x = x + correction` (orange) — where the two diverge, the trained RW correction is large. *Bottom panel*: the CEGAR gate (green) and the per-step `|correction|` score (purple); the red band is the labelled anomaly. A detector scores well when both the gate and `|correction|` spike **inside** the red band and stay flat outside — that contrast is what the anomaly score (`mean|correction|`) turns into AUC-PR. The top strip shows where the zoom window sits in the whole series.

**Analysis.** P5 has the tightest localization of the five (GECCO gate→label 0.945, correction ≈12.6× normal, 88% coverage): the gate and `|correction|` light up almost exactly on the red band and stay flat elsewhere — the visual counterpart of its GECCO win. On the long SWaT block the correction spreads and misses, matching its collapse there.

### Screening collections

**GECCO (block) — the win**

![P5 GECCO (block) — the win correction example](../figures/P5_h5_gecco_correction_example.png)

**OPPORTUNITY (block)**

![P5 OPPORTUNITY (block) correction example](../figures/P5_h5_opportunity_correction_example.png)

**CreditCard (point)**

![P5 CreditCard (point) correction example](../figures/P5_h5_creditcard_correction_example.png)

### Shape extension

**TAO (point)**

![P5 TAO (point) correction example](../figures/P5_h5_116_TAO_id_1_Environment_tr_500_1st_3.csv_correction_example.png)

**PSM (mixed)**

![P5 PSM (mixed) correction example](../figures/P5_h5_115_PSM_id_1_Facility_tr_50000_1st_129872.csv_correction_example.png)

**MSL (block)**

![P5 MSL (block) correction example](../figures/P5_h5_002_MSL_id_1_Sensor_tr_500_1st_900.csv_correction_example.png)

**SWaT (block)**

![P5 SWaT (block) correction example](../figures/P5_h5_171_SWaT_id_1_Sensor_tr_3749_1st_9522.csv_correction_example.png)

## Reproduce
```bash
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
sbatch experiments/proposals/runs/submit_p5_coll.sh
python experiments/proposals/aggregate_collection.py --proposal 5
```

## Update — cr × l1 re-ranking (post-screening)
The table above is the **initial fail-fast screening** (fixed `correction_rate=0.1`, 100ep,
gate on after warm-up, indicative baseline). A later sweep re-ran P5 over
`cr∈{0.001,0.01,0.1} × l1∈{0.001,0.01,0.1,1.0}` at 200ep. `correction_rate` turned out to
be a dominant, collection-dependent knob that the fixed 0.1 mis-set for most collections.

**Oracle** (per-series best over the grid, collection mean — over-optimistic upper bound, vs our
cr=0.1-fixed reproduction RW-1):

| method | GECCO | OPP | CC | TAO | PSM | MSL | SWaT | MEAN(7) | wins |
|---|--|--|--|--|--|--|--|--|--|
| P5-persistence | 0.595 | 0.608 | 0.173 | 1.000 | 0.166 | 0.210 | 0.535 | **0.469** | 6/7 |

**Deployable** (one fixed config `cr0.001/l10.001`) vs **Baldo thesis RW 1** (Table 6.2, properly tuned):

| method | GECCO | OPP | CC | TAO | PSM | MSL | SWaT | MEAN(7) | wins |
|---|--|--|--|--|--|--|--|--|--|
| thesis RW 1 | 0.621 | 0.059 | 0.173 | 1.000 | 0.238 | 0.086 | 0.227 | 0.343 | — |
| P5-persistence | 0.422 | 0.607 | 0.173 | 1.000 | 0.152 | 0.119 | 0.500 | **0.425** | 3/7 |

**Takeaway:** the oracle 6/7 collapses to **3/7** under one deployable config,
and the remaining wins (OPPORTUNITY/SWaT) are protocol-confounded (same RW 1: thesis SWaT 0.227
vs our gate-off reproduction 0.444). So this gate's own contribution is **≈ 0** — all five
proposals cluster at 0.41–0.43. See `SUMMARY.md` and `proposal5_tune_results.md` for the full
P1–P5 comparison; figures `figures/crtune_rerank.png` + `figures/crtune_fixed_vs_thesis.png`.
