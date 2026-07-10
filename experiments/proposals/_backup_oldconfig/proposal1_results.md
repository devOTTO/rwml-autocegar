# Proposal 1 — Residual-Gated RW-CEGAR: Results

**Verdict: P1 loses to the RW-1 baseline on all evaluated collections → fail-fast to Proposal 2.**

## What Proposal 1 is
Simplest CEGAR × RW-ML hybrid, built on the reproduced RW-1 (Algorithm 2). It
overrides only the wrongness / confidence signals:
- **Wrongness** `E_t = sigmoid(k·(robust_z − τ))`, `robust_z = (r − median)/MAD`
- **Confidence** `C_t = 1` (basic) or `sigmoid(k·(r − Q_q(r))/MAD)` (selective) — the
  variant only switches C_t; τ and λ are variant-independent.
- **Gate** `g = E_t·C_t`, **scale** `s = (1+λg)/mean(1+λg)`, applied to the
  per-window forecasting-loss **gradient** via `ScaleGrad` (loss value unchanged).
- Anomaly score = `mean|correction|` (same as RW-1). Note the Δ vs the RW-1 baseline
  is **config-confounded** (P1 default-HP/100ep vs baseline best-HP/200ep) — indicative,
  not a clean isolation of the gate.

## Methodology (collection-level)
Fixed config: `epochs=100`, `warmup=10`, `window=50`, `batch=256`, `l1_weight=0.001`,
`activation=linear`, fixed HP (`lam=1, tau=2, k=1`), auto-tuning off.

**Unit = whole collection.** Each series in a collection is run and averaged to a
per-collection mean AUC-PR/ROC. **Baseline** RW-1 (and DeepAnT reference) are the
**reproduction per-collection means** (`rw/reproduction/summary_rw1_besthp.csv`,
`deepant/reproduction/summary_per_dataset.csv`; best-HP / 200ep / TSB-AD eval — so
the delta is indicative, config not identical to P1's 100ep). `all` verdict set =
opportunity + gecco + creditcard (gecco & creditcard are n=1, i.e. already whole
collections; opportunity = mean over its 8 series).

**Cost** (gecco, 100 epochs, GPU wall-clock; indicative, single no-seed run):
**P1 5:05 (1.0×)** · P2 6:02 (1.19×) · P3 6:44 (1.32×). P1 is the cheapest — one
forward pass per window, the same as RW-1. (Drivers: P2 = M=5 MC-dropout passes;
P3 = direction-stability + gradient amplification + preserve write-back.)

## Collection-level results (primary verdict)

`*` DeepAnT / RW-1 = reproduction per-collection means (reference). Δ = P1 − RW-1.

| collection | n | DeepAnT AUC-PR* | RW-1 AUC-PR* | P1 AUC-PR | **Δ (P1−RW-1)** | P1 AUC-ROC |
|---|:-:|:--:|:--:|:--:|:--:|:--:|
| OPPORTUNITY | 8 | 0.272 | 0.138 | 0.068 | **−0.070** | 0.427 |
| GECCO | 1 | 0.454 | 0.639 | 0.486 | **−0.153** | 0.930 |
| CreditCard | 1 | 0.147 | 0.111 | 0.004 | **−0.108** | 0.617 |

**P1 beats RW-1 on 0/3 collections.** The opportunity collection mean (8 series)
confirms the earlier single-series screen — P1 does not recover at collection level.
DeepAnT is strongest on opportunity/creditcard, RW-1 on gecco; P1 is never best.

### Correction diagnostics (thesis §8.4)
`Overlap` = of high-|correction| timesteps (top 5%), fraction that are anomalies
(precision); `AnomalyCoverage` = of anomalies, fraction that are high-|correction|
(recall); `corr@anom/norm` = anomaly-vs-normal correction ratio; `gate→label AUC`
= how well the gate localizes anomalies.

| collection | gate→label AUC | corr@anom/norm | Overlap | AnomalyCoverage |
|---|:--:|:--:|:--:|:--:|
| GECCO | 0.90 | 5.28 | 0.204 | **0.820** |
| CreditCard | 0.95 | 1.05 | 0.005 | 0.140 |
| OPPORTUNITY | 0.48 | 0.98 | 0.070 | 0.055 |

On GECCO the gate localizes anomalies (AUC 0.90) and correction concentrates there
(5.3×, covering 82% of anomalies) — an observed mechanism (the gate does target
anomalies and keep correction on them). Its highest-signal collection also shows the
largest gap to the best-HP RW-1, but that gap is config-confounded (see Methodology),
so we describe the mechanism without claiming it is the cause of the loss. On
opportunity the gate is uninformative (AUC 0.48).

## Interpretability (why it fails)
Matches the risk the proposal flagged: the gate correctly localizes anomalies
(gate→label AUC ≈ 0.90 on gecco) and pours correction into them (gecco
corr@anom/norm ≈ 5.5×), which "erases" the anomalies from the `mean|correction|`
score, so AUC-PR drops. Per-series `gate/*` and `corr/*` are on the `interp`/
`collection`-tagged wandb runs. A Baldo-Fig-6.1-style original-vs-corrected plot is
in `experiments/proposals/plot_correction_example.py` (gecco figure in wandb).

## Single-series screen — stages 1–4 (exploratory)
The earlier **fail-fast single-series** stage screen (one representative series per
set: opportunity **id_1**, gecco, creditcard). Kept for the record; the collection
means above are the verdict. Manual sweeps were held for the collection re-run;
auto-tuning was checked separately (see **Auto-tuning ablation** below).

> Note: gecco appears with two figures — stage-1 (0.4565 / RW-1 0.6671) uses a
> **100ep co-trained** baseline, the collection table (0.486 / RW-1 0.639) uses the
> **200ep reproduction** baseline; separate runs (dropout variance), hence the small
> difference. RW-1@100 (0.667) ≥ RW-1@200 (0.639), so shorter training doesn't
> inflate the baseline.

## Auto-tuning ablation (verdict set)
Auto-CEGAR's controllers, kept off elsewhere, turned on: P1 `lam_mode=auto_tr`
(auto-λ, adapts λ toward a tail-ratio target, capped at lam_max=1.5).

| collection | auto-λ AUC-PR | RW-1 | beats RW-1? |
|---|:--:|:--:|:--:|
| GECCO | 0.531 | 0.639 | no |
| OPPORTUNITY | 0.072 | 0.138 | no |
| CreditCard | 0.004 | 0.111 | no |

Still **0/3**. The auto-λ figures are within run-to-run variance of the fixed-λ
verdict above (GECCO 0.486, OPPORTUNITY 0.068, CreditCard 0.004; no fixed seed, so
these are separate runs), so auto-λ neither clearly helps nor hurts — it does not
change the verdict.

### Stage 1 — default HP vs RW-1 (single series) — Δ = P1 − RW-1
| dataset | P1 AUC-PR | RW-1 AUC-PR | **Δ AUC-PR** | P1 AUC-ROC |
|---|:--:|:--:|:--:|:--:|
| opportunity (id_1) | 0.0209 | 0.0284 | **−0.0075** | 0.367 |
| gecco | 0.4565 | 0.6671 | **−0.2105** | 0.922 |
| creditcard | 0.0032 | 0.1227 | **−0.1195** | 0.595 |

### Stage 2 — τ sweep (opportunity id_1, basic, λ=1)
| τ | AUC-PR | AUC-ROC |
|:--:|:--:|:--:|
| 1.5 | 0.0207 | 0.365 |
| 2.0 | 0.0209 | 0.367 |
| 2.5 | 0.0208 | 0.336 |
| 3.0 | 0.0200 | 0.340 |

τ has essentially no effect; AUC-ROC stays < 0.5.

### Stage 3 — λ sweep (opportunity id_1, basic, τ=2)
| λ | AUC-PR | AUC-ROC |
|:--:|:--:|:--:|
| 0.5 | 0.0202 | 0.332 |
| 1.0 | 0.0209 | 0.367 |
| 2.0 | 0.0235 | 0.426 |

Stronger gating (λ↑) helps slightly but stays below RW-1 (0.0284) and AUC-ROC 0.5.

### Stage 4 — selective confidence (single series, τ=2, λ=1)
| dataset | basic AUC-PR | selective AUC-PR | basic AUC-ROC | selective AUC-ROC |
|---|:--:|:--:|:--:|:--:|
| opportunity (id_1) | 0.0209 | 0.0221 | 0.367 | 0.355 |
| gecco | 0.4565 | 0.4473 | 0.922 | 0.914 |
| creditcard | 0.0032 | 0.0041 | 0.595 | 0.631 |

Selective ≈ basic — no meaningful improvement.

## Decision
**Fail-fast → Proposal 2.** P1 (RW-CEGAR form) does not beat RW-1 on any collection.

## Reproduce
```bash
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
sbatch experiments/proposals/proposal1/submit_p1_coll.sh       # collection-level, per-series array
python experiments/proposals/aggregate_collection.py --proposal 1   # -> collection table
wandb sync --include-offline ./wandb/offline-run-<date>_*
```
