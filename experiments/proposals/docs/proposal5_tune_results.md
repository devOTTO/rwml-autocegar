# Proposal 5 — Per-dataset HP tuning (like-for-like vs RW-1 best-HP)

**Headline: with per-series HP tuning, `l1_weight` (not λ) is the dominant axis, and
different collections need opposite l1 optima. P5 has exactly one clear fixed-config win
(MSL); the aggregate "4/7" is an oracle-HP artifact (per-series grid-best), not a single
recipe that wins on 4/7.**

## Setup
P5 (variant `h5`) swept per series over `l1_weight ∈ {1, 0.1, 0.01, 0.001} × λ ∈ {1, 1.5, auto}`
at 200 epochs, on the same 42 series as the RW-1 best-HP reproduction baseline.
- **P5 best-HP** = per-series max AUC-PR over the 12-cell grid, then collection mean
  (mirrors how RW-1 `best_pr` is a per-file best over its own l1 sweep).
- `matched RW-1` = plain gate-off RW-1 at our fixed `l1_weight=0.001` (100/200ep) — the
  config-confound control.
- `correction_rate` fixed at 0.1 — a *separate* knob (the RMSprop step size for the
  correction, thesis §6.3.2) that is swept in a follow-up (`proposal5_crtune`).
- Source: `results_p5_proposal5_tune.csv` (504 runs) + `results_rw1matched_...csv` (84),
  aggregated by `agg_tune.py` → `docs/tune_aggregate.txt`.

## Result — P5 best-HP vs RW-1 best-HP
| collection | DeepAnT | RW-1 best | P5 best | Δ (P5−RW1) | matched RW-1 (l1=.001, 200/100) |
|---|:--:|:--:|:--:|:--:|:--:|
| GECCO | 0.454 | 0.639 | 0.608 | −0.030 | 0.616 / 0.667 |
| OPPORTUNITY | 0.272 | 0.138 | 0.142 | +0.003 | 0.089 / 0.099 |
| CreditCard | 0.147 | 0.111 | 0.014 | −0.098 | 0.114 / 0.119 |
| TAO | 0.996 | 0.995 | 0.996 | +0.000 | 0.995 |
| PSM | 0.407 | 0.137 | 0.133 | −0.004 | 0.137 / 0.134 |
| **MSL** | 0.116 | 0.131 | **0.226** | **+0.095** | 0.101 / 0.120 |
| SWaT | 0.516 | 0.444 | 0.451 | +0.006 | 0.142 / 0.158 |

P5 best-HP > RW-1 best-HP on **4/7**, but OPPORTUNITY / TAO / SWaT are ties within no-seed
noise — **the one real win is MSL (+0.095)**.

## Figures (local analysis, `figures/`)
- **`P5_tune_l1_curves.png`** — AUC-PR vs `l1_weight` per collection, one curve per λ-mode
  (1.0 / 1.5 / auto) + the RW-1 reference line (Baldo Fig 6.6–6.9 style).
- **`P5_tune_delta_heatmap.png`** — Δ(P5 − RW-1) over the 12-cell `l1 × λ` grid.
- **`P5_tune_lambda_slices.png`** — best-over-l1 per λ-mode vs RW-1.

## Key findings
1. **`l1_weight` is dominant; λ (and auto-λ) is not.** In the curves/heatmap the three λ
   cells (1.0 / 1.5 / auto) sit almost on top of each other for every collection → **auto-λ
   adds essentially nothing.** The earlier "auto-λ win on GECCO" was a config effect (higher
   effective λ / epoch), not adaptivity.
2. **Opposite l1 optima across collections** (the central result):
   - **TAO**: only `l1=0.001` ties RW-1 (Δ ≈ +0.00); heavier l1 crashes (down to ≈ −0.88).
   - **SWaT**: the reverse — needs `l1 ≥ 0.1` (Δ ≈ 0); `l1 ≤ 0.01` gives ≈ −0.3. Confirms
     "SWaT prefers heavy l1".
   - **MSL**: `l1=0.01` + λ=1.5/auto is the **one clear win** over RW-1 (Δ +0.04…+0.095).
   - **GECCO / CreditCard / PSM**: no cell beats RW-1.
3. **The "4/7" is oracle HP.** It picks each series' best cell post hoc; at any *single*
   fixed config the winning collection differs, so no one recipe wins on 4/7. A fair claim
   needs a fixed config (or a principled per-series HP selector).

## SWaT collapse: config, not gating (resolved)
The earlier "gate collapses on the SWaT block" suspicion is **disproven**: the matched
gate-off RW-1 at `l1=0.001` also collapses to ~0.14, and gate-on P5 with `l1` tuned
(id_1→0.1, id_2→1.0, exactly RW-1's best l1) recovers to **0.451 ≈ RW-1's 0.444**. The
collapse was the fixed `l1_weight`, not the gate. **Where the gate genuinely hurts is
CreditCard** (gate-on 0.014 vs gate-off 0.114 at the same config — isolated point
anomalies).

## Caveats
- Single run per (series, config), no fixed seed; P5's per-series max is over 12 cells vs
  RW-1's 4 (l1-only) → P5-best is mildly noise-favored.
- `correction_rate` (thesis §6.3.2 axis) not yet swept — the follow-up is running.

## Next steps
- **`correction_rate {0.001,0.01,0.1} × l1_weight` sweep** (thesis-faithful; submitted as
  `proposal5_crtune`, job 42253497). Results appended here when done.
- For a headline claim, report the single fixed config that maximizes mean Δ (not oracle).
