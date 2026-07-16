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

## correction_rate sweep — results (`proposal5_crtune`, 504 runs)
P5 (h5, auto-λ) swept `correction_rate ∈ {0.001, 0.01, 0.1} × l1_weight ∈ {1, 0.1, 0.01, 0.001}`,
200ep. **`correction_rate` turns out to be a large, collection-dependent lever — bigger than
l1 for several collections.** Mean AUC-PR at `l1=0.001`:

| collection | cr=0.001 | cr=0.01 | cr=0.1 (our old fixed) |
|---|:--:|:--:|:--:|
| GECCO | 0.422 | 0.575 | **0.595** |
| OPPORTUNITY | **0.607** | 0.124 | 0.078 |
| CreditCard | **0.173** | 0.165 | 0.008 |
| TAO | **1.000** | 0.998 | 0.995 |
| PSM | 0.152 | **0.166** | 0.130 |
| MSL | 0.119 | 0.109 | 0.119 |
| SWaT | **0.500** | 0.366 | 0.153 |

- **Most collections want LOW cr (0.001)**: OPPORTUNITY 0.078→0.607, SWaT 0.153→0.500,
  CreditCard 0.008→0.173. **GECCO is the exception — it wants HIGH cr (0.1).**
- **Our fixed `correction_rate=0.1` (every earlier run) was near-optimal for GECCO but poor
  for the rest** — it silently shaped the whole earlier "P5 only wins GECCO" picture.

P5 crtune-best (per-series best over cr × l1) vs RW-1 best-HP:

| collection | RW-1 best | P5 crtune-best | Δ | best (cr, l1) |
|---|:--:|:--:|:--:|:--:|
| GECCO | 0.639 | 0.595 | −0.044 | cr0.1 / l10.001 |
| OPPORTUNITY | 0.138 | 0.608 | **+0.470** | cr0.001 / l1≥0.1 |
| CreditCard | 0.111 | 0.173 | +0.062 | cr0.001 / l11.0 |
| TAO | 0.995 | 1.000 | +0.005 | cr0.001 |
| PSM | 0.137 | 0.166 | +0.029 | cr0.01 / l10.001 |
| MSL | 0.131 | 0.210 | +0.080 | cr0.1 / l10.001 |
| SWaT | 0.444 | 0.535 | +0.091 | cr0.1 / l11.0 |

P5 crtune-best > RW-1 best-HP on **6/7** (all but GECCO).

**Critical caveat (do not over-claim):** RW-1's best-HP was tuned over `l1_weight` ONLY
(cr fixed at 0.1); P5 here got the extra `correction_rate` axis. So the 6/7 is **not**
like-for-like — P5 simply has one more tuning knob. The fair comparison is to give RW-1
the same `cr × l1` sweep (next step). The solid, defensible finding is: **`correction_rate`
is a dominant, collection-dependent HP, and our earlier fixed 0.1 was mis-set for most
collections — which retro-explains much of the earlier per-collection win/loss pattern.**

## Next steps
- **RW-1 `cr × l1` sweep** (baseline-only) so the 6/7 becomes a fair like-for-like — this is
  now the priority (P5 has an unfair extra axis until RW-1 gets it too).
- For a headline claim, report the single fixed config that maximizes mean Δ (not oracle).
