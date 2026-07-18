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

## Full P1–P5 cr × l1 re-ranking (all gates, 504 runs each)
The same `cr{0.001,0.01,0.1} × l1{0.001,0.01,0.1,1.0}` sweep at 200ep was run for **every**
proposal (P4 in the `gradnorm_wb` variant, i.e. with the docx write-back on). Per-series best
over the grid, collection mean:

| collection | RW-1 best | P1 | P2 | P3 | P4 (wb) | **P5** |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| GECCO | 0.639 | 0.559 | 0.532 | 0.524 | 0.574 | 0.595 |
| OPPORTUNITY | 0.138 | 0.608 | 0.608 | 0.602 | 0.530 | 0.608 |
| CreditCard | 0.111 | 0.173 | 0.173 | 0.173 | 0.173 | 0.173 |
| TAO | 0.995 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| PSM | 0.137 | 0.168 | 0.155 | 0.157 | 0.172 | 0.166 |
| MSL | 0.131 | 0.178 | 0.176 | 0.162 | 0.187 | **0.210** |
| SWaT | 0.444 | 0.536 | 0.537 | 0.513 | 0.498 | 0.535 |
| **MEAN(7)** | **0.371** | **0.460** | 0.454 | 0.447 | 0.448 | **0.470** |
| wins vs RW-1 | — | 6/7 | 6/7 | 6/7 | 6/7 | 6/7 |

Figure: `figures/crtune_rerank.png`. wandb: run `crtune-rerank-P1toP5` (group `crtune_rerank`).

**What this shows:**
1. **The five gates cluster tightly (0.447–0.470).** Ordering P5 > P1 > P2 ≈ P4 > P3, but the
   spread is ~0.02 — **the choice of gate is essentially second-order**; consistent with the
   earlier finding that λ (incl. auto) has ~no effect. **cr and l1 are what move the number.**
2. **P4's write-back does not lift P4** — `gradnorm_wb` lands mid-pack (0.448), below P1/P5.
   The docx correctable-point write-back is implemented and runs cleanly, but adds no aggregate
   AUC-PR over plain amplification here.
3. **P5 stays the best single gate** (0.470), edging P1 (0.460), mainly on MSL (its one real
   gate win, +0.079 vs RW-1) and a small OPPORTUNITY/SWaT margin.
4. **Same like-for-like caveat as above:** RW-1's best-HP swept l1 only, so the uniform "6/7"
   is inflated by the extra cr axis given to P1–P5, **not** by the gates.

## Deployable single-config vs thesis RW 1 (the honest comparison)
The oracle table above takes a per-series best over the grid — not deployable, and it compares
against our own cr=0.1-fixed reproduction. Two fixes make it fair, with **no new runs**:
(a) pick **one** `(cr, l1)` per proposal (the config maximizing mean-over-collections AUC-PR),
and (b) use **Baldo's own properly-tuned RW 1** (thesis Table 6.2) as the baseline instead of
our cr-fixed reproduction. Result:

| collection | thesis RW 1 | P1 | P2 | P3 | P4 (wb) | P5 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| GECCO | **0.621** | 0.420 | 0.418 | 0.413 | 0.402 | 0.422 |
| OPPORTUNITY | 0.059 | 0.608 | 0.608 | 0.602 | 0.520 | 0.608 |
| CreditCard | 0.173 | 0.173 | 0.173 | 0.173 | 0.173 | 0.173 |
| TAO | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| PSM | **0.238** | 0.152 | 0.151 | 0.151 | 0.155 | 0.152 |
| MSL | 0.086 | 0.118 | 0.118 | 0.117 | 0.120 | 0.119 |
| SWaT | 0.227 | 0.500 | 0.500 | 0.501 | 0.497 | 0.500 |
| **MEAN(7)** | 0.343 | 0.424 | 0.424 | 0.422 | 0.409 | **0.425** |
| **wins vs thesis** | — | 3/7 | 3/7 | 3/7 | 3/7 | 3/7 |

Chosen single config: P1/P2/P5 = `cr0.001/l10.001`, P3/P4 = `cr0.001/l10.1`.
Figure `figures/crtune_fixed_vs_thesis.png`.

**What this actually shows — the oracle 6/7 collapses to 3/7:**
- With one deployable config, each proposal beats thesis RW 1 on only **3/7** (OPPORTUNITY,
  MSL, SWaT), loses on GECCO + PSM, ties CreditCard + TAO.
- **The two big "wins" (OPPORTUNITY, SWaT) are protocol-confounded, not gate gains.** For the
  *same* RW 1 method, thesis reports SWaT 0.227 while our gate-off reproduction already gets
  0.444, and OPPORTUNITY 0.059 vs our 0.138 — the gaps are preprocessing/protocol, present
  before any gate. So the gate's own contribution is **≈ 0**, exactly as the tight P1–P5
  clustering (0.409–0.425) implies.
- **Honest headline:** correction_rate tuning is what moves the number; the CEGAR gate (any of
  P1–P5, incl. P4's write-back) neither helps nor hurts on aggregate once HP and protocol are
  controlled. P5 remains nominal-best (0.425) but within noise of P1/P2.

## P5 full-200 benchmark (per-file best over cr × l1)
P5 (`h5`, auto-λ) over the full 200-series RW benchmark, per-file best over `cr × l1`:
**mean AUC-PR = 0.364** (199 series) vs **thesis RW-1 0.28** / reproduction RW-1 best-HP 0.289.
This is an oracle per-file best (upper bound), and RW-1 was **not** re-run on full-200 under the
same cr×l1 grid, so it is P5-vs-thesis, not a matched head-to-head. Source
`results_p5_p5_full200_besthp.csv`; aggregated by `log_crtune_rerank.py`.

## Next steps
- ~~RW-1 `cr × l1` sweep~~ — **not needed**: Baldo's thesis Table 6.2 already reports the
  properly-tuned RW 1 per collection, so it serves as the fair (deployable) baseline without
  re-running RW-1. The deployable single-config comparison above uses it.
- ~~report a single fixed config, not oracle~~ — **done** (deployable table above).
- **Remaining, for the meeting with Luís:** discuss gate designs beyond P1–P5 given that the
  current CEGAR gate contributes ≈ 0 on aggregate. The λ-controller ablations are dropped
  (λ 1.0 / 1.5 / auto near-identical). Open question: is there a gate formulation that helps
  where correction-magnitude scoring provably fails (e.g. CreditCard point anomalies)?
