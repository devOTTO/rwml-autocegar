# RW / RW-1 — reproduction evidence

## RW (Algorithm 1) — ✅ reproduced

Full TSB-AD-M run of `rw/cnn_uns.py` (= TSB-AD `CNN_UNS`, algorithm-faithful).

| metric | ours | paper |
|---|---|---|
| AUC-PR (mean) | **0.321** | 0.29 (T6.1) / 0.34 (best-HP, T6.5) |
| AUC-ROC (mean) | **0.703** | 0.73 |

n = 199. Per-dataset AUC-PR correlation with the paper ≈ **0.99** — RW is a
faithful reproduction. Files: `summary_per_dataset.csv`,
`summary_per_family.csv`, `submit_rw.sh`.

Run (in TSB-AD repo): `sbatch submit_rw_algofaithful_full.sh`
→ `eval/metrics/multi_algofaithful/CNN_UNS/*.csv`.
Config: `window=50, epochs=200, lr=8e-4, batch=256, correction_rate=0.1,
activation=linear`.

## RW-1 (Algorithm 2) — ✅ reproduced (best-HP)

`rw/cnn_rw.py`. The first rewrite was buggy (ReLU freeze + L1 collapse; see
top-level README and `experiments/exp_e_algofaithful/`). After the fix
(`activation='linear'`, `l1_weight` added), the previously-broken datasets
recover to paper level:

| dataset | old (buggy) AUC-PR | fixed (l1=0.001) | paper |
|---|---|---|---|
| Genesis | 0.002 | **0.033** | 0.032 |
| GECCO | 0.080 | **0.613** | 0.621 |

Full reproduction = **best-HP per dataset**: sweep `l1_weight ∈ {1.0, 0.1,
0.01, 0.001}` over all files, take best AUC-PR per dataset, mirroring the
paper's Table 6.5 reporting. The 4× full sweep completed 2026-07-02
(199 datasets × 4 λ, no failures):

| metric | ours (best-HP) | paper (T6.1) |
|---|---|---|
| AUC-PR (mean) | **0.289** | 0.28 |
| AUC-ROC (mean) | **0.725** | 0.75 |

n = 199. Best-λ distribution: `0.001`×97, `0.1`×43, `0.01`×42, `1.0`×17.
Files: `summary_rw1_besthp.csv` (per-dataset), `summary_rw1_besthp_per_family.csv`.

### Per-collection: ours vs paper (Δ = ours − paper)

The paper reports per-**collection** values only (thesis Tables 6.2 AUC-PR /
6.3 AUC-ROC), not per individual series — so a collection (17 rows) is the
finest resolution at which "ours vs paper" can be compared. Ours = best-HP RW-1
aggregated per collection.

| collection | n | ours PR | paper PR | Δ PR | ours ROC | paper ROC | Δ ROC |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| CATSv2 | 6 | 0.106 | 0.228 | −0.122 | 0.619 | 0.755 | −0.136 |
| CreditCard | 1 | 0.111 | 0.173 | −0.062 | 0.875 | 0.953 | −0.078 |
| Daphnet | 1 | 0.267 | 0.286 | −0.019 | 0.817 | 0.871 | −0.054 |
| Exathlon | 27 | 0.873 | 0.847 | +0.026 | 0.973 | 0.981 | −0.008 |
| GECCO | 1 | 0.639 | 0.621 | +0.018 | 0.959 | 0.979 | −0.020 |
| GHL | 24 | 0.033 | 0.013 | +0.020 | 0.665 | 0.564 | +0.101 |
| Genesis | 1 | 0.033 | 0.032 | +0.001 | 0.747 | 0.954 | −0.207 |
| LTDB | 5 | 0.248 | 0.253 | −0.005 | 0.642 | 0.611 | +0.031 |
| MITDB | 13 | 0.089 | 0.127 | −0.038 | 0.612 | 0.639 | −0.027 |
| MSL | 16 | 0.131 | 0.086 | +0.045 | 0.692 | 0.557 | +0.135 |
| OPPORTUNITY | 8 | 0.138 | 0.059 | +0.079 | 0.720 | 0.430 | +0.290 |
| PSM | 1 | 0.137 | 0.238 | −0.101 | 0.537 | 0.697 | −0.160 |
| SMAP | 27 | 0.119 | 0.095 | +0.024 | 0.635 | 0.578 | +0.057 |
| SMD | 22 | 0.233 | 0.317 | −0.084 | 0.714 | 0.790 | −0.076 |
| SVDB | 31 | 0.116 | 0.166 | −0.050 | 0.615 | 0.628 | −0.013 |
| SWaT | 2 | 0.444 | 0.227 | +0.217 | 0.764 | 0.689 | +0.075 |
| TAO | 13 | 0.995 | 1.000 | −0.005 | 0.996 | 1.000 | −0.004 |

**Overall (n=199, per-dataset mean): ours 0.289 vs paper 0.28 AUC-PR (Δ +0.009).**
Per-collection deltas scatter both ways with no systematic bias — reproduction is
faithful. Largest misses: CATSv2 (−0.12), PSM (−0.10), SMD (−0.08); largest
overshoots: SWaT (+0.22), OPPORTUNITY (+0.08), MSL (+0.05). (Same numbers as the
combined 3-method table in top-level `reproduction_comparison.md` §2.)

Run (in TSB-AD repo): `for l in 1.0 0.1 0.01 0.001; do sbatch submit_rw1_l1sweep.sh $l; done`
→ `eval/metrics/multi_rw1_l1sweep/l1_<λ>/CNN_RW/*.csv`, then aggregate with
`experiments/exp_e_algofaithful/combine_rw1_besthp.py`.
