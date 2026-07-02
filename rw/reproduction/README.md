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

Run (in TSB-AD repo): `for l in 1.0 0.1 0.01 0.001; do sbatch submit_rw1_l1sweep.sh $l; done`
→ `eval/metrics/multi_rw1_l1sweep/l1_<λ>/CNN_RW/*.csv`, then aggregate with
`experiments/exp_e_algofaithful/combine_rw1_besthp.py`.
