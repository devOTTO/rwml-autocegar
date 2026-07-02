# rwml-autocegar

Reproduction of the **DeepAnT / RW / RW-1** anomaly detectors from Afonso Baldo's
2025 FEUP MSc thesis *"Read & Write Machine Learning for Outlier Detection in
Multivariate Time Series"* on the **TSB-AD-M** benchmark (180 files / 199 result
rows), plus **AutoCEGAR** — a residual-CEGAR gate built on top of the reproduced
RW-1.

The reproduction itself was run inside the `TSB-AD` repo
(`/ocean/projects/cis260190p/yhwang2/TSB-AD`) via `benchmark_exp.Run_Detector_M`.
This repo holds the **reproduction-successful model code** (copied here, made
self-contained) and the novel AutoCEGAR extension.

## Layout

```
rwml-autocegar/
  tsb_common/           vendored TSB-AD utils so the models run standalone
    utility.py            get_activation_by_name
    torch_utility.py      get_gpu, min_memory_id, EarlyStoppingTorch
    dataset.py            ForecastDataset
  deepant/
    cnn.py                DeepAnT baseline  (= TSB-AD CNN.py)
  rw/
    cnn_uns.py            RW    — paper Algorithm 1 (= TSB-AD CNN_uns.py)
    cnn_rw.py             RW-1  — paper Algorithm 2 (= TSB-AD CNN_RW.py, fixed)
  autocegar/
    scale_grad.py         ScaleGrad autograd function
    gate.py               gate(confidence, wrongness)
    controllers.py        tail-ratio lambda / valley-detection tau controllers
    residual_signals.py   ResidualStats + E_t/C_t (PLACEHOLDER formulas)
    rw_cegar.py           CNN_RW_CEGAR = reproduced RW-1 + CEGAR gate
  experiments/
    exp_e_algofaithful/   RW-1 diagnostics, L1-weight sweep, analysis scripts
  backup/                 pre-restructure standalone track + tsb_ad_models history
  02_2025_FEUP_MSc_Afonso_Baldo.pdf
```

## The four methods

| Method | Code | Paper section | Idea |
|---|---|---|---|
| DeepAnT | `deepant/cnn.py` | baseline | CNN forecaster; large prediction error = anomaly |
| RW | `rw/cnn_uns.py` | Algorithm 1 | correct the **input** by its gradient; correction size = score |
| RW-1 | `rw/cnn_rw.py` | Algorithm 2 | separate `correction` tensor (init `-X`) + L1 sparsity |
| RW-2 | — (dropped) | Algorithm 3 | RW-1 + Savitzky-Golay smoothing — **out of scope** |

Common config (RW / RW-1): `window=50, epochs=200, lr=8e-4, batch=256,
correction_rate=0.1`; model = Adam, correction = RMSprop, epoch-wise update.

## Reproduction results (TSB-AD-M) vs paper

| Method | ours AUC-PR | ours AUC-ROC | paper AUC-PR | status |
|---|---|---|---|---|
| DeepAnT (TSB-AD CNN) | 0.350 | 0.770 | 0.33 | ✅ reproduced |
| RW | 0.321 | 0.703 | 0.29 (T6.1) / 0.34 (best-HP) | ✅ reproduced (per-dataset corr 0.99) |
| RW-1 | in progress | — | 0.28 / 0.35 | 🔧 fixed, best-HP sweep running |

Paper headline (Table 6.1, avg over its 17 datasets): RW 0.29 / RW-1 0.28.
The 0.34 / 0.35 in Table 6.5 are **best-HP-per-dataset**.

## RW-1: the two bugs and the fix

The first RW-1 rewrite diverged from Algorithm 2 in two places, both fixed in
`rw/cnn_rw.py`:

1. **ReLU gradient gating** — Algorithm 2 has *no* activation on the correction
   gradient. A `relu(grad)` had been added (from Table 6.5's "ReLU" row); with
   `correction = -X` the L1 gradient is `sign(-X) ≈ -1` everywhere, so `relu`
   zeroed the gradient and **froze** the correction (Genesis AUC-ROC 0.137 vs
   paper 0.954). → default `activation='linear'`.

2. **L1 penalty ~100-1000x too strong** — `torch.norm(correction, p=1)` sums over
   all elements, crushing `correction` to 0 over 200 epochs (all points
   reintroduced -> anomaly signal collapses/inverts). RW-1 actually hits
   paper-level AUC in the first few epochs then decays. → added `l1_weight`
   (default `0.001`); recovered Genesis final AUC-PR to 0.033 (paper 0.032),
   GECCO to 0.61 (0.62).

Because the best `l1_weight` is dataset-dependent, RW-1 is reproduced the way the
paper reports it — **best-HP per dataset**: sweep `l1_weight in {1.0, 0.1, 0.01,
0.001}` over all 180 files and take the best AUC-PR per dataset. See
`experiments/exp_e_algofaithful/` (sweep submit + `combine_rw1_besthp.py`).

## AutoCEGAR (`autocegar/rw_cegar.py`)

`CNN_RW_CEGAR` subclasses the reproduced RW-1 and injects a CEGAR gate into the
per-window predictive loss:

```
residual  = y - y_hat
E_t       = clamp(residual / residual_q95, 1.0)      # wrongness
C_t       = sigmoid(k * (tau - residual_ema))        # confidence (historical accuracy)
gate      = clamp(E_t * C_t, 0, 1)
loss      = mean( per_window_RMSE * (1 + lam*gate) ) + l1_weight*||correction||_1
```

Windows the model gets wrong *while historically accurate* (likely true
anomalies) are up-weighted, steering `correction` toward them. Init=-X, L1,
linear-activated RMSprop step and the `|correction|` score are inherited
unchanged from the reproduced RW-1.

> **PLACEHOLDER:** the `E_t` / `C_t` formulas and `tau` / `k` constants are
> scaffolding pending Luis's confidence/wrongness notebook. The wiring
> (residual -> signals -> gate -> gated loss) is final.

## Usage

```python
import sys; sys.path.insert(0, ".")   # run from repo root
from rw.cnn_rw import CNN_RW              # RW-1
from autocegar import CNN_RW_CEGAR        # RW-1 + CEGAR

clf = CNN_RW(window_size=50, feats=n_feats, l1_weight=0.001)   # activation='linear'
scores = clf.fit(data)                    # data: [T, n_feats]; scores: [T]
```

GPU smoke test: `sbatch submit_autocegar_smoke.sh` (runs plain RW-1 vs
RW-1+CEGAR on Genesis).

## Provenance

- `deepant/cnn.py`, `rw/cnn_uns.py`, `rw/cnn_rw.py` copied from the `TSB-AD` repo
  models (`TSB_AD/models/{CNN,CNN_uns,CNN_RW}.py`), imports repointed to
  `tsb_common/`.
- CEGAR gate machinery (`autocegar/{scale_grad,gate,controllers,residual_signals}.py`)
  extracted from the Auto-CEGAR repo.
- The pre-restructure standalone track (old vendored DeepAnT package, `rw/`
  trainers, `cegar/`, `run_all_tsb.py`, results, `tsb_ad_models/` version
  history) is preserved under `backup/pre_restructure_2026-07-01/`.
