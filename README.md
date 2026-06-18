# rwml-autocegar

Residual-based **CEGAR** adaptation of the **DeepAnT + RW-ML** pipeline.

This repo is **self-contained**: it vendors the DeepAnT forecasting pipeline and
extracts the Auto-CEGAR gate machinery so the two can be combined without fragile
cross-repo imports. It replaces Auto-CEGAR's classification-confidence signals
(`p_y`, `p_max`) with residual-domain equivalents (`E_t`, `C_t`).

> **Scope (Week 5, pre-notebook):** this is structural scaffolding. The exact
> `E_t` / `C_t` formulas and the `tau` / `k` constants are **PLACEHOLDERS**
> pending Luis's confidence/wrongness notebook. Everything is marked with a
> `PLACEHOLDER` comment. The *wiring* (residual -> signals -> gate -> ScaleGrad
> -> loss, plus the tau/lambda controllers) is final.

## Concept

Auto-CEGAR is **gradient rescaling**, not a sample/label rewrite. Per sample it
builds `gate = wrongness x confidence`, turns it into `scale = 1 + lambda*gate`,
and applies it to the model output via `ScaleGrad` (forward identity, backward
`grad * scale`). Only the *gradient* is reweighted; the loss value is unchanged.

The residual translation:

| Auto-CEGAR (classification) | rwml-autocegar (residual AD) |
|---|---|
| `wrong_gate = 1 - p_y` | `E_t = clamp(residual / residual_q95, 1.0)` |
| `conf_gate = sigmoid(k*(conf - tau))` | `C_t = sigmoid(k*(tau - residual_ema))` |
| applied to `logits`, then cross-entropy | applied to `y_hat`, then L1 loss |

```
DeepAnT forward -> residual (y - y_hat)
                -> E_t (wrongness), C_t (confidence)     [cegar/residual_signals.py]
                -> gate = E_t * C_t -> scale = 1+lam*gate [cegar/gate.py]
                -> ScaleGrad on y_hat -> L1 loss          [cegar/scale_grad.py]
controllers (per epoch):
  - tail-ratio lambda  <- gate_mean/gate_p99 EMAs         [cegar/controllers.py]
  - valley-detection tau <- accumulated E_t histogram     [cegar/controllers.py]
```

## Layout

```
rwml-autocegar/
  deepant/              vendored DeepAnT (model, predictor, detector, dataset, ...)
  cegar/
    scale_grad.py       ScaleGrad autograd function (copied verbatim)
    gate.py             gate(confidence, wrongness, lam, scale_normalize)
    controllers.py      tail_ratio_lambda_controller, valley_quantile_controller
    residual_signals.py ResidualStats + E_t/C_t (PLACEHOLDER formulas)
  config.py             DeepAnT + CEGAR + W&B config; init_wandb()
  train_rwml.py         RWMLPredictor: gate-integrated training loop
  eval_msl.py           MSL eval harness (baseline AUC-ROC 0.679 control)
  tests/test_gate.py    gate mechanics tests (+ skipped worked-example placeholder)
  requirements.txt
```

## Usage

Install:

```bash
pip install -r requirements.txt
```

Run tests (no data / GPU needed):

```bash
pytest tests/ -v
```

Baseline MSL control (reproduce ~0.679):

```bash
python eval_msl.py --train-csv <MSL>.train.csv --test-csv <MSL>.test.csv --mode baseline
```

Residual-CEGAR run (after the placeholder formulas are finalized):

```bash
python eval_msl.py --train-csv <MSL>.train.csv --test-csv <MSL>.test.csv --mode cegar
```

W&B logs to entity `yoonmeehwang-carnegie-mellon-university`, project `RWML`
(disable with `--no-wandb`).

## Pending Luis's notebook

- Finalize `compute_wrongness_E_t` / `compute_confidence_C_t` in
  `cegar/residual_signals.py` (normalization, `tau`, `k`).
- Fill `test_worked_example_pending_notebook` in `tests/test_gate.py`
  (value=2, conf=0.9, q95=10 -> sigmoid ~= 0.881 -> no-rewrite).
- Run the full residual-CEGAR MSL comparison vs. the 0.679 baseline.

## Provenance

- DeepAnT vendored from `TimeEval-algorithms/deepant`.
- CEGAR machinery extracted from `autocegar/ecg_loss.py` and `autocegar/train.py`
  (`ecg_loss` gate/scale, `ScaleGrad`, `_ecg_on_epoch_end` valley detection,
  `LossFunction` tail-ratio lambda rule).
