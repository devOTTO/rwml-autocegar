# AutoCEGAR — validation evidence

AutoCEGAR (`autocegar/rw_cegar.py`, `CNN_RW_CEGAR`) is a **novel extension**, not
a paper reproduction — there is no paper number to match. What is validated here
is that it *runs end-to-end* on top of the reproduced RW-1 and behaves sanely.

## GPU smoke test (Genesis, 50 epochs)

`smoke_result.txt` (job 41884833, `submit_autocegar_smoke.sh` at repo root):

| model | AUC-PR | AUC-ROC |
|---|---|---|
| RW-1 (plain) | 0.0139 | 0.738 |
| RW-1 + CEGAR | 0.0146 | 0.739 |

Confirms: `CNN_RW_CEGAR` trains without error, produces a valid score, and the
CEGAR gate is a light perturbation of RW-1 (as expected). Absolute numbers are
low only because this is a 50-epoch smoke run on one dataset.

> The `E_t` / `C_t` formulas and `tau` / `k` are still PLACEHOLDERS pending
> Luis's notebook, so the tiny plain-vs-CEGAR gap is not yet meaningful — the
> point of this test is the wiring, not the score.

To run: `sbatch submit_autocegar_smoke.sh` (from repo root).

## Wiring implemented 2026-07-03 (verified to run, not yet evaluated)

`CNN_RW_CEGAR` now wires the previously-scaffolded components (all default to
the old behaviour; opt in via kwargs):

- **gate() + ScaleGrad** — the gate is applied to the per-window RMSE *gradient*
  via `ScaleGrad` (identity forward), so the reported loss is the true RMSE and
  only the correction gradient of gated windows is up-weighted. Replaces the old
  loss-value multiplication (gradient-equivalent, cleaner).
- **auto-λ** (`lam_mode='auto_tr'`) — `tail_ratio_lambda_controller` sets λ each
  epoch from the gate tail ratio; does **not** depend on the E_t/C_t formulas.
- **auto-τ** (`tau_mode='auto_q_valley'`) — `valley_quantile_controller` wired
  with cross-epoch state; a no-op while C_t is a scalar placeholder (a degenerate
  confidence histogram has no valley) and becomes active once C_t is per-window.
- **warm-up curriculum** (`warmup_epochs`, `correction_init`) — N epochs of
  forecaster-only training (no correction/gate) before the gate switches on;
  residual stats keep warming during warm-up. Use `correction_init='zero'` for a
  continuous transition.
- Controller state (`_lam_smooth`, `_tau_q`) carries across epochs **and** the
  warm-up→main boundary (phase continuity).

Verified end-to-end on CPU (synthetic series, 3 configs: default / warmup+zero /
full-auto) — all produce finite scores and the controllers move λ, τ. **Still
pending: (A)** final E_t/C_t/τ/k from Luis's notebook, then a GPU re-run of the
smoke + full 16-collection quantitative evaluation.
