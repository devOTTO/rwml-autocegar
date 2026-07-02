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
