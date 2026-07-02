# DeepAnT — reproduction evidence

Baseline reproduction of the paper's **DeepAnT** on TSB-AD-M, using TSB-AD's
`CNN` detector (`deepant/cnn.py` here = `TSB_AD/models/CNN.py`).

| metric | ours | paper (Table 6.1) |
|---|---|---|
| AUC-PR (mean) | **0.350** | 0.33 |
| AUC-ROC (mean) | **0.770** | 0.81 |

n = 199 result rows (180 files). ✅ Reproduces the paper's DeepAnT.

## Files
- `summary_per_dataset.csv` — AUC-PR/ROC/VUS per dataset (199 rows).
- `summary_per_family.csv` — mean AUC-PR/ROC per dataset family.
- `submit_cnn.sh` — the exact SLURM script used (runs
  `benchmark_exp.Run_Detector_M --AD_Name CNN` over all TSB-AD-M files).

## How it was run
Inside the TSB-AD repo (`/ocean/projects/cis260190p/yhwang2/TSB-AD`):
```
sbatch submit_cnn.sh   # → eval/metrics/multi/CNN/*.csv
```
Config: `window=50, num_channel=[32,32,40], epochs=50, lr=8e-4, batch=128`.
