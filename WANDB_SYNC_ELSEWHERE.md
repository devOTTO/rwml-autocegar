# Syncing wandb offline runs from a machine with internet

The PSC login node is overloaded and compute nodes have no internet, so the
809 offline runs were archived here (`wandb_offline_runs.tar.gz`, 26 MB) and
committed to git. Sync them from any machine that has internet (e.g. a laptop):

```bash
git pull
tar -xzf wandb_offline_runs.tar.gz                 # -> wandb/offline-run-*
pip install wandb && wandb login                   # once
wandb sync --include-offline wandb/offline-run-*    # uploads to wandb.ai

# move the corrected-config runs (correction_init=neg_x) into their own group
python experiments/proposals/regroup_corrected.py --apply
```

Notes
- `wandb sync` is idempotent per run dir — re-running skips already-synced runs.
- Do NOT sync from the PSC login node while it is loaded; that is what this
  archive is meant to avoid.
- The report itself (tables + figures) is already in `experiments/proposals/docs/`
  and `figures/`; wandb is a bonus dashboard, not required for the results.
