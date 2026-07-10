#!/bin/bash
#SBATCH --job-name=corr-plots
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32000M
#SBATCH --time=01:30:00
#SBATCH --output=logs/corr-plots_%j.out
#SBATCH --error=logs/corr-plots_%j.err
# Correction-example figures for every tested representative: P1 (basic) + P2 (mc5).
# 3-panel each (anomaly structure / original vs corrected / gate+|correction|).
# wandb offline; synced at the end.
mkdir -p logs
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
export WANDB_MODE=offline

echo "===== P1 (basic) correction plots: $(date) ====="
python experiments/proposals/plot_correction_example.py --proposal 1 --dataset all --variant basic --wandb

echo "===== P2 (mc5) correction plots: $(date) ====="
python experiments/proposals/plot_correction_example.py --proposal 2 --dataset all --variant mc5 --wandb

echo "===== syncing wandb: $(date) ====="
wandb sync --include-offline ./wandb/offline-run-* 2>&1 | tail -3 || true
echo "===== PLOTS DONE: $(date) ====="
ls -la experiments/proposals/figures/
