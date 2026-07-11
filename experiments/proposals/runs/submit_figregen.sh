#!/bin/bash
#SBATCH --job-name=figregen
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32000M
#SBATCH --time=00:40:00
#SBATCH --array=1-41%8
#SBATCH --output=logs/figregen_%A_%a.out
#SBATCH --error=logs/figregen_%A_%a.err
# Regenerate all 41 correction-example figures at larger fonts / dpi 200.
# wandb disabled -> only the local PNGs under figures/ are refreshed (no new runs).
mkdir -p logs
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
export WANDB_MODE=disabled
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" experiments/proposals/runs/figregen_grid.txt)
echo "[figregen ${SLURM_ARRAY_TASK_ID}] $LINE"
eval "python experiments/proposals/plot_correction_example.py ${LINE}"
echo "Done ${SLURM_ARRAY_TASK_ID}: $(date)"
