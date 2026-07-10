#!/bin/bash
#SBATCH --job-name=plotext
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32000M
#SBATCH --time=00:50:00
#SBATCH --array=1-15%8
#SBATCH --output=logs/plotext_%A_%a.out
#SBATCH --error=logs/plotext_%A_%a.err
mkdir -p logs
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
export WANDB_MODE=offline
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" experiments/proposals/runs/plotext_grid.txt)
echo "[plotext ${SLURM_ARRAY_TASK_ID}] $LINE"
eval "python experiments/proposals/plot_correction_example.py ${LINE}"
echo "Done ${SLURM_ARRAY_TASK_ID}: $(date)"
