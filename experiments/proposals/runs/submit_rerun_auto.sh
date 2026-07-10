#!/bin/bash
#SBATCH --job-name=rerun-auto
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32000M
#SBATCH --time=00:50:00
#SBATCH --array=1-50%12
#SBATCH --output=logs/rerun-auto_%A_%a.out
#SBATCH --error=logs/rerun-auto_%A_%a.err
# Fresh re-run of P1-P5 on the verdict set under the corrected config:
#   warm-up = plain RW-1 (gate off) then gate on; correction_init = neg_x (RW-1-faithful).
# 5 proposals x 10 verdict series (default variant) = 50 tasks.
mkdir -p logs
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
export WANDB_MODE=offline
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" experiments/proposals/runs/rerun_auto_grid.txt)
echo "[rerun-auto ${SLURM_ARRAY_TASK_ID}] $LINE"
eval "python run_proposal.py ${LINE}"
echo "Done ${SLURM_ARRAY_TASK_ID}: $(date)"
