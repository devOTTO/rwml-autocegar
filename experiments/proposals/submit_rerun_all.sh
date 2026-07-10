#!/bin/bash
#SBATCH --job-name=rerun-all
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32000M
#SBATCH --time=00:50:00
#SBATCH --array=1-100%12
#SBATCH --output=logs/rerun-all_%A_%a.out
#SBATCH --error=logs/rerun-all_%A_%a.err
# Fresh consistent re-run of P1-P5 under BOTH configs on the verdict set:
#   cfgA = zero-init + warmup 10 (current)   cfgB = neg_x-init + warmup 0 (thesis-faithful)
# 5 proposals x 2 configs x 10 verdict series = 100 tasks. Decide which config to keep
# after seeing the A-vs-B comparison.
mkdir -p logs
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
export WANDB_MODE=offline
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" experiments/proposals/rerun_all_grid.txt)
echo "[rerun-all ${SLURM_ARRAY_TASK_ID}] $LINE"
eval "python run_proposal.py ${LINE}"
echo "Done ${SLURM_ARRAY_TASK_ID}: $(date)"
