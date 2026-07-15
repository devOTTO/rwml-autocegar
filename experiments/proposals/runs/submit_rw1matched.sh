#!/bin/bash
#SBATCH --job-name=rw1m
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32000M
#SBATCH --time=01:30:00
#SBATCH --array=1-84%16
#SBATCH --output=logs/rw1m_%A_%a.out
#SBATCH --error=logs/rw1m_%A_%a.err
# Matched-config RW-1 control: plain RW-1 at the proposals' fixed l1_weight=0.001,
# at 100ep and 200ep, on the same 7 collections. De-confounds every delta (esp. SWaT).
# wandb group = proposal5_rw1matched (offline; sync after).
mkdir -p logs
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
export WANDB_MODE=offline
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" experiments/proposals/runs/rw1_matched_grid.txt)
echo "[rw1m ${SLURM_ARRAY_TASK_ID}] $LINE"
eval "python run_proposal.py ${LINE}"
echo "Done ${SLURM_ARRAY_TASK_ID}: $(date)"
