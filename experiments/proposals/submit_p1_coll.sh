#!/bin/bash
#SBATCH --job-name=p1-coll
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32000M
#SBATCH --time=00:40:00
#SBATCH --array=1-10%6
#SBATCH --output=logs/p1-coll_%A_%a.out
#SBATCH --error=logs/p1-coll_%A_%a.err
# Collection-level P1: one task per series (verdict collections). No --baseline
# (reuse the reproduction RW-1 per-collection means). wandb offline; sync after.
mkdir -p logs
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
export WANDB_MODE=offline
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" experiments/proposals/p1_coll_grid.txt)
echo "[p1-coll ${SLURM_ARRAY_TASK_ID}] $LINE"
eval "python run_proposal.py ${LINE}"
echo "Done ${SLURM_ARRAY_TASK_ID}: $(date)"
