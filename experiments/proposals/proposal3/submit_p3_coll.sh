#!/bin/bash
#SBATCH --job-name=p3-coll
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32000M
#SBATCH --time=00:40:00
#SBATCH --array=1-30%6
#SBATCH --output=logs/p3-coll_%A_%a.out
#SBATCH --error=logs/p3-coll_%A_%a.err
# Collection-level P3 (verdict set): one task per (series x variant). 10 verdict
# series x {full, preserve_only, full@auto-lambda} = 30 tasks. No --baseline (reuse
# reproduction RW-1 per-collection means). wandb offline; sync after.
mkdir -p logs
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
export WANDB_MODE=offline
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" experiments/proposals/proposal3/p3_coll_grid.txt)
echo "[p3-coll ${SLURM_ARRAY_TASK_ID}] $LINE"
eval "python run_proposal.py ${LINE}"
echo "Done ${SLURM_ARRAY_TASK_ID}: $(date)"
