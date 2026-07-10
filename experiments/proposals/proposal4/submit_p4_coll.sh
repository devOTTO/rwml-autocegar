#!/bin/bash
#SBATCH --job-name=p4-coll
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32000M
#SBATCH --time=00:50:00
#SBATCH --array=1-20%6
#SBATCH --output=logs/p4-coll_%A_%a.out
#SBATCH --error=logs/p4-coll_%A_%a.err
mkdir -p logs
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
export WANDB_MODE=offline
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" experiments/proposals/proposal4/p4_coll_grid.txt)
echo "[p4-coll ${SLURM_ARRAY_TASK_ID}] $LINE"
eval "python run_proposal.py ${LINE}"
echo "Done ${SLURM_ARRAY_TASK_ID}: $(date)"
