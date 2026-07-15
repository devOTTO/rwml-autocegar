#!/bin/bash
#SBATCH --job-name=p5tune
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32000M
#SBATCH --time=01:30:00
#SBATCH --array=1-504%16
#SBATCH --output=logs/p5tune_%A_%a.out
#SBATCH --error=logs/p5tune_%A_%a.err
# P5 hyperparameter tuning sweep (l1_weight x lambda x 200ep) over the 7 collections,
# on the same series as the RW-1 best-HP baseline. wandb group = proposal5_tune
# (offline on compute nodes; sync from the login node afterwards).
mkdir -p logs
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
export WANDB_MODE=offline
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" experiments/proposals/runs/p5_tune_grid.txt)
echo "[p5tune ${SLURM_ARRAY_TASK_ID}] $LINE"
eval "python run_proposal.py ${LINE}"
echo "Done ${SLURM_ARRAY_TASK_ID}: $(date)"
