#!/bin/bash
#SBATCH --job-name=p2-grid
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32000M
#SBATCH --time=01:30:00
#SBATCH --array=1-10%6
#SBATCH --output=logs/p2-grid_%A_%a.out
#SBATCH --error=logs/p2-grid_%A_%a.err
#
# Proposal-2 fail-fast grid (see experiments/proposals/proposal2/p2_grid.txt).
# One array task per grid line. P2 = MC-dropout uncertainty confidence, so each
# task is slower than P1 (K extra forward passes/batch) -> 1h30 time limit.
# wandb offline on the compute node; sync afterward:
#     wandb sync --include-offline ./wandb/offline-run-<date>_*
#
mkdir -p logs
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
export WANDB_MODE=offline
GRID=experiments/proposals/proposal2/p2_grid.txt
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$GRID")
echo "[task ${SLURM_ARRAY_TASK_ID}] python run_proposal.py ${LINE}"
eval "python run_proposal.py ${LINE}"
echo "Done task ${SLURM_ARRAY_TASK_ID}: $(date)"
