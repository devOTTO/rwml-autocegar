#!/bin/bash
#SBATCH --job-name=p1-grid
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32000M
#SBATCH --time=01:00:00
#SBATCH --array=1-11%6
#SBATCH --output=logs/p1-grid_%A_%a.out
#SBATCH --error=logs/p1-grid_%A_%a.err
#
# Proposal-1 fail-fast grid (see experiments/proposals/runs/p1_grid.txt).
# One array task per line of the grid file. Each task runs run_proposal.py with
# those args and logs to wandb (per-epoch curve + final AUC/delta).
#
# wandb runs OFFLINE on the compute node (Bridges2 GPU nodes may be air-gapped),
# writing to ./wandb. After the array finishes, sync everything to the online
# rwml-autocegar project from the login node:
#     source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
#     cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
#     wandb sync --include-offline ./wandb/offline-run-*
#
mkdir -p logs
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar

export WANDB_MODE=offline   # sync to online project afterward (see header)

GRID=experiments/proposals/runs/p1_grid.txt
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$GRID")
echo "[task ${SLURM_ARRAY_TASK_ID}] python run_proposal.py ${LINE}"
eval "python run_proposal.py ${LINE}"
echo "Done task ${SLURM_ARRAY_TASK_ID}: $(date)"
