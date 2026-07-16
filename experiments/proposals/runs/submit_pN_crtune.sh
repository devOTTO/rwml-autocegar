#!/bin/bash
#SBATCH --job-name=pNcrtune
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32000M
#SBATCH --time=02:00:00
#SBATCH --array=1-504%16
#SBATCH --output=logs/pNcrtune_%A_%a.out
#SBATCH --error=logs/pNcrtune_%A_%a.err
# P{N} correction_rate x l1_weight re-ranking sweep. Usage: sbatch submit_pN_crtune.sh <N>
# group proposal{N}_crtune; CSV results_p{N}_proposal{N}_crtune.csv
mkdir -p logs
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
export WANDB_MODE=offline
N=${1:?usage: sbatch submit_pN_crtune.sh <proposal N>}
GRID=experiments/proposals/runs/p${N}_crtune_grid.txt
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$GRID")
echo "[p${N}crtune ${SLURM_ARRAY_TASK_ID}] $LINE"
eval "python run_proposal.py ${LINE}"
echo "Done ${SLURM_ARRAY_TASK_ID}: $(date)"
