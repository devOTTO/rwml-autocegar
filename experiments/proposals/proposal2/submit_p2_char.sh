#!/bin/bash
#SBATCH --job-name=p2-char
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32000M
#SBATCH --time=02:00:00
#SBATCH --array=1-3%3
#SBATCH --output=logs/p2-char_%A_%a.out
#SBATCH --error=logs/p2-char_%A_%a.err
#
# Proposal-2 CHARACTERIZATION grid (see experiments/proposals/proposal2/p2_char_grid.txt).
# NOT the fail-fast verdict — tests the hypothesis "P2 helps where anomalies are
# genuinely uncertain (satellite telemetry), is neutral on industrial (SMD), and
# hurts on periodic signals (ECG/MITDB, like gecco)". Each task = P2 mc5 + RW-1
# baseline on one new-domain dataset. MITDB (ECG, 336k rows) is the slow one ->
# 2h limit. wandb offline; sync afterward.
#
mkdir -p logs
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
export WANDB_MODE=offline
GRID=experiments/proposals/proposal2/p2_char_grid.txt
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$GRID")
echo "[char task ${SLURM_ARRAY_TASK_ID}] python run_proposal.py ${LINE}"
eval "python run_proposal.py ${LINE}"
echo "Done task ${SLURM_ARRAY_TASK_ID}: $(date)"
