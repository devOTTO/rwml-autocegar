#!/bin/bash
#SBATCH --job-name=p5crtune
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32000M
#SBATCH --time=01:30:00
#SBATCH --array=1-504%16
#SBATCH --output=logs/p5crtune_%A_%a.out
#SBATCH --error=logs/p5crtune_%A_%a.err
# P5 correction_rate x l1_weight sweep (thesis §6.3.2 axis), auto-λ, 200ep.
# wandb group = proposal5_crtune (offline; sync after). CSV: results_p5_proposal5_crtune.csv
mkdir -p logs
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
export WANDB_MODE=offline
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" experiments/proposals/runs/p5_crtune_grid.txt)
echo "[p5crtune ${SLURM_ARRAY_TASK_ID}] $LINE"
eval "python run_proposal.py ${LINE}"
echo "Done ${SLURM_ARRAY_TASK_ID}: $(date)"
