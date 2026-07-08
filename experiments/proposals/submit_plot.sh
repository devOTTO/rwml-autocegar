#!/bin/bash
#SBATCH --job-name=p1-plot
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32000M
#SBATCH --time=00:30:00
#SBATCH --output=logs/p1-plot_%j.out
#SBATCH --error=logs/p1-plot_%j.err
#
# Baldo-Fig-6.1-style original-vs-corrected example plot for one dataset.
# Usage: sbatch experiments/proposals/submit_plot.sh [dataset]   (default gecco)
# PNG -> experiments/proposals/figures/ ; wandb offline (sync afterward).
#
mkdir -p logs
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
export WANDB_MODE=offline
DATASET=${1:-gecco}
echo "plotting dataset=${DATASET}"
python experiments/proposals/plot_correction_example.py --dataset "${DATASET}" --wandb
echo "Done: $(date)"
