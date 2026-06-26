#!/bin/bash
#SBATCH --job-name=deepant-tsb
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=63000M
#SBATCH --time=48:00:00
#SBATCH --output=logs/deepant-tsb_%j.out
#SBATCH --error=logs/deepant-tsb_%j.err

mkdir -p logs

source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo "---"

python run_all_tsb.py

echo "---"
echo "End: $(date)"
