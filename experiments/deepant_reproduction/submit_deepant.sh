#!/bin/bash
#SBATCH --job-name=deepant-timeeval
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=63000M
#SBATCH --time=48:00:00
#SBATCH --output=logs/deepant-timeeval_%j.out
#SBATCH --error=logs/deepant-timeeval_%j.err

mkdir -p logs

source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/TimeEval-algorithms

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo "---"

python exp_deepant_all.py

echo "---"
echo "End: $(date)"
