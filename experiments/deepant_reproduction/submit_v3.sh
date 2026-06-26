#!/bin/bash
#SBATCH --job-name=deepant-v3
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=63000M
#SBATCH --time=48:00:00
#SBATCH --output=logs/deepant-v3_%j.out
#SBATCH --error=logs/deepant-v3_%j.err

mkdir -p logs

source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/TimeEval-algorithms

echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start: $(date)"
echo "---"

python run_cnn_normal_all.py

echo "---"
echo "End: $(date)"
