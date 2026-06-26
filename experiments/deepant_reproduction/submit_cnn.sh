#!/bin/bash
#SBATCH --job-name=tsb-cnn
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=63000M
#SBATCH --time=48:00:00
#SBATCH --output=logs/tsb-cnn_%j.out
#SBATCH --error=logs/tsb-cnn_%j.err

mkdir -p logs

source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/TSB-AD

echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start: $(date)"
echo "---"

dataset_dir="/ocean/projects/cis260190p/yhwang2/data/TSB-AD-M/TSB-AD-M/"

for dataset in "$dataset_dir"*.csv; do
    filename=$(basename "$dataset")
    echo "Processing: $filename"
    python -m benchmark_exp.Run_Detector_M \
        --AD_Name CNN \
        --dataset_dir "$dataset_dir" \
        --filename "$filename" \
        --save True \
        2>&1
    echo "Done: $filename"
    echo "---"
done

echo "End: $(date)"
