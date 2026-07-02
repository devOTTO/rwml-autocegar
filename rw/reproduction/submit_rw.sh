#!/bin/bash
#SBATCH --job-name=EXP-E-RW-full
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=63000M
#SBATCH --time=48:00:00
#SBATCH --output=logs/EXP-E-RW-full_%j.out
#SBATCH --error=logs/EXP-E-RW-full_%j.err

# EXP-E (RW half): full 180-dataset run of CNN_UNS (RW, Algorithm 1) using
# the algorithm-faithful rewrite (RMSprop, RMSE, epoch-wise, no savgol).
# Writes to a NEW results dir so EXP-A's (wrong-algorithm) reference run in
# eval/metrics/multi/ is preserved untouched for comparison.

mkdir -p logs

source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/TSB-AD

echo "Job ID: $SLURM_JOB_ID"
echo "Experiment: EXP-E algo-faithful-full-run (RW half)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start: $(date)"
echo "---"

dataset_dir="/ocean/projects/cis260190p/yhwang2/data/TSB-AD-M/TSB-AD-M/"

for dataset in "$dataset_dir"*.csv; do
    filename=$(basename "$dataset")
    echo "Processing: $filename"
    python -m benchmark_exp.Run_Detector_M \
        --AD_Name CNN_UNS \
        --dataset_dir "$dataset_dir" \
        --filename "$filename" \
        --save True \
        --save_dir eval/metrics/multi_algofaithful/ \
        2>&1 | awk -F'[][/]' '!/^Epoch \[/ {print; next} ($2==1 || $2%20==0) {print}'
    echo "Done: $filename"
    echo "---"
done

echo "End: $(date)"
