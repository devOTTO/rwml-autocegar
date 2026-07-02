#!/bin/bash
#SBATCH --job-name=val-rw1-linear
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32000M
#SBATCH --time=01:00:00
#SBATCH --output=/ocean/projects/cis260190p/yhwang2/rwml-autocegar/experiments/exp_e_algofaithful/val_rw1_%j.out
#SBATCH --error=/ocean/projects/cis260190p/yhwang2/rwml-autocegar/experiments/exp_e_algofaithful/val_rw1_%j.err

# Validate the Algorithm-2 fix for RW-1: does removing the (non-algorithmic)
# ReLU gate un-freeze the correction and recover the paper's results on the
# freeze datasets? Runs RW-1 with relu (old, baseline) vs linear (Alg-2 fix)
# on the three worst-inverting overlaps. Paper AUC-ROC targets:
#   Genesis 0.954 | GECCO 0.979 | GHL 0.564   (we froze at 0.137/0.611/0.056)
# Small datasets -> minimal SU.

source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
set -a; source .env; set +a
BASE=/ocean/projects/cis260190p/yhwang2/.wandb_scratch
mkdir -p "$BASE"/{cache,data,artifacts,tmp}
export WANDB_CACHE_DIR="$BASE/cache" WANDB_DATA_DIR="$BASE/data" \
       WANDB_ARTIFACT_DIR="$BASE/artifacts" TMPDIR="$BASE/tmp"

DIAG=experiments/exp_e_algofaithful/diag_ghl_freeze.py
GEN=001_Genesis_id_1_Sensor_tr_4055_1st_15538.csv
GEC=173_GECCO_id_1_Sensor_tr_16165_1st_16265.csv
GHL=032_GHL_id_1_Sensor_tr_50000_1st_65001.csv

echo "Job $SLURM_JOB_ID | $(date)"
for FILE in "$GEN" "$GEC" "$GHL"; do
  for ACT in relu linear; do
    echo "===== RW-1 | $ACT | $FILE ====="
    python "$DIAG" --method RW-1 --filename "$FILE" --epochs 200 --activation "$ACT" --wandb
  done
done
echo "Done: $(date)"
