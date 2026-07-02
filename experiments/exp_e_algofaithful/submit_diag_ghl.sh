#!/bin/bash
#SBATCH --job-name=diag-ghl-freeze
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32000M
#SBATCH --time=01:00:00
#SBATCH --output=/ocean/projects/cis260190p/yhwang2/rwml-autocegar/experiments/exp_e_algofaithful/diag_ghl_%j.out
#SBATCH --error=/ocean/projects/cis260190p/yhwang2/rwml-autocegar/experiments/exp_e_algofaithful/diag_ghl_%j.err

# Diagnostic on ONE GHL file (032, worst-inverting for both RW and RW-1):
# instrument the correction dynamics per epoch to find where the AUC-ROC<<0.5
# score inversion comes from. Runs BOTH methods for direct comparison.
# Keep GPU/SU cost minimal — single dataset, ~30 min total expected.

source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar

# W&B scratch on /ocean (NOT /jet/home — hard quota). See exp README gotcha.
set -a; source .env; set +a
BASE=/ocean/projects/cis260190p/yhwang2/.wandb_scratch
mkdir -p "$BASE"/{cache,data,artifacts,tmp}
export WANDB_CACHE_DIR="$BASE/cache" WANDB_DATA_DIR="$BASE/data" \
       WANDB_ARTIFACT_DIR="$BASE/artifacts" TMPDIR="$BASE/tmp"

DIAG=experiments/exp_e_algofaithful/diag_ghl_freeze.py
FILE=032_GHL_id_1_Sensor_tr_50000_1st_65001.csv

echo "Job $SLURM_JOB_ID | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) | $(date)"
echo "=== RW-1 (CNN_RW, ReLU-gated) ==="
python "$DIAG" --method RW-1 --filename "$FILE" --epochs 200 --wandb
echo "=== RW (CNN_uns, linear) ==="
python "$DIAG" --method RW   --filename "$FILE" --epochs 200 --wandb
echo "Done: $(date)"
