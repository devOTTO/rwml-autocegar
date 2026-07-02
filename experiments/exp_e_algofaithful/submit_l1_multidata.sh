#!/bin/bash
#SBATCH --job-name=l1-multidata-rw1
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32000M
#SBATCH --time=02:00:00
#SBATCH --output=/ocean/projects/cis260190p/yhwang2/rwml-autocegar/experiments/exp_e_algofaithful/l1multi_%j.out
#SBATCH --error=/ocean/projects/cis260190p/yhwang2/rwml-autocegar/experiments/exp_e_algofaithful/l1multi_%j.err

# Pick a robust L1 weight for RW-1 (linear, Algorithm 2): does λ≈0.001-0.01
# recover the freeze datasets WITHOUT regressing the ones RW-1 already
# matched? Runs each dataset at λ in {1.0 baseline, 0.01, 0.001}.
#   freeze/recover:  Genesis (paper ROC 0.954), GECCO (0.979), GHL (0.564)
#   controls:        SMD (0.790), Exathlon (0.981)  <- must NOT regress

source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
set -a; source .env; set +a
BASE=/ocean/projects/cis260190p/yhwang2/.wandb_scratch
mkdir -p "$BASE"/{cache,data,artifacts,tmp}
export WANDB_CACHE_DIR="$BASE/cache" WANDB_DATA_DIR="$BASE/data" \
       WANDB_ARTIFACT_DIR="$BASE/artifacts" TMPDIR="$BASE/tmp"

DIAG=experiments/exp_e_algofaithful/diag_ghl_freeze.py
FILES=(
  001_Genesis_id_1_Sensor_tr_4055_1st_15538.csv
  173_GECCO_id_1_Sensor_tr_16165_1st_16265.csv
  032_GHL_id_1_Sensor_tr_50000_1st_65001.csv
  057_SMD_id_1_Facility_tr_4529_1st_4629.csv
  174_Exathlon_id_1_Facility_tr_10766_1st_12590.csv
)
echo "Job $SLURM_JOB_ID | $(date)"
for FILE in "${FILES[@]}"; do
  for L1 in 1.0 0.01 0.001; do
    echo "===== RW-1 linear | l1=$L1 | $FILE ====="
    python "$DIAG" --method RW-1 --filename "$FILE" --epochs 200 \
        --activation linear --l1_weight "$L1" --wandb
  done
done
echo "Done: $(date)"
