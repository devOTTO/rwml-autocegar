#!/bin/bash
#SBATCH --job-name=l1sweep-rw1
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32000M
#SBATCH --time=01:00:00
#SBATCH --output=/ocean/projects/cis260190p/yhwang2/rwml-autocegar/experiments/exp_e_algofaithful/l1sweep_%j.out
#SBATCH --error=/ocean/projects/cis260190p/yhwang2/rwml-autocegar/experiments/exp_e_algofaithful/l1sweep_%j.err

# Does weakening the L1 penalty stop `correction` from collapsing to 0, so
# RW-1 (linear, Algorithm 2) holds its early paper-level AUC to the last
# epoch? Sweep L1 weight on Genesis (small, clearest signal: paper 0.954,
# collapses from 0.891@ep7 to 0.286@ep200 at l1_weight=1.0).

source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
set -a; source .env; set +a
BASE=/ocean/projects/cis260190p/yhwang2/.wandb_scratch
mkdir -p "$BASE"/{cache,data,artifacts,tmp}
export WANDB_CACHE_DIR="$BASE/cache" WANDB_DATA_DIR="$BASE/data" \
       WANDB_ARTIFACT_DIR="$BASE/artifacts" TMPDIR="$BASE/tmp"

DIAG=experiments/exp_e_algofaithful/diag_ghl_freeze.py
GEN=001_Genesis_id_1_Sensor_tr_4055_1st_15538.csv

echo "Job $SLURM_JOB_ID | $(date)"
for L1 in 1.0 0.1 0.01 0.001 0.0; do
  echo "===== RW-1 linear | l1_weight=$L1 | Genesis ====="
  python "$DIAG" --method RW-1 --filename "$GEN" --epochs 200 \
      --activation linear --l1_weight "$L1" --wandb
done
echo "Done: $(date)"
