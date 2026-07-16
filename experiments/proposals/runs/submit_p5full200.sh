#!/bin/bash
#SBATCH --job-name=p5full200
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32000M
#SBATCH --time=01:30:00
#SBATCH --array=1-1000%16
#SBATCH --output=logs/p5full200_%A_%a.out
#SBATCH --error=logs/p5full200_%A_%a.err
# P5 full-benchmark: cr x l1 sweep on ALL 199 TSB-AD-M series (2388 lines), submitted in
# chunks of 1000 via an offset arg: sbatch submit_p5full200.sh 0|1000|2000
# group p5_full200_besthp; CSV results_p5_p5_full200_besthp.csv
mkdir -p logs
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
export WANDB_MODE=offline
GRID=experiments/proposals/runs/p5_full200_besthp_grid.txt
LN=$(( ${1:-0} + SLURM_ARRAY_TASK_ID ))
TOTAL=$(wc -l < "$GRID")
if [ "$LN" -gt "$TOTAL" ]; then echo "skip line $LN > $TOTAL"; exit 0; fi
LINE=$(sed -n "${LN}p" "$GRID")
echo "[p5full200 line ${LN}] $LINE"
eval "python run_proposal.py ${LINE}"
echo "Done line ${LN}: $(date)"
