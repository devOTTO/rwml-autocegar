#!/bin/bash
#SBATCH --job-name=p4smoke
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16000M
#SBATCH --time=00:15:00
#SBATCH --output=logs/p4smoke_%j.out
#SBATCH --error=logs/p4smoke_%j.err
# Quick runtime check of the P4 write-back (gradnorm_wb): tiny series, warmup 2 / epochs 5
# so post-warmup epochs actually exercise _writeback_scale. no-wandb, separate CSV.
mkdir -p logs
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
python run_proposal.py --proposal 4 --variant gradnorm_wb \
  --dataset 116_TAO_id_1_Environment_tr_500_1st_3.csv \
  --correction_rate 0.001 --l1_weight 0.001 --epochs 5 --warmup 2 --lam 1.0 \
  --extra lam_mode=auto_tr --no-wandb --wandb-group p4_smoke
echo "SMOKE DONE $(date)"
