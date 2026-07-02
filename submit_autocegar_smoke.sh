#!/bin/bash
#SBATCH --job-name=autocegar-smoke
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260190p
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32000M
#SBATCH --time=00:20:00
#SBATCH --output=logs/autocegar-smoke_%j.out
#SBATCH --error=logs/autocegar-smoke_%j.err
mkdir -p logs
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
python autocegar_smoke.py
echo "Done: $(date)"
