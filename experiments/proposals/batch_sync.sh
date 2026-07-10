#!/bin/bash
# Gentle, sequential wandb sync of the corrected-config offline runs.
# Kills any running sync first, then syncs in small `nice` batches so the login
# node stays responsive. Idempotent: already-synced runs are skipped on re-run.
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
set -a && source .env 2>/dev/null && set +a
export WANDB_SILENT=true

BATCH=${1:-40}
mapfile -t DIRS < <(ls -d wandb/offline-run-20260709_* wandb/offline-run-20260710_* 2>/dev/null)
echo "total offline dirs: ${#DIRS[@]}, batch=$BATCH  $(date)"
n=0
for ((s=0; s<${#DIRS[@]}; s+=BATCH)); do
  chunk=("${DIRS[@]:s:BATCH}")
  n=$((n+1))
  echo "=== batch $n : ${#chunk[@]} runs ($(date +%H:%M:%S)) ==="
  nice -n 19 wandb sync --include-offline "${chunk[@]}" 2>&1 | grep -cE "Synced|already" | xargs echo "  synced/skipped:"
  sleep 15
done
echo "ALL BATCHES DONE $(date)"
