#!/bin/bash

# Ablation script: disable experience replay for ALL supported datasets in mainRL_time0.py
# Usage: bash rl_run_ablation_all.sh
# Optionally edit GPUS array below to match available devices.

# List of datasets with batch sizes defined in mainRL_time0.py
#DATASETS=(Chinatown ECG200 GunPoint GunPointAgeSpan GunPointMaleVersusFemale GunPointOldVersusYoung TwoLeadECG)

DATASETS=(FordA FordB FreezerRegularTrain HandOutlines Wafer)
# GPUs to cycle through (edit as needed)
GPUS=(0 1 2 3 4 5 6 7)

# Output log directory (separate from single Coffee ablation)
LOG_DIR="Log_ablation_no_replay_all"
mkdir -p "${LOG_DIR}"

# Optional: limit number of simultaneous background jobs (0 = unlimited)
MAX_PARALLEL=0

running_jobs() { jobs -p | wc -l | tr -d ' '; }

for i in "${!DATASETS[@]}"; do
  DATASET=${DATASETS[$i]}
  GPU=${GPUS[$i % ${#GPUS[@]}]}
  export CUDA_VISIBLE_DEVICES=$GPU

  # Respect MAX_PARALLEL if set > 0
  if [ "$MAX_PARALLEL" -gt 0 ]; then
    while [ "$(running_jobs)" -ge "$MAX_PARALLEL" ]; do
      echo "Waiting for free slot... (current: $(running_jobs), limit: $MAX_PARALLEL)"
      sleep 5
    done
  fi

  echo "[Ablation no_replay] Starting dataset $DATASET on GPU $GPU"
  nohup python3 mainRL_time0.py --dataset "$DATASET" --no_replay > "${LOG_DIR}/${DATASET}_gpu${GPU}.log" 2>&1 &

done

echo "Launched ${#DATASETS[@]} ablation runs (experience replay disabled). Logs: ${LOG_DIR}/*.log"
echo "Monitor with: tail -f ${LOG_DIR}/<dataset>_gpu<gpu>.log"

