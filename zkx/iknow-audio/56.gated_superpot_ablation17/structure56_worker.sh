#!/usr/bin/env bash
set -euo pipefail

RUNNER="$1"
MODULE_PATH="$2"
GPU_ID="$3"
OUTPUT_JSON="$4"
LOG_PATH="$5"

source /home/star/anaconda3/etc/profile.d/conda.sh
conda activate zkx
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python "${RUNNER}" \
  --module-path "${MODULE_PATH}" \
  --gpu-id "${GPU_ID}" \
  --output-json "${OUTPUT_JSON}" \
  >> "${LOG_PATH}" 2>&1
