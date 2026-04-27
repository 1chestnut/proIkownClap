#!/usr/bin/env bash
set -euo pipefail

MODULE_PATH="$1"
METHOD="$2"
GPU_ID="$3"
OUTPUT_JSON="$4"

source /home/star/anaconda3/etc/profile.d/conda.sh
conda activate zkx
export LD_LIBRARY_PATH="/home/star/anaconda3/envs/zkx/lib:${LD_LIBRARY_PATH:-}"

python /data/zkx/zkx/iknow-audio/54.structure_methods/structure54_runner.py \
  --module-path "$MODULE_PATH" \
  --method "$METHOD" \
  --gpu-id "$GPU_ID" \
  --output-json "$OUTPUT_JSON"
