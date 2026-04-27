#!/usr/bin/env bash
set -euo pipefail

MODULE_PATH="$1"
GPU_ID="$2"
OUTPUT_JSON="$3"

source /home/star/anaconda3/etc/profile.d/conda.sh
conda activate zkx
export LD_LIBRARY_PATH="/home/star/anaconda3/envs/zkx/lib:${LD_LIBRARY_PATH:-}"

python /data/zkx/zkx/iknow-audio/53.negative_hardmax/negative53_runner.py \
  --module-path "$MODULE_PATH" \
  --gpu-id "$GPU_ID" \
  --output-json "$OUTPUT_JSON"
