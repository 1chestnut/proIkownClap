#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib:${LD_LIBRARY_PATH:-}
cd /data/zkx/zkx/iknow-audio/38.oracle_guided_candidate/esc50
rm -f esc50_oracle_guided_candidate.log
CUDA_VISIBLE_DEVICES=0 /home/star/anaconda3/envs/zkx/bin/python -u test_oracle_guided_candidate.py 2>&1 | tee esc50_oracle_guided_candidate.log
