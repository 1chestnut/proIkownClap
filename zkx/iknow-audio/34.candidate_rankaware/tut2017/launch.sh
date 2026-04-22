#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib:${LD_LIBRARY_PATH:-}
cd /data/zkx/zkx/iknow-audio/34.candidate_rankaware/tut2017
/home/star/anaconda3/envs/zkx/bin/python -u test_rankaware_router.py 2>&1 | tee tut2017_rankaware.log