#!/usr/bin/env bash
set -euo pipefail
cd /data/zkx/zkx/iknow-audio/29.router/esc50
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib:
/home/star/anaconda3/envs/zkx/bin/python -u esc29_router.py 2>&1 | tee esc50_router.log
