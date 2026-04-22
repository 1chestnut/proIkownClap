#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib:
cd /data/zkx/zkx/iknow-audio/26.sigmoid_moe/esc50
/home/star/anaconda3/envs/zkx/bin/python -u test_sigmoid_moe.py 2>&1 | tee esc50_sigmoid_moe.log
