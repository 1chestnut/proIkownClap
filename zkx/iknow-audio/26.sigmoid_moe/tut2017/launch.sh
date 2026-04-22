#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib:
cd /data/zkx/zkx/iknow-audio/26.sigmoid_moe/tut2017
/home/star/anaconda3/envs/zkx/bin/python -u test_sigmoid_moe.py 2>&1 | tee tut2017_sigmoid_moe.log
