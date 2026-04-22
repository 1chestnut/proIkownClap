#!/usr/bin/env bash
set -euo pipefail
cd /data/zkx/zkx/iknow-audio/29.router/tut2017
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib:
/home/star/anaconda3/envs/zkx/bin/python -u tut29_router.py 2>&1 | tee tut2017_router.log
