#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib:
cd /data/zkx/zkx/iknow-audio/36.margin_backoff/tut2017
/home/star/anaconda3/envs/zkx/bin/python -u test_margin_backoff.py 2>&1 | tee tut2017_margin_backoff.log
