#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib:
cd /data/zkx/zkx/iknow-audio/25.gmu_residual/tut2017
/home/star/anaconda3/envs/zkx/bin/python -u test_gmu_residual.py 2>&1 | tee tut2017_gmu_residual.log
