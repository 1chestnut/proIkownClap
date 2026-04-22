#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib:
cd /data/zkx/zkx/iknow-audio/24.branch_lse/tut2017
/home/star/anaconda3/envs/zkx/bin/python -u test_branch_lse.py 2>&1 | tee tut2017_branch_lse.log
