#!/usr/bin/env bash
set -euo pipefail
cd /data/zkx/zkx/iknow-audio/37.oracle_profile/tut2017
export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib:
/home/star/anaconda3/envs/zkx/bin/python -u test_oracle_profile.py 2>&1 | tee tut2017_oracle_profile.log
