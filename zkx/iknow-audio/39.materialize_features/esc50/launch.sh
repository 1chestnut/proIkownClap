#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib:

a=
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib:
cd /data/zkx/zkx/iknow-audio/39.materialize_features/esc50
rm -f esc50_materialize.log
CUDA_VISIBLE_DEVICES=0 /home/star/anaconda3/envs/zkx/bin/python -u test_materialize.py 2>&1 | tee esc50_materialize.log
