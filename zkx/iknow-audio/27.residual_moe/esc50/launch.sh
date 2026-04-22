#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib:
cd /data/zkx/zkx/iknow-audio/27.residual_moe/esc50
/home/star/anaconda3/envs/zkx/bin/python -u test_residual_moe.py 2>&1 | tee esc50_residual_moe.log
