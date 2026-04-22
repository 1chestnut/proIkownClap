#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib:
cd /data/zkx/zkx/iknow-audio/27.residual_moe/tut2017
/home/star/anaconda3/envs/zkx/bin/python -u test_residual_moe.py 2>&1 | tee tut2017_residual_moe.log
