#!/usr/bin/env bash
set -e
cd /data/zkx/zkx/iknow-audio/30.repulsive_rest/fsd50
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib:4LD_LIBRARY_PATH
/home/star/anaconda3/envs/zkx/bin/python -u fsd29_repulsive.py 2>&1 | tee fsd50_repulsive.log
